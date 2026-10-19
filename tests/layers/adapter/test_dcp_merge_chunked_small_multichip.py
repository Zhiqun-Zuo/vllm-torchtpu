# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TPU_DCP_MERGE_CHUNKED_SMALL (chunk count of the buckets that gather their
lse once) over real collectives, 8 ranks, eager and compiled.

Merge env: the defaults of the chunked merge (8 chunks from 1024 tokens, one
lse all-gather up to 2048 tokens, bf16 reduce-scatter). Arms: off (SMALL=0,
8 chunks everywhere) and on (SMALL=2: buckets of at most 2048 tokens split
into 2 chunks). T=1024 and 2048 take SMALL; T=4096 (8 chunks) and T=16
(unchunked) and the MTP drafter (is_backbone=False, never chunked) must be
bit for bit off.

Splitting the token axis changes no per-row arithmetic, but larger chunks can
move the bf16 reduce-scatter onto the SparseCore offload, which sums in
another order. So on 1024 / 2048 the on arm is held to the bf16 wire's own
error bound against an f64 reference (`RS_BF16_MAX_ULP`, in bf16 ulps of
sum_r w_r |o_r|), not to off bit for bit. Positive control: the on arm with
every rank's lse rows rolled by one row before the gather must exceed it.
"""

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_cp_mla_merge_multichip as mm  # noqa: E402

pytestmark = [pytest.mark.multichip, pytest.mark.spawns_tpu_workers]

WORLD_SIZE = 8
HEADS = 16  # 2 per rank
VDIM = 512
SMALL_TOKENS = (1024, 2048)  # take TPU_DCP_MERGE_CHUNKED_SMALL
OTHER_TOKENS = (4096, 16)  # keep their chunk count
MERGE_ENV = {
    "TPU_DCP_LSE_MERGE_A2A": False,
    "TPU_DCP_RS_OUT_PRECISION": False,
    "TPU_DCP_MERGE_CHUNKED": 8,
    "TPU_DCP_MERGE_CHUNKED_MIN_TOKENS": 1024,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG": True,
    "TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS": 2048,
    "TPU_DCP_MERGE_RS_BF16": 1,
}
RS_BF16_MAX_ULP = mm.RS_BF16_MAX_ULP[1]
LAUNCH_TIMEOUT = 3000


def _set(cp_mla, small):
    for name, value in {**MERGE_ENV, "TPU_DCP_MERGE_CHUNKED_SMALL": small}.items():
        setattr(cp_mla.envs, name, value)


def _rank_partial(rank, tokens):
    """Rank `rank`'s (bf16 partial, f32 lse), the same on every process."""
    import torch

    g = torch.Generator().manual_seed(20261019 + 1000 * rank + tokens)
    o = torch.randn(tokens, HEADS, VDIM, generator=g).to(torch.bfloat16)
    lse = torch.randn(tokens, HEADS, generator=g) * 4.0 + 3.0
    lse[2::7, rank % HEADS] = float("-inf")  # tokens this rank does not own
    return o, lse


def _ulp_diff(a, b):
    import torch

    def line(x):
        x = x.view(torch.int16).to(torch.int32)
        return torch.where(x < 0, -(x & 0x7FFF), x)

    return (line(a) - line(b)).abs()


def _check(rank, world, cp_mla):
    import torch

    report = {}
    real_gather = cp_mla._gather_lse

    def rolled(lse_f32, group):
        return real_gather(torch.roll(lse_f32, 1, dims=0), group)

    for tokens in SMALL_TOKENS + OTHER_TOKENS:
        o, lse = _rank_partial(rank, tokens)
        o_dev, l_dev = o.to("tpu"), lse.to("tpu")
        everyone = None
        if tokens in SMALL_TOKENS:
            parts = [_rank_partial(r, tokens) for r in range(world)]
            everyone = (
                torch.stack([p[0] for p in parts]),
                torch.stack([p[1] for p in parts]),
            )
        for how in ("eager", "compiled"):
            key = f"T{tokens}/{how}"
            try:

                def run(backbone=True, how=how, o_dev=o_dev, l_dev=l_dev):
                    def fn(a, b):
                        return cp_mla.merge_lse_partials_scatter_heads(
                            a, b, is_backbone=backbone
                        )

                    f = mm._compile(fn) if how == "compiled" else fn
                    return f(o_dev, l_dev).cpu()

                c = {"tokens": tokens}
                _set(cp_mla, 0)
                c["off_chunks"] = cp_mla._merge_chunks(tokens)
                off = run()
                _set(cp_mla, 2)
                c["on_chunks"] = cp_mla._merge_chunks(tokens)
                on = run()
                c["ndiff"] = int((_ulp_diff(on, off) > 0).sum())
                c["elements"] = int(off.numel())
                if everyone is not None:
                    outs, lses = everyone
                    c["off_max_ulp"] = mm._scale_ulp_stats(
                        off, outs, lses, rank, world
                    )["max_ulp"]
                    c["on_max_ulp"] = mm._scale_ulp_stats(on, outs, lses, rank, world)[
                        "max_ulp"
                    ]
                    cp_mla._gather_lse = rolled
                    try:
                        bad = run()
                    finally:
                        cp_mla._gather_lse = real_gather
                    c["control_max_ulp"] = mm._scale_ulp_stats(
                        bad, outs, lses, rank, world
                    )["max_ulp"]
                if tokens == 2048:
                    _set(cp_mla, 0)
                    d_off = run(backbone=False)
                    _set(cp_mla, 2)
                    d_on = run(backbone=False)
                    c["drafter_ndiff"] = int((_ulp_diff(d_on, d_off) > 0).sum())
                report[key] = c
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                traceback.print_exc()
                report[key] = {"error": repr(exc)[:600]}
    return report


def _run_worker(result_dir: Path) -> None:
    import torch
    import torch_tpu  # noqa: F401
    from torch import distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    result: dict[str, object] = {"rank": rank}
    try:
        dist.init_process_group(backend="tpu_dist")
        torch.empty((1,), device="tpu").cpu()
        from vllm.distributed.parallel_state import (
            get_world_group,
            init_distributed_environment,
            init_model_parallel_group,
        )

        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="tpu_dist",
        )
        from vllm_torchtpu.distributed import dcp as dcp_mod
        from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

        wg = get_world_group()
        group = init_model_parallel_group(
            [list(range(world))],
            wg.local_rank,
            dist.get_backend(wg.device_group),
            group_name="chunked_small_merge",
        )
        cp_mla.get_dcp_group = lambda: group
        dcp_mod.get_dcp_group = lambda: group
        result["check"] = _check(rank, world, cp_mla)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        traceback.print_exc()
        result["error"] = repr(exc)
    finally:
        (result_dir / f"rank_{rank}.json").write_text(json.dumps(result, indent=1))
    if "error" in result:
        raise SystemExit(1)


def test_dcp_merge_chunked_small_on_real_collectives(tmp_path):
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--result-dir",
        str(tmp_path),
    ]
    done = subprocess.run(
        cmd,
        cwd=Path(__file__).resolve().parents[3],
        env=mm._prepare_worker_env(),
        capture_output=True,
        text=True,
        errors="replace",
        timeout=LAUNCH_TIMEOUT,
    )
    results = mm._results(tmp_path, WORLD_SIZE)
    assert not [r for r in results if "error" in r], (
        results,
        (done.stdout or "")[-6000:],
        (done.stderr or "")[-6000:],
    )
    failures = []
    for r in results:
        for key, c in sorted(r["check"].items()):
            print(f"CHUNKED_SMALL rank{r['rank']} {key}: {json.dumps(c)}")
            if "error" in c:
                failures.append((r["rank"], key, c))
                continue
            if c["tokens"] in SMALL_TOKENS:
                ok = c["off_chunks"] == 8 and c["on_chunks"] == 2
                ok &= c["on_max_ulp"] <= RS_BF16_MAX_ULP
                ok &= c["control_max_ulp"] > RS_BF16_MAX_ULP
            else:
                ok = c["on_chunks"] == c["off_chunks"] and c["ndiff"] == 0
            if "drafter_ndiff" in c:
                ok &= c["drafter_ndiff"] == 0
            if not ok:
                failures.append((r["rank"], key, c))
    assert not failures, failures


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", type=Path, required=True)
    a = p.parse_args()
    _run_worker(a.result_dir)
