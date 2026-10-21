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
"""TPU_DCP_MLA_SEND_LAYOUT: the decode MLA kernel writes the packed real-rows
DCP merge send buffer itself.

Every row the ragged exchange sends (each destination's lse word row and the
real tokens' partial rows) must equal what `cp_mla_attention._all_to_all_packed`
builds from the kernel's (output, lse) without the send layout. This is
checked per kernel call, through `sparse_mla_attention_dcp` on DCP ranks with
the merge, and composed with TPU_DCP_MLA_MERGE_PIECES on a mixed bucket.

`DCP_SEND_LAYOUT_NPZ=<path>` and `DCP_MERGE_PIECES_NPZ=<path>` save the
partials for the real-collective checks in
tests/layers/adapter/test_dcp_send_layout_merge_multichip.py and
tests/layers/adapter/test_dcp_merge_pieces_multichip.py.
"""

import os
import sys
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

import test_mla_decode_prologue as prologue  # noqa: E402
import test_t3_final_composition as final  # noqa: E402
import test_t4_combined_composition as comb  # noqa: E402

from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla  # noqa: E402

pytestmark = pytest.mark.tpu_test

PAD = sparse_mla.SEND_PAD_ROWS
SEND_ENV = {
    "TPU_MLA_DECODE_PROLOGUE": "1",
    "TPU_MLA_KV_WRITE_IN_GATHER": "1",
    "TPU_DCP_RANK_OPERAND": "1",
    "TPU_DCP_LSE_MERGE_A2A": "1",
    "TPU_DCP_LSE_MERGE_A2A_PACK": "1",
    "TPU_DCP_LSE_MERGE_A2A_REAL_ROWS": "1",
}


def packed_rows(out16, lse32, world, real):
    """The rows the packed real-rows exchange sends, from one rank's
    incumbent (output, lse): int16 lse words [W, V] (hi halves, lo halves,
    zeros) and int16 partial rows [W, real * L, V] (destination-major)."""
    out16 = np.asarray(out16).view(np.int16)
    tokens, heads, vdim = out16.shape
    per = heads // world
    rows = tokens * per
    o = out16.reshape(tokens, world, per, vdim).transpose(1, 0, 2, 3)
    o = o.reshape(world, rows, vdim)
    bits = np.asarray(lse32).view(np.int32).reshape(tokens, world, per)
    bits = bits.transpose(1, 0, 2).reshape(world, rows)
    words = np.zeros((world, vdim), np.int16)
    words[:, :rows] = (bits >> 16).astype(np.int16)
    words[:, rows : 2 * rows] = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(np.int16)
    return words, o[:, : real * per]


def sent_rows(send16, real, per):
    send16 = np.asarray(send16).view(np.int16)
    return send16[:, PAD - 1], send16[:, PAD : PAD + real * per]


def _diffs(send16, out16, lse32, world, real):
    per = out16.shape[1] // world
    want_w, want_o = packed_rows(out16, lse32, world, real)
    got_w, got_o = sent_rows(send16, real, per)
    return int((got_w != want_w).sum()), int((got_o != want_o).sum())


# --- one kernel call ---------------------------------------------------------


def _kernel_send(p, world):
    cache = prologue._cos_sin_cache()
    _, send = sparse_mla.sparse_ragged_paged_attention(
        p["ql_nope"],
        p["nope"],
        p["rope"],
        p["topk"],
        p["page_table"],
        p["cu_q_lens"],
        p["distribution"],
        sm_scale=prologue.SM_SCALE,
        k_scale=prologue.K_SCALE,
        return_lse=True,
        cache_layout=sparse_mla.SPARSECORE_CACHE_LAYOUT,
        gather_and_attention_chunk_size=256,
        topk_cap=512,
        small_chunk_batch_size=4,
        small_chunk_tokens=32,
        lse_lane_dense=True,
        lse_mask_empty=True,
        fused_window_prep=True,
        mixed_live_gather=True,
        mixed_live_widths=(256, 384),
        mixed_body_levers=sparse_mla.MIXED_BODY_DEFAULT,
        q_pe=p["q_raw"],
        q_rope_table=prologue.rope_table(p["positions"], cache),
        lse_in_kernel=True,
        send_world=world,
    )
    return np.asarray(jax.block_until_ready(send))


GEOMETRIES = ((16, 64), (8, 64), (8, 32))  # (ranks, heads)
DECODE_CELLS = tuple(c for c in prologue.CELLS if c[0] <= 32)


def _problem(query_tokens, seq_q, heads, zero=False):
    with mock.patch.object(prologue, "HEADS", heads):
        p = prologue._problem(
            np.random.default_rng(1012 + query_tokens + heads), query_tokens, seq_q
        )
    if zero:
        p["cu_q_lens"] = jnp.array([0, 0], jnp.int32)
        p["distribution"] = jnp.array([0, 0, 0], jnp.int32)
    return p


@pytest.mark.parametrize("world,heads", GEOMETRIES)
@pytest.mark.parametrize("cell", DECODE_CELLS + ((16, ()),), ids=str)
def test_kernel_send_rows_bit_identical(world, heads, cell):
    prologue._require_tpu()
    query_tokens, seq_q = cell
    p = _problem(query_tokens, seq_q, heads, zero=not seq_q)
    real = int(sum(seq_q))
    out, lse = prologue._kernel(p, candidate=True)
    send = _kernel_send(p, world)
    per = heads // world
    assert send.shape == (world, PAD + query_tokens * per, out.shape[-1]), send.shape
    nw, no = _diffs(send, prologue._bits16(out), np.asarray(lse), world, real)
    assert nw == 0 and no == 0, (nw, no)
    # Non-vacuous: the sent lse holds -inf (empty / padding rows) and finite
    # values, and the head rows other than the word row are zero.
    lse_np = np.asarray(lse)
    assert np.isneginf(lse_np).any()
    if real:
        assert np.isfinite(lse_np[:real]).any()
    assert not np.asarray(send).view(np.int16)[:, : PAD - 1].any()


# --- sparse_mla_attention_dcp on DCP ranks -----------------------------------


def _rank_sends(step, dcp):
    import torch

    send = torch.from_numpy(step["out"].view(np.int16).copy())
    return send.reshape(dcp, dcp, *send.shape[1:])


def _merge_send_every_rank(sends, real, tokens):
    """`merge_send_layout` on every DCP rank (CPU, lockstep threads standing
    in for the ragged exchange)."""
    import concurrent.futures
    import threading

    import torch

    from vllm_torchtpu.layers.adapter import cp_mla_attention as cp_mla

    world = sends.shape[0]
    slots = [None] * world
    barrier = threading.Barrier(world)
    local = threading.local()

    class _Group:
        def __init__(self, rank):
            self.world_size, self.rank_in_group, self.device_group = world, rank, None

    def ragged(x, rows, group, skip=0):
        slots[group.rank_in_group] = x
        barrier.wait()
        recv = torch.stack([slots[s][group.rank_in_group] for s in range(world)])
        barrier.wait()
        n = int(rows.reshape(-1)[0])
        out = torch.zeros_like(recv)
        out[:, skip : skip + n] = recv[:, skip : skip + n]
        return out

    qsl = torch.tensor([0, real], dtype=torch.int32)

    def run(rank):
        local.group = _Group(rank)
        return cp_mla.merge_send_layout(sends[rank], qsl, tokens)

    with (
        mock.patch.object(cp_mla, "_ragged_all_to_all", ragged),
        mock.patch.object(cp_mla, "get_dcp_group", lambda: local.group),
        concurrent.futures.ThreadPoolExecutor(world) as pool,
    ):
        return torch.cat(list(pool.map(run, range(world))), dim=1)


def _dcp_size():
    return min(8, len(jax.devices()))


def _send_layout_dcp_op(dcp, attention_interface, real_fn):
    """`sparse_mla_attention_dcp` as the layer calls it with the send layout:
    the send op on decode buckets, the partial + lse op otherwise."""

    def wrapped(ql_nope, *args, **kwargs):
        if ql_nope.shape[0] > sparse_mla.FUSED_PREP_MAX_TOKENS:
            return real_fn(ql_nope, *args, **kwargs)
        nope, rope, send = real_fn(ql_nope, *args, send_layout=True, **kwargs)
        lse = jnp.zeros((dcp * ql_nope.shape[0], ql_nope.shape[1]), jnp.float32)
        return nope, rope, send, lse

    return mock.patch.object(attention_interface, "sparse_mla_attention_dcp", wrapped)


def _send_dcp_run_with_rank(dcp):
    from vllm_torchtpu.layers.core import attention_interface

    real_fn = attention_interface.sparse_mla_attention_dcp
    with _send_layout_dcp_op(dcp, attention_interface, real_fn):
        return prologue._dcp_run(dcp, candidate=True, dcp_rank_ids=comb._rank_ids(dcp))


def test_dcp_send_layout_bit_identical_and_merges():
    prologue._require_tpu()
    dcp = _dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    import torch

    env = dict(prologue.PROD_ENV, **SEND_ENV)
    with mock.patch.dict(os.environ, env):
        jax.clear_caches()
        ref = prologue._dcp_run(dcp, candidate=True, dcp_rank_ids=comb._rank_ids(dcp))
        jax.clear_caches()
        got = _send_dcp_run_with_rank(dcp)
    dump = {}
    checked = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        tokens = a["out"].shape[0] // dcp
        if tokens > sparse_mla.FUSED_PREP_MAX_TOKENS:
            continue
        real = a["real"]
        outs, lses = prologue._rank_partials(a, dcp)
        sends = _rank_sends(b, dcp)
        for r in range(dcp):
            nw, no = _diffs(
                sends[r].numpy(),
                outs[r].view(torch.int16).numpy(),
                lses[r].numpy(),
                dcp,
                real,
            )
            assert nw == 0 and no == 0, f"step {i} rank {r}: {nw} / {no}"
        want = comb._merge_every_rank(outs, lses, on=True, real=real)
        one = _merge_send_every_rank(sends, real, tokens)
        diff = int((one.view(torch.int16) != want.view(torch.int16)).sum())
        assert diff == 0, f"step {i}: {diff} of {want.numel()} elements differ"
        assert torch.isfinite(one.float()).all()
        assert (one[:real] != 0).any()
        dump[f"s{checked}_on_out"] = a["out"]
        dump[f"s{checked}_on_lse"] = a["lse"]
        dump[f"s{checked}_send"] = b["out"]
        dump[f"s{checked}_real"] = np.int32(real)
        checked += 1
    assert checked == 2, checked  # decode 16 and decode 32
    path = os.environ.get("DCP_SEND_LAYOUT_NPZ")
    if path:
        np.savez(path, dcp=np.int32(dcp), steps=np.int32(checked), **dump)


COMPOSED_FLAGS_ON = dict(
    final.T3_FLAGS_ON,
    TPU_DCP_LSE_MERGE_A2A="1",
    TPU_DCP_MERGE_CHUNKED="8",
    TPU_DCP_MLA_MERGE_PIECES="2",
    TPU_DCP_MLA_MERGE_PIECES_MIN_TOKENS="2048",
)


def test_send_layout_and_merge_pieces_compose_with_dcp_flags():
    """All DCP attention flags with the send layout on the decode steps and
    TPU_DCP_MLA_MERGE_PIECES=2 on the 2048-token mixed step, against all
    flags off. Caches, mixed-step partials, sent rows and merged rows must
    match bit for bit."""
    prologue._require_tpu()
    dcp = comb._dcp_size()
    if dcp < 2:
        pytest.skip("DCP needs at least 2 devices")
    import torch

    from vllm_torchtpu.layers.core import attention_interface

    for k in COMPOSED_FLAGS_ON:
        os.environ.pop(k, None)
    ref, _ = comb._attention_run(dcp, on=False)
    seen = []
    real_k = sparse_mla.sparse_ragged_paged_attention

    def spy(*args, **kwargs):
        seen.append(
            (
                tuple(kwargs.get("decode_row_widths", ())),
                int(kwargs.get("send_world", 0)),
                int(kwargs.get("out_pieces", 0)),
            )
        )
        return real_k(*args, **kwargs)

    real_fn = attention_interface.sparse_mla_attention_dcp
    with (
        mock.patch.object(comb, "FLAGS_ON", COMPOSED_FLAGS_ON),
        _send_layout_dcp_op(dcp, attention_interface, real_fn),
        mock.patch.object(sparse_mla, "sparse_ragged_paged_attention", spy),
    ):
        got, got_w = comb._attention_run(dcp, on=True)
    assert got_w["tc"] == 0 and got_w["kvw"] == 2 and got_w["iota"] == 0, got_w
    assert got_w["prep"] > 0, got_w
    widths = tuple(int(w) for w in final.ROW_WIDTHS.split(","))
    assert widths in [w for w, _, _ in seen], seen
    assert {w for w, _, _ in seen} <= {(), widths}, seen
    assert [s for _, s, _ in seen if s] == [dcp, dcp], seen
    # One piece per merge chunk (TPU_DCP_MERGE_CHUNKED=8), the 2048 call only.
    assert [p for _, _, p in seen if p] == [8], seen

    mixed = {}
    ns = nm = 0
    for i, (a, b) in enumerate(zip(ref, got)):
        for k in ("nope", "rope"):
            np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
        real = a["real"]
        tokens = a["out"].shape[0] // dcp
        outs, lses = prologue._rank_partials(a, dcp)
        want = comb._merge_every_rank(outs, lses, on=False)
        if tokens <= sparse_mla.FUSED_PREP_MAX_TOKENS:
            sends = _rank_sends(b, dcp)
            for r in range(dcp):
                nw, no = _diffs(
                    sends[r].numpy(),
                    outs[r].view(torch.int16).numpy(),
                    lses[r].numpy(),
                    dcp,
                    real,
                )
                assert nw == 0 and no == 0, f"step {i} rank {r}: {nw} / {no}"
            one = _merge_send_every_rank(sends, real, tokens)
            ns += 1
        else:
            for k in ("out", "lse"):
                np.testing.assert_array_equal(b[k], a[k], err_msg=f"step {i} {k}")
            one = comb._merge_every_rank(
                *prologue._rank_partials(b, dcp), on=True, real=real
            )
            for tag, r in (("off", a), ("on", b)):
                mixed[f"s{nm}_{tag}_out"] = r["out"]
                mixed[f"s{nm}_{tag}_lse"] = r["lse"]
            mixed[f"s{nm}_real"] = np.int32(real)
            nm += 1
        diff = int(
            (one[:real].view(torch.int16) != want[:real].view(torch.int16)).sum()
        )
        assert diff == 0, f"step {i}: {diff} elements differ"
        assert (one[real:] == 0).all(), f"step {i}"
        assert torch.isfinite(one.float()).all(), f"step {i}"
    assert ns == 2 and nm == 2, (ns, nm)
    path = os.environ.get("DCP_MERGE_PIECES_NPZ")
    if path:
        np.savez(path, dcp=np.int32(dcp), steps=np.int32(nm), **mixed)


@pytest.mark.parametrize("rank", [False, True])
def test_sparse_mla_dcp_send_op_schema(rank):
    """The send-layout torch op registers with the prologue op's operands."""
    import torch

    from vllm_torchtpu.kernels.deepseek_v4 import streamindex_topk
    from vllm_torchtpu.layers.adapter import attention

    mesh = jax.sharding.Mesh(
        np.array(jax.devices("cpu")[:1]), (streamindex_topk.DCP_AXIS_NAME,)
    )
    name = f"send_schema_{int(rank)}"
    layer = mock.Mock(
        kv_cache_quantized_dtype=None, layer_name=name, scale=0.1, mla_kv_spec=(1, 2)
    )
    impl = mock.Mock(_dcp_mesh=mesh)
    env = {"TPU_DCP_RANK_OPERAND": "1"} if rank else {}
    with (
        mock.patch.dict(os.environ, env),
        mock.patch.object(
            attention,
            "dcp_rank_operand",
            return_value=torch.zeros(1, dtype=torch.int32),
        ),
    ):
        if not rank:
            os.environ.pop("TPU_DCP_RANK_OPERAND", None)
        attention.PallasMLAttentionBackendImpl._build_sparse_mla_dcp_op(
            impl, layer, 8, 4, q_rope=True, send_layout=True
        )
    op = f"sparse_mla_attention_dcp_{name}_qrope" + ("_rank" if rank else "") + "_send"
    schema = getattr(torch.ops.pallas, op).default._schema
    assert len(schema.arguments) == 12 + int(rank), schema
    # Caches + the send buffer: a schema with the partial + lse pair's four
    # returns segfaults torch's functionalization on the three real outputs.
    assert len(schema.returns) == 3, schema
