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
"""`weighted_heads_major` (TPU_DCP_MERGE_WEIGHTED_HM) against the torch
merge's weight multiply, transposed: bit for bit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla import dcp_weighted_hm

pytestmark = pytest.mark.tpu_test


def _case(t, h, d=512, seed=0):
    rng = np.random.default_rng(seed)
    o = rng.normal(size=(t, h, d)).astype(np.float32) * 4
    o[1, :, :7] = np.nan  # NaN partial rows: weighted 0 where w == 0
    o[2, 3, 5] = np.inf
    lse = rng.normal(size=(t, h)).astype(np.float32) * 3
    lse[1] = -np.inf
    lse[5, ::3] = -np.inf
    merged = np.logaddexp(lse, rng.normal(size=(t, h)).astype(np.float32) * 3)
    with np.errstate(invalid="ignore"):
        w = np.exp(lse - merged)
    w[7, 0] = np.nan  # exp(-inf - -inf)
    return jnp.asarray(o, jnp.bfloat16), jnp.asarray(w, jnp.float32)


def _reference(o, w):
    x = o.astype(jnp.float32) * w[..., None]
    x = jnp.where(w[..., None] > 0, x, jnp.zeros_like(x))
    return jnp.transpose(x, (1, 0, 2))


@pytest.mark.parametrize("t,h", [(512, 64), (512, 32), (8, 64)])
def test_weighted_heads_major_bit_identical(t, h):
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Pallas TPU kernel")
    o, w = _case(t, h, seed=t + h)
    got = np.asarray(dcp_weighted_hm.weighted_heads_major(o, w)).view(np.uint32)
    want = np.asarray(jax.jit(_reference)(o, w)).view(np.uint32)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("t,h", [(512, 64), (512, 32), (16, 64)])
def test_weighted_heads_major_bf16_out_bit_identical(t, h):
    """`out_dtype=bf16` (TPU_DCP_MERGE_RS_BF16=1): the f32 value rounded once
    to bf16, bit for bit the torch merge's `.to(bfloat16)`."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Pallas TPU kernel")
    o, w = _case(t, h, seed=3 * t + h)
    got = dcp_weighted_hm.weighted_heads_major(o, w, out_dtype=jnp.bfloat16)
    assert got.dtype == jnp.bfloat16 and got.shape == (h, t, o.shape[-1])
    want = jax.jit(lambda o, w: _reference(o, w).astype(jnp.bfloat16))(o, w)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint16), np.asarray(want).view(np.uint16)
    )


@pytest.mark.parametrize("out_dtype", [jnp.float32, jnp.bfloat16])
def test_weighted_heads_major_xla_matches_kernel(out_dtype):
    """The XLA path below TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS writes the
    kernel's bits, so the gate changes no merged row."""
    if jax.devices()[0].platform != "tpu":
        pytest.skip("Pallas TPU kernel")
    t, h = 256, 64
    o, w = _case(t, h, seed=5 * t + h)
    view = np.uint16 if jnp.dtype(out_dtype).itemsize == 2 else np.uint32
    got = dcp_weighted_hm.weighted_heads_major_xla(o, w, out_dtype=out_dtype)
    want = dcp_weighted_hm.weighted_heads_major(o, w, out_dtype=out_dtype)
    assert got.dtype == want.dtype and got.shape == want.shape == (h, t, o.shape[-1])
    np.testing.assert_array_equal(
        np.asarray(got).view(view), np.asarray(want).view(view)
    )


@pytest.mark.parametrize(
    "min_tokens,chunks,rows,kernel",
    [
        (0, 8, 128, True),  # default: every bucket
        (2048, 8, 128, False),  # 1024-token bucket
        (2048, 8, 256, True),  # 2048-token bucket
        (2048, 4, 256, False),  # the gate counts buckets, not chunk rows
    ],
)
def test_weighted_hm_min_tokens_dispatch(monkeypatch, min_tokens, chunks, rows, kernel):
    adapter = pytest.importorskip("vllm_torchtpu.layers.adapter.dcp_weighted_hm")
    import vllm_torchtpu.envs as envs

    monkeypatch.setattr(
        envs, "TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS", min_tokens, raising=False
    )
    monkeypatch.setattr(envs, "TPU_DCP_MERGE_CHUNKED", chunks, raising=False)
    calls = []
    monkeypatch.setattr(
        adapter._k,
        "weighted_heads_major",
        lambda o, w, out_dtype: calls.append("kernel"),
    )
    monkeypatch.setattr(
        adapter._k,
        "weighted_heads_major_xla",
        lambda o, w, out_dtype: calls.append("xla"),
    )
    adapter._build(
        jnp.zeros((rows, 64, 512), jnp.bfloat16), jnp.zeros((rows, 64)), jnp.bfloat16
    )
    assert calls == ["kernel" if kernel else "xla"]


def _vspace_case(t, h=64, c=512, v=256, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.random((t, h)).astype(np.float32)
    w[rng.random((t, h)) < 0.1] = 0.0
    o = rng.standard_normal((t, h, c)).astype(np.float32)
    o[w == 0.0] = np.nan  # a zero-weight partial may be NaN
    w_uv = rng.standard_normal((h, c, v)) / np.sqrt(c)
    return (
        jnp.asarray(o, jnp.bfloat16),
        jnp.asarray(w),
        jnp.asarray(w_uv, jnp.bfloat16),
    )


def _vspace_reference(o, w, w_uv, scale=None):
    o = np.nan_to_num(np.asarray(o.astype(jnp.float32), np.float64))
    v = np.matmul(
        o.transpose(1, 0, 2), np.asarray(w_uv.astype(jnp.float32), np.float64)
    )
    if scale is not None:
        v = v * np.asarray(scale, np.float64)
    w = np.asarray(w, np.float64).T[:, :, None]
    return np.where(w > 0, v * w, 0.0)


def _max_rel_err(got, want):
    got = np.asarray(jnp.asarray(got, jnp.float32), np.float64)
    return float(np.abs(got - want).max() / np.abs(want).max())


def test_vspace_weighted_heads_major_matches_f64():
    o, w, w_uv = _vspace_case(256)
    got = dcp_weighted_hm.vspace_weighted_heads_major(o, w, w_uv, out_dtype=jnp.float32)
    assert got.shape == (64, 256, 256)
    assert bool(jnp.isfinite(got).all())
    assert _max_rel_err(got, _vspace_reference(o, w, w_uv)) < 1e-5


def test_vspace_weighted_heads_major_applies_the_scale():
    o, w, w_uv = _vspace_case(128, seed=1)
    scale = np.random.default_rng(3).random((64, 1, 256)).astype(np.float32) + 0.5
    got = dcp_weighted_hm.vspace_weighted_heads_major(
        o, w, w_uv, jnp.asarray(scale), out_dtype=jnp.float32
    )
    assert _max_rel_err(got, _vspace_reference(o, w, w_uv, scale)) < 1e-5


def test_vspace_weighted_heads_major_bf16_is_f32_rounded_once():
    o, w, w_uv = _vspace_case(128, seed=2)
    f32 = dcp_weighted_hm.vspace_weighted_heads_major(o, w, w_uv, out_dtype=jnp.float32)
    bf16 = dcp_weighted_hm.vspace_weighted_heads_major(
        o, w, w_uv, out_dtype=jnp.bfloat16
    )
    assert bf16.dtype == jnp.bfloat16
    np.testing.assert_array_equal(
        np.asarray(bf16).view(np.uint16),
        np.asarray(f32.astype(jnp.bfloat16)).view(np.uint16),
    )


def test_vspace_torch_op_compiled_matches_jax():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tpu")
    from torch_tpu._internal import compile as tpu_compile

    from vllm_torchtpu.layers.adapter import dcp_weighted_hm as adapter

    o, w, w_uv = _vspace_case(256, seed=4)
    want = dcp_weighted_hm.vspace_weighted_heads_major(
        o, w, w_uv, out_dtype=jnp.bfloat16
    )

    def to_torch(x):
        dtype = torch.bfloat16 if x.dtype == jnp.bfloat16 else torch.float32
        return torch.from_numpy(np.asarray(x.astype(jnp.float32))).to(dtype).to("tpu")

    torch._dynamo.reset()
    fn = torch.compile(
        adapter.vspace_weighted_heads_major,
        fullgraph=True,
        dynamic=False,
        backend=tpu_compile.TpuBackend(),
    )
    got = fn(to_torch(o), to_torch(w), to_torch(w_uv)).cpu()
    assert tuple(got.shape) == (64, 256, 256) and got.dtype == torch.bfloat16
    np.testing.assert_array_equal(
        got.view(torch.int16).numpy(), np.asarray(want).view(np.int16)
    )
