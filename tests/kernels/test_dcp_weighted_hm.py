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
