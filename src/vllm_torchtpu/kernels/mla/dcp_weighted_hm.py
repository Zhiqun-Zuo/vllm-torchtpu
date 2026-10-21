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
"""The DCP merge's weighted reduce-scatter operand, built heads-major.

The reduce-scatter merge scatters the head axis, so XLA lays its operand out
heads-major and adds a relayout copy after a tokens-major weight multiply.
This kernel writes `where(w > 0, o * w, 0)` directly into a `[H, T, D]`
buffer in one pass, with the same f32 products.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel(o_ref, w_ref, out_ref, *, heads: int):
    # o_ref [bt, heads, D] (tokens-major, as the attention wrote it), w_ref
    # f32[bt, heads] (the weights as the merge computed them), out_ref
    # f32[heads, bt, D]: head j's rows are a strided sublane read of the input
    # block, its weights a lane column, written contiguously.
    w_all = w_ref[...]
    for j in range(heads):
        w = w_all[:, j : j + 1]  # f32[bt, 1]
        x = o_ref[:, j, :].astype(jnp.float32) * w
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        # A bf16 out_ref (TPU_DCP_MERGE_RS_BF16) rounds the same f32 value
        # once, as the torch merge's `weighted.to(bfloat16)` does.
        out_ref[j] = jnp.where(w > 0, x, jnp.zeros_like(x)).astype(out_ref.dtype)


# Bytes of one grid step's input (bf16) plus output (f32) block.
_STEP_BYTES = 6 << 20


def _row_block(tokens: int, heads: int, dim: int, min_rows: int = 8) -> int:
    for bt in (256, 128, 64, 32, 16, 8):
        if bt >= min_rows and tokens % bt == 0 and bt * heads * dim * 6 <= _STEP_BYTES:
            return bt
    return min_rows if tokens % min_rows == 0 else tokens


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def weighted_heads_major(
    o: jax.Array, w: jax.Array, out_dtype=jnp.float32
) -> jax.Array:
    """`o` [T, H, D] (bf16), `w` f32[T, H] -> f32[H, T, D] =
    `where(w > 0, o * w, 0)` transposed to heads-major, bit for bit the
    torch merge's `output.float() * weight` / `where` per element.

    `o` is read in its own [T, H, D] layout, because a 2-D view of a tiled
    3-D buffer is a relayout copy. A bf16 `out_dtype` (TPU_DCP_MERGE_RS_BF16)
    rounds each f32 value once, for the bf16 reduce-scatter.
    """
    out_dtype = jnp.dtype(out_dtype)
    t, h, d = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if d % 128:
        raise ValueError(f"head dim {d} must be a multiple of 128")
    # Every head per step: the [bt, H] weight block's lane dim must be all
    # of H (< 128 heads).
    hb = h
    # bf16 output blocks pack 16 rows per sublane tile.
    bt = _row_block(t, hb, d, min_rows=16 if out_dtype.itemsize == 2 else 8)
    kernel = functools.partial(_kernel, heads=hb)
    return pl.pallas_call(
        kernel,
        grid=(1, t // bt),
        in_specs=[
            pl.BlockSpec((bt, hb, d), lambda g, i: (i, g, 0)),
            pl.BlockSpec((bt, hb), lambda g, i: (i, g)),
        ],
        out_specs=pl.BlockSpec((hb, bt, d), lambda g, i: (g, i, 0)),
        out_shape=jax.ShapeDtypeStruct((h, t, d), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel")
        ),
        name=f"dcp_weighted_hm_t{t}_h{h}"
        + ("_bf16" if out_dtype.itemsize == 2 else ""),
    )(o, w.astype(jnp.float32))


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def weighted_heads_major_xla(
    o: jax.Array, w: jax.Array, out_dtype=jnp.float32
) -> jax.Array:
    """`weighted_heads_major` in XLA ops, with the same bits.

    Used below TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS, where the kernel launch
    costs more than the XLA fusion.
    """
    t, h, _ = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    w = w.astype(jnp.float32)[:, :, None]
    x = o.astype(jnp.float32) * w
    x = jnp.where(w > 0, x, jnp.zeros_like(x)).astype(out_dtype)
    return jnp.transpose(x, (1, 0, 2))


@functools.partial(jax.jit, static_argnames=("out_dtype",))
def vspace_weighted_heads_major(
    o: jax.Array,
    w: jax.Array,
    w_uv: jax.Array,
    w_uv_scale: jax.Array | None = None,
    out_dtype=jnp.bfloat16,
) -> jax.Array:
    """`o` [T, H, C] (bf16), `w` f32[T, H], `w_uv` [H, C, V] with an optional
    per-channel scale [H, 1, V] -> [H, T, V] =
    `where(w > 0, (o_h @ W_UV_h) * w, 0)`, heads-major (TPU_DCP_MERGE_VSPACE).

    `w_uv` holds W_UV for the same H heads as `o`, in the same order. The
    projection takes bf16 operands and accumulates in f32, and the weight
    multiplies the f32 product, so the cast to `out_dtype` is the only
    rounding before the collective.
    """
    out_dtype = jnp.dtype(out_dtype)
    t, h, c = o.shape
    if w.shape != (t, h):
        raise ValueError(f"w {w.shape} must be [{t}, {h}]")
    if w_uv.shape[:2] != (h, c):
        raise ValueError(f"w_uv {w_uv.shape} must be [{h}, {c}, V]")
    v = jax.lax.dot_general(
        o,
        w_uv.astype(o.dtype),
        (((2,), (1,)), ((1,), (0,))),
        preferred_element_type=jnp.float32,
    )  # [H, T, V]
    if w_uv_scale is not None:
        v = v * w_uv_scale.astype(jnp.float32).reshape(h, 1, -1)
    wt = jnp.transpose(w.astype(jnp.float32), (1, 0))[:, :, None]  # [H, T, 1]
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    return jnp.where(wt > 0, v * wt, jnp.zeros_like(v)).astype(out_dtype)
