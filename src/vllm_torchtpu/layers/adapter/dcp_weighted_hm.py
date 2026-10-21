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
"""Torch bridge for the DCP merge's heads-major weighted operand
(`kernels/mla/dcp_weighted_hm.py`, TPU_DCP_MERGE_WEIGHTED_HM)."""

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.mla import dcp_weighted_hm as _k


def use_kernel(chunk_rows: int) -> bool:
    """Whether the bucket reaches TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS.

    The op sees one merge chunk; the chunked merge splits the bucket into
    TPU_DCP_MERGE_CHUNKED equal chunks.
    """
    bucket = chunk_rows * max(int(envs.TPU_DCP_MERGE_CHUNKED), 1)
    return bucket >= envs.TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS


def _build(o: jax.Array, w: jax.Array, out_dtype) -> jax.Array:
    if use_kernel(o.shape[0]):
        return _k.weighted_heads_major(o, w, out_dtype=out_dtype)
    return _k.weighted_heads_major_xla(o, w, out_dtype=out_dtype)


def _fn(o: jax.Array, w: jax.Array) -> jax.Array:
    return _build(o, w, jnp.float32)


def _fake(o: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    t, h, d = o.shape
    return torch.empty((h, t, d), dtype=torch.float32, device=o.device)


_op = pallas.jax_op("pallas::dcp_weighted_heads_major", _fn)
_op.register_fake(_fake)


def _fn_bf16(o: jax.Array, w: jax.Array) -> jax.Array:
    return _build(o, w, jnp.bfloat16)


def _fake_bf16(o: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    t, h, d = o.shape
    return torch.empty((h, t, d), dtype=torch.bfloat16, device=o.device)


_op_bf16 = pallas.jax_op("pallas::dcp_weighted_heads_major_bf16", _fn_bf16)
_op_bf16.register_fake(_fake_bf16)


def weighted_heads_major(
    output: torch.Tensor, weight: torch.Tensor, bf16: bool = False
) -> torch.Tensor:
    """[T, H, D], f32[T, H] -> f32[H, T, D] = where(w > 0, o * w, 0), heads-major.
    `bf16` (TPU_DCP_MERGE_RS_BF16): the same values rounded once to bf16."""
    op = _op_bf16 if bf16 else _op
    return op(output.contiguous(), weight.to(torch.float32).contiguous())


def _register_vspace(bf16: bool, scaled: bool):
    out_dtype = jnp.bfloat16 if bf16 else jnp.float32
    t_dtype = torch.bfloat16 if bf16 else torch.float32

    if scaled:

        def fn(
            o: jax.Array, w: jax.Array, w_uv: jax.Array, scale: jax.Array
        ) -> jax.Array:
            return _k.vspace_weighted_heads_major(
                o, w, w_uv, scale, out_dtype=out_dtype
            )

        def fake(o, w, w_uv, scale):
            t, h, _ = o.shape
            return torch.empty((h, t, w_uv.shape[-1]), dtype=t_dtype, device=o.device)

    else:

        def fn(o: jax.Array, w: jax.Array, w_uv: jax.Array) -> jax.Array:
            return _k.vspace_weighted_heads_major(o, w, w_uv, None, out_dtype=out_dtype)

        def fake(o, w, w_uv):
            t, h, _ = o.shape
            return torch.empty((h, t, w_uv.shape[-1]), dtype=t_dtype, device=o.device)

    name = "pallas::dcp_vspace_weighted_hm" + ("_bf16" if bf16 else "_f32")
    op = pallas.jax_op(name + ("_scaled" if scaled else ""), fn)
    op.register_fake(fake)
    return op


# The V-space operand (TPU_DCP_MERGE_VSPACE): one op per (bf16 output, scaled
# W_UV), all registered at import, never inside a trace.
_VSPACE_OPS = {
    (bf16, scaled): _register_vspace(bf16, scaled)
    for bf16 in (True, False)
    for scaled in (False, True)
}


def vspace_weighted_heads_major(
    output: torch.Tensor,
    weight: torch.Tensor,
    w_uv: torch.Tensor,
    w_uv_scale: torch.Tensor | None = None,
    bf16: bool = True,
) -> torch.Tensor:
    """[T, H, C], f32[T, H], W_UV [H, C, V] (+ scale) -> [H, T, V] =
    where(w > 0, (o_h @ W_UV_h) * w, 0), heads-major, in bf16 (`bf16`,
    TPU_DCP_MERGE_RS_BF16=1) or f32."""
    args = [
        output.contiguous(),
        weight.to(torch.float32).contiguous(),
        w_uv.contiguous(),
    ]
    if w_uv_scale is not None:
        args.append(w_uv_scale.contiguous())
    return _VSPACE_OPS[(bf16, w_uv_scale is not None)](*args)


def _keep_fn(a: jax.Array, b: jax.Array) -> jax.Array:
    del b
    return a


def _keep_fake(a: torch.Tensor, b: torch.Tensor):
    return torch.empty_like(a)


_keep_op = pallas.jax_op("pallas::dcp_keep_input", _keep_fn)
_keep_op.register_fake(_keep_fake)


def keep_input(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """`a` unchanged; `b` becomes an input of the traced graph without any
    computation on it (the exported function ignores it, so XLA drops the
    use: no copy, no op). For graphs that must lift the same buffers."""
    return _keep_op(a.contiguous(), b)
