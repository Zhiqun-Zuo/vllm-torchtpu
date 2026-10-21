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
