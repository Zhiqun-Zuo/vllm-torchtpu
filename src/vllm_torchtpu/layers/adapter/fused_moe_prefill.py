# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Fused MoE prefill kernel path of moe_tp_decode's TP-expert layers.

With ``FUSED_MOE_PREFILL_KERNEL=1``, the moe_tp_decode op runs steps of at
least ``FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS`` tokens through
``fused_moe_prefill`` per rank under shard_map: routed and shared experts, the
combine and the all-reduce. The op picks the path inside its exported program,
where the token count is static.
"""

import vllm_torchtpu.envs as envs
from vllm_torchtpu.distributed.ep_mesh import EP_AXIS_NAME
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import fused_moe_prefill
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def prefill_min_tokens() -> int | None:
    """The smallest step the prefill kernel serves, or None when it is off."""
    if not envs.FUSED_MOE_PREFILL_KERNEL:
        return None
    min_tokens = envs.FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS
    if min_tokens < 1:
        raise ValueError(
            f"FUSED_MOE_PREFILL_KERNEL_MIN_TOKENS={min_tokens} must be at least 1."
        )
    return min_tokens


def prefill_layer(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    shared_w1,
    shared_w1_scale,
    shared_w2,
    shared_w2_scale,
    shared_gate_weight,
    tw,
    ti,
    start,
    routed_scale=1.0,
):
    """Per-rank body under shard_map: the all-reduced shared + routed output.

    `start` is the rank's first global expert under EP, None under TP.
    `shared_gate_weight` is None for a shared expert without a gate.
    `routed_scale` multiplies the routed output only.
    """
    # Logged at trace time, so once per token bucket.
    logger.info("Fused MoE prefill kernel traced for %d tokens", x.shape[0])
    return fused_moe_prefill(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        shared_w1,
        shared_w1_scale,
        shared_w2,
        shared_w2_scale,
        shared_gate_weight,
        tw,
        ti,
        axis_name=EP_AXIS_NAME,
        expert_start=start,
        routed_scale=routed_scale,
    )
