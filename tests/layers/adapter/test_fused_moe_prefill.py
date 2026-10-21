# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused MoE prefill kernel path of moe_tp_decode. No TPU needed:
the kernel is a stub."""

import numpy as np

from vllm_torchtpu.layers.adapter import fused_moe_prefill

H, TOPK = 6144, 8
ROUTED_SCALE = 2.5


def test_prefill_layer_passes_gate_and_scale_to_kernel(monkeypatch):
    """GLM-5.2: a shared expert without a gate and a routed scale reach the
    kernel as given."""
    seen = {}

    def kernel(*args, axis_name, expert_start, routed_scale):
        seen.update(gate=args[9], expert_start=expert_start, routed_scale=routed_scale)
        return args[0]

    monkeypatch.setattr(fused_moe_prefill, "fused_moe_prefill", kernel)
    w = np.zeros((1, 1), np.float32)
    fused_moe_prefill.prefill_layer(
        np.zeros((256, H), np.float32),
        *([w] * 8),
        None,
        np.zeros((256, TOPK), np.float32),
        np.zeros((256, TOPK), np.int32),
        None,
        ROUTED_SCALE,
    )
    assert seen == dict(gate=None, expert_start=None, routed_scale=ROUTED_SCALE)
