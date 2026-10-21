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
"""TPU_FUSED_ADD_RMSNORM torch bridge: the OOT RMSNorm vs the incumbent.

The incumbent is upstream ``forward_native`` compiled by the TPU backend --
the graph the served model runs -- so the comparison covers whatever bf16
roundings XLA keeps or drops there. Importing ``layers.adapter.rmsnorm``
registers the OOT class for the whole process.

Under TPU_FUSED_ADD_RMSNORM_QUANT a norm paired with a per-channel FP8 linear
hands it ``(y_q, scale)``; the linear's output and the residual must equal the
unpaired path bit for bit.
"""

import pytest

torch = pytest.importorskip("torch")

from vllm_torchtpu.layers.adapter import linear_common  # noqa: E402
from vllm_torchtpu.layers.adapter import rmsnorm as tpu_rmsnorm  # noqa: E402
from vllm_torchtpu.layers.adapter.linear_common import PREQUANT_ATTR  # noqa: E402
from vllm_torchtpu.layers.adapter.quantization.fp8 import (  # noqa: E402
    VllmFp8LinearMethodTPU,
)

pytestmark = pytest.mark.tpu_test

MAX_DIFF_FRACTION = 1e-3


def _bits_diff(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def _norm(hidden=6144, eps=1e-5):
    from vllm.model_executor.layers.layernorm import RMSNorm

    n = RMSNorm(hidden, eps=eps)
    assert isinstance(n, tpu_rmsnorm.TpuRMSNorm)
    with torch.no_grad():
        n.weight.copy_(1 + 0.2 * torch.randn(hidden))
    return n.to(torch.bfloat16).to("tpu")


def _assert_close(got, ref, has_res):
    if has_res:
        assert torch.equal(got[1].cpu(), ref[1].cpu()), "residual must be bit-identical"
        d = _bits_diff(got[0].cpu(), ref[0].cpu())
    else:
        d = _bits_diff(got.cpu(), ref.cpu())
    frac = (d > 0).float().mean().item()
    assert d.max().item() <= 2 and frac <= MAX_DIFF_FRACTION, (d.max().item(), frac)


@pytest.mark.parametrize("rows", [1024, 4096])
@pytest.mark.parametrize("has_res", [True, False])
def test_oot_forward_matches_compiled_native(vllm_config_context, rows, has_res):
    torch.manual_seed(rows)
    torch._dynamo.reset()
    n = _norm()
    x = (0.7 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu") if has_res else None
    assert n._use_pallas(x, r)
    got = n.forward(x, r)

    def native(a, b):
        return n.forward_native(a, b)

    compiled = torch.compile(native, fullgraph=True, dynamic=False)
    ref = compiled(x.clone(), None if r is None else r.clone())
    _assert_close(got, ref, has_res)


@pytest.mark.parametrize("has_res", [True, False])
def test_oot_forward_matches_eager_native(vllm_config_context, has_res):
    torch.manual_seed(3)
    n = _norm()
    x = (0.7 * torch.randn(2048, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(2048, 6144)).to(torch.bfloat16).to("tpu") if has_res else None
    got = n.forward(x, r)
    ref = n.forward_native(x.clone(), None if r is None else r.clone())
    _assert_close(got, ref, has_res)


def test_decode_sized_batches_keep_upstream_path(vllm_config_context):
    n = _norm()
    x = torch.randn(16, 6144).to(torch.bfloat16).to("tpu")
    assert not n._use_pallas(x, x.clone())
    # a strided slice (the q/kv latent norm pattern) is not taken either
    wide = torch.randn(1024, 6144 + 512).to(torch.bfloat16).to("tpu")
    assert not n._use_pallas(wide[:, :6144], None)


def test_small_hidden_not_taken(vllm_config_context):
    n = _norm(hidden=512)
    x = torch.randn(4096, 512).to(torch.bfloat16).to("tpu")
    assert not n._pallas_ok and not n._use_pallas(x, None)


def test_keep_upstream_path_pins_a_traced_once_model(vllm_config_context):
    model = torch.nn.Sequential(_norm(), _norm())
    assert tpu_rmsnorm.keep_upstream_path(model) == 2
    x = torch.randn(4096, 6144).to(torch.bfloat16).to("tpu")
    assert not any(m._use_pallas(x, None) for m in model)


class _Fp8Linear(torch.nn.Module):
    """What ``VllmFp8LinearMethodTPU.apply`` reads: a (k, n) fp8 weight, a 1-D
    f32 per-channel scale and no bias."""

    def __init__(self, hidden=6144, n_out=2624):
        super().__init__()
        w = (64 * torch.randn(hidden, n_out)).to(torch.float8_e4m3fn)
        self.weight = torch.nn.Parameter(w.to("tpu"), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            (1e-4 + 9e-4 * torch.rand(n_out)).to("tpu"), requires_grad=False
        )
        self.bias = None
        self.quant_method = object.__new__(VllmFp8LinearMethodTPU)

    def forward(self, x):
        return self.quant_method.apply(self, x)


def _norm_then_linear(n, lin, x, r):
    y, res = n(x, r)
    return lin(y), res


def _residual_stream(rows):
    x = (0.7 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    r = (4 * torch.randn(rows, 6144)).to(torch.bfloat16).to("tpu")
    return x, r


@pytest.mark.parametrize("compiled", [False, True])
def test_paired_matches_unpaired(vllm_config_context, compiled):
    torch.manual_seed(1)
    torch._dynamo.reset()
    n, lin = _norm(), _Fp8Linear()
    x, r = _residual_stream(1024)

    def run():
        if compiled:
            fn = torch.compile(_norm_then_linear, fullgraph=True, dynamic=False)
        else:
            fn = _norm_then_linear
        return fn(n, lin, x.clone(), r.clone())

    ref_out, ref_res = run()
    assert tpu_rmsnorm._pair(n, lin, False)
    torch._dynamo.reset()
    got_out, got_res = run()
    assert getattr(lin, PREQUANT_ATTR).value is None, "the linear must clear the slot"
    assert torch.equal(got_res.cpu(), ref_res.cpu())
    assert torch.equal(got_out.cpu().view(torch.int16), ref_out.cpu().view(torch.int16))


def test_paired_linear_reads_the_slot(vllm_config_context):
    # Positive control: with a slot set, the linear ignores its own input.
    torch.manual_seed(5)
    n, lin = _norm(), _Fp8Linear()
    assert tpu_rmsnorm._pair(n, lin, False)
    x, r = _residual_stream(1024)
    good, _ = _norm_then_linear(n, lin, x.clone(), r.clone())
    n(x.clone(), r.clone())
    other = (0.7 * torch.randn(1024, 6144)).to(torch.bfloat16).to("tpu")
    from_slot = lin(other)
    # The first call cleared the slot, so this one computes from `other`.
    from_input = lin(other)
    assert torch.equal(from_slot.cpu(), good.cpu())
    assert not torch.equal(from_input.cpu(), good.cpu())


def test_decode_sized_batches_keep_unpaired_path(vllm_config_context):
    n, lin = _norm(), _Fp8Linear()
    assert tpu_rmsnorm._pair(n, lin, True)
    x = torch.randn(16, 6144).to(torch.bfloat16).to("tpu")
    n(x, x.clone())
    assert getattr(lin, PREQUANT_ATTR).value is None


def test_compiled_forward_uses_op_built_at_pairing(vllm_config_context):
    """The prequantized matmul op must be built by ``_pair``, not lazily inside
    the compiled region, where its lock fails the Dynamo capture."""
    torch.manual_seed(9)
    torch._dynamo.reset()
    n, lin = _norm(), _Fp8Linear()
    x, r = _residual_stream(1024)
    ref_out, ref_res = _norm_then_linear(n, lin, x.clone(), r.clone())
    linear_common._prequantized_matmul_op = None
    assert tpu_rmsnorm._pair(n, lin, False)
    fn = torch.compile(_norm_then_linear, fullgraph=True, dynamic=False)
    got_out, got_res = fn(n, lin, x.clone(), r.clone())
    assert torch.equal(got_res.cpu(), ref_res.cpu())
    assert torch.equal(got_out.cpu(), ref_out.cpu())


class _DenseLayer(torch.nn.Module):
    """A norm, then an MLP whose gate_up_proj reads it."""

    def __init__(self):
        super().__init__()
        self.post_attention_layernorm = _norm()
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = _Fp8Linear()

    def forward(self, x, r):
        y, res = self.post_attention_layernorm(x, r)
        return self.mlp.gate_up_proj(y), res


def test_both_rows_branches_lift_the_same_inputs(vllm_config_context):
    """The paired (Pallas rows) and unpaired (decode rows) traces must lift the
    same inputs under the same names, or a bucket cannot move between them."""
    layer = _DenseLayer()
    assert tpu_rmsnorm._pair(
        layer.post_attention_layernorm, layer.mlp.gate_up_proj, False
    )
    names = {}

    def capture(rows):
        def backend(gm, example_inputs):
            names[rows] = sorted(
                node.name
                for node in gm.graph.nodes
                if node.op == "placeholder"
                and not node.name.startswith(("l_x_", "l_r_"))
            )
            return gm.forward

        return backend

    for rows in (16, 1024):
        torch._dynamo.reset()
        x = torch.randn(rows, 6144).to(torch.bfloat16).to("tpu")
        fn = torch.compile(layer, backend=capture(rows), fullgraph=True, dynamic=False)
        fn(x, x.clone())
    assert names[16] == names[1024]
    assert not any("prequant" in name for name in names[1024])
