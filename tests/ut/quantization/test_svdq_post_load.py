#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import pytest
import torch
from types import SimpleNamespace

import vllm_ascend.quantization.methods.w4a8_svdq as w4a8_svdq
from vllm_ascend.ops.fused_moe.moe_runtime_args import build_fused_experts_input
from vllm_ascend.quantization.methods.w4a8_svdq import AscendW4A8SVDQFusedMoEMethod
from vllm_ascend.quantization.methods.svdq_post_load import (
    SVDQ_BF16_DEBUG_STAGE_NAMES,
    FINAL_SVDQ_FACTOR_NAMES,
    audit_svdq_operator_factors,
    build_svdq_bf16_stage_reference,
    build_svdq_operator_factors,
)
from vllm_ascend.quantization.quant_type import QuantType


def _bf16_arange(shape, offset=0):
    total = 1
    for dim in shape:
        total *= dim
    return (torch.arange(offset, offset + total, dtype=torch.float32).reshape(shape) / 32).to(torch.bfloat16)


def _make_raw_factor_layer():
    layer = torch.nn.Module()
    layer.layer_name = "model.language_model.layers.0.mlp.experts"
    layer.local_num_experts = 2
    raw = {
        "gate_svd_l1_raw": _bf16_arange((2, 2, 4), 0),
        "gate_svd_l2_raw": _bf16_arange((2, 3, 2), 100),
        "up_svd_l1_raw": _bf16_arange((2, 1, 4), 200),
        "up_svd_l2_raw": _bf16_arange((2, 3, 1), 300),
        "down_svd_l1_raw": _bf16_arange((2, 2, 3), 400),
        "down_svd_l2_raw": _bf16_arange((2, 4, 2), 500),
    }
    for name, tensor in raw.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return layer


def test_svdq_post_load_builds_five_operator_factors_and_audits_branches():
    layer = _make_raw_factor_layer()

    factors = build_svdq_operator_factors(layer)

    assert set(factors) == set(FINAL_SVDQ_FACTOR_NAMES)
    assert tuple(layer.gate_up_svdq_l1.shape) == (2, 3, 4)
    assert torch.equal(layer.gate_up_svdq_l1[:, :2], layer.gate_svd_l1_raw)
    assert torch.equal(layer.gate_up_svdq_l1[:, 2:3], layer.up_svd_l1_raw)
    assert tuple(layer.gate_svdq_l2.shape) == (2, 3, 2)
    assert tuple(layer.up_svdq_l2.shape) == (2, 3, 1)
    assert tuple(layer.down_svdq_l1.shape) == (2, 2, 3)
    assert tuple(layer.down_svdq_l2.shape) == (2, 4, 2)
    assert layer.svdq_gate_rank == 2
    assert layer.svdq_up_rank == 1
    assert layer.svdq_down_rank == 2
    assert layer.svdq_gate_rank_offset == 0
    assert layer.svdq_up_rank_offset == 2

    audit = audit_svdq_operator_factors(layer, max_experts=2, num_tokens=2)
    assert audit["passed"]
    assert audit["max_abs"] == 0.0
    assert audit["all_finite"]
    assert audit["bf16_stage_names"] == list(SVDQ_BF16_DEBUG_STAGE_NAMES)
    assert audit["bf16_stage_max_abs"] == 0.0
    assert audit["bf16_stage_all_finite"]
    assert len(audit["branch_errors"]) == 2
    for entry in audit["branch_errors"]:
        assert set(entry["stage_shapes"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
        assert entry["stage_shapes"]["routing_input"] == [2, 4]
        assert entry["stage_shapes"]["gate_up_l1_rank"] == [2, 3]
        assert entry["stage_shapes"]["gate_rank_split"] == [2, 2]
        assert entry["stage_shapes"]["up_rank_split"] == [2, 1]
        assert entry["stage_shapes"]["gate_l2_output"] == [2, 3]
        assert entry["stage_shapes"]["up_l2_output"] == [2, 3]
        assert entry["stage_shapes"]["down_l1_rank"] == [2, 2]
        assert entry["stage_shapes"]["down_l2_output"] == [2, 4]
        assert set(entry["stage_errors"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
        for stage_name, stage_error in entry["stage_errors"].items():
            assert stage_error["max_abs"] == 0.0
            assert stage_error["mean_abs"] == 0.0
            assert stage_error["max_signal_relative"] == 0.0
            assert stage_error["mean_signal_relative"] == 0.0
            assert stage_error["actual_finite"]
            assert stage_error["expected_finite"]
            assert stage_error["diff_finite"]
            assert stage_error["numel"] == torch.tensor(entry["stage_shapes"][stage_name]).prod().item()
        assert entry["branch_isolation"] == {
            "gate_rank_perturb_does_not_change_up_l2": {
                "max_abs": 0.0,
                "mean_abs": 0.0,
            },
            "up_rank_perturb_does_not_change_gate_l2": {
                "max_abs": 0.0,
                "mean_abs": 0.0,
            },
        }
    assert audit["rank_metadata"] == {
        "gate_rank": 2,
        "up_rank": 1,
        "down_rank": 2,
        "gate_rank_offset": 0,
        "up_rank_offset": 2,
    }
    factor_metadata = audit["factor_metadata"]
    assert set(factor_metadata) == set(FINAL_SVDQ_FACTOR_NAMES)
    gate_up_metadata = factor_metadata["gate_up_svdq_l1"]
    assert gate_up_metadata["name"] == "gate_up_svdq_l1"
    assert gate_up_metadata["dtype"] == "torch.bfloat16"
    assert gate_up_metadata["device"] == "cpu"
    assert gate_up_metadata["logical_shape"] == [2, 3, 4]
    assert gate_up_metadata["physical_shape"] == [2, 3, 4]
    assert gate_up_metadata["stride"] == [12, 4, 1]
    assert gate_up_metadata["storage_size_bytes"] >= gate_up_metadata["numel"] * gate_up_metadata["element_size_bytes"]
    assert gate_up_metadata["storage_offset"] == 0
    assert gate_up_metadata["element_size_bytes"] == 2
    assert gate_up_metadata["numel"] == 24
    assert gate_up_metadata["npu_format"] == "not_npu"
    assert gate_up_metadata["expert_dimension"] == 0
    assert gate_up_metadata["tp_local_dimensions"] == {
        "expert": 2,
        "rank": 3,
        "hidden": 4,
    }
    assert isinstance(gate_up_metadata["sample_checksum"], str)
    assert len(gate_up_metadata["sample_checksum"]) == 64

    assert factor_metadata["gate_svdq_l2"]["tp_local_dimensions"] == {
        "expert": 2,
        "intermediate": 3,
        "rank": 2,
    }
    assert factor_metadata["up_svdq_l2"]["tp_local_dimensions"] == {
        "expert": 2,
        "intermediate": 3,
        "rank": 1,
    }
    assert factor_metadata["down_svdq_l1"]["tp_local_dimensions"] == {
        "expert": 2,
        "rank": 2,
        "intermediate": 3,
    }
    assert factor_metadata["down_svdq_l2"]["tp_local_dimensions"] == {
        "expert": 2,
        "hidden": 4,
        "rank": 2,
    }


def test_svdq_bf16_stage_reference_exposes_e2_e7_intermediates():
    layer = _make_raw_factor_layer()
    build_svdq_operator_factors(layer)

    x = _bf16_arange((2, 4), 600).float()
    hidden = _bf16_arange((2, 3), 700).float()
    stage_audit = build_svdq_bf16_stage_reference(layer, expert=1, x=x, hidden=hidden, num_tokens=2)

    assert stage_audit["expert"] == 1
    assert set(stage_audit["actual"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
    assert set(stage_audit["reference"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
    assert set(stage_audit["stage_shapes"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
    assert set(stage_audit["stage_errors"]) == set(SVDQ_BF16_DEBUG_STAGE_NAMES)
    assert stage_audit["rank_metadata"] == {
        "gate_rank": 2,
        "up_rank": 1,
        "down_rank": 2,
        "gate_rank_offset": 0,
        "up_rank_offset": 2,
    }
    for stage_name in SVDQ_BF16_DEBUG_STAGE_NAMES:
        assert torch.equal(stage_audit["actual"][stage_name], stage_audit["reference"][stage_name])
        assert stage_audit["stage_errors"][stage_name]["max_abs"] == 0.0
    assert stage_audit["branch_isolation"] == {
        "gate_rank_perturb_does_not_change_up_l2": {
            "max_abs": 0.0,
            "mean_abs": 0.0,
        },
        "up_rank_perturb_does_not_change_gate_l2": {
            "max_abs": 0.0,
            "mean_abs": 0.0,
        },
    }


def test_svdq_runtime_payload_requires_complete_factor_contract():
    hidden_states = torch.empty(1, 4)
    topk_weights = torch.empty(1, 1)
    topk_ids = torch.empty(1, 1, dtype=torch.int32)
    w1 = torch.empty(2, 4, 1, dtype=torch.int32)
    w2 = torch.empty(2, 3, 1, dtype=torch.int32)
    gate_up_l1 = torch.empty(2, 3, 4, dtype=torch.bfloat16)
    gate_l2 = torch.empty(2, 3, 2, dtype=torch.bfloat16)
    up_l2 = torch.empty(2, 3, 1, dtype=torch.bfloat16)
    down_l1 = torch.empty(2, 2, 3, dtype=torch.bfloat16)
    down_l2 = torch.empty(2, 4, 2, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="all SVDQ factor tensors"):
        build_fused_experts_input(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            quant_type=QuantType.W4A8_SVDQ,
            dynamic_eplb=False,
            gate_up_svdq_l1=gate_up_l1,
        )

    payload = build_fused_experts_input(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1=w1,
        w2=w2,
        quant_type=QuantType.W4A8_SVDQ,
        dynamic_eplb=False,
        gate_up_svdq_l1=gate_up_l1,
        gate_svdq_l2=gate_l2,
        up_svdq_l2=up_l2,
        down_svdq_l1=down_l1,
        down_svdq_l2=down_l2,
        gate_rank=2,
        up_rank=1,
        down_rank=2,
        gate_rank_offset=0,
        up_rank_offset=2,
    )

    assert payload.weights.svdq is not None
    assert payload.weights.svdq.gate_up_svdq_l1 is gate_up_l1
    assert payload.weights.svdq.gate_rank == 2
    assert payload.weights.svdq.up_rank_offset == 2


def test_svdq_apply_builds_runtime_payload(monkeypatch):
    layer = _make_raw_factor_layer()
    build_svdq_operator_factors(layer)
    layer.swiglu_limit = 0
    layer.n_shared_experts = 0
    layer.w13_weight = torch.nn.Parameter(torch.empty(2, 4, 1, dtype=torch.int32), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(torch.empty(2, 3, 1, dtype=torch.int32), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.int64), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.int64), requires_grad=False)
    layer.w13_scale_bias = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.float32), requires_grad=False)
    layer.w2_scale_bias = torch.nn.Parameter(torch.empty(2, 4, dtype=torch.float32), requires_grad=False)

    method = AscendW4A8SVDQFusedMoEMethod.__new__(AscendW4A8SVDQFusedMoEMethod)
    method.dynamic_eplb = False
    method.is_per_channel_weight = True

    captured = {}

    class FakeComm:
        def fused_experts(self, fused_experts_input):
            captured["payload"] = fused_experts_input
            return torch.full_like(fused_experts_input.hidden_states, 7)

    monkeypatch.setattr(w4a8_svdq, "_EXTRA_CTX", SimpleNamespace(moe_comm_method=FakeComm()))
    monkeypatch.setattr(
        w4a8_svdq,
        "select_experts",
        lambda **_: (
            torch.ones(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.int32),
        ),
    )

    out = method.apply(
        layer=layer,
        x=torch.empty(2, 4, dtype=torch.bfloat16),
        router_logits=torch.empty(2, 2, dtype=torch.float32),
        top_k=1,
        renormalize=True,
        num_experts=2,
    )

    assert torch.equal(out, torch.full((2, 4), 7, dtype=torch.bfloat16))
    payload = captured["payload"]
    assert payload.quant.quant_type is QuantType.W4A8_SVDQ
    assert payload.weights.svdq is not None
    assert payload.weights.svdq.gate_up_svdq_l1 is layer.gate_up_svdq_l1
    assert payload.weights.svdq.gate_svdq_l2 is layer.gate_svdq_l2
    assert payload.weights.svdq.up_svdq_l2 is layer.up_svdq_l2
    assert payload.weights.svdq.gate_rank_offset == 0
    assert payload.weights.svdq.up_rank_offset == layer.svdq_gate_rank
