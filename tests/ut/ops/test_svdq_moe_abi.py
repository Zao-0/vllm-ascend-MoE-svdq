#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_comm_method import FusedMC2CommImpl
from vllm_ascend.ops.fused_moe.moe_stage_contracts import MoESVDQWeights
from vllm_ascend.ops.fused_moe.svdq_abi import SVDQ_OPERATOR_ARG_ORDER, validate_svdq_operator_abi

REPO_ROOT = Path(__file__).resolve().parents[3]


def _valid_inputs():
    x = torch.empty(4, 8, dtype=torch.bfloat16)
    return {
        "x": x,
        "weight1": [torch.empty(2, 8, 1, dtype=torch.int32)],
        "weight2": [torch.empty(2, 4, 1, dtype=torch.int32)],
        "expert_idx": torch.zeros(4, 2, dtype=torch.int32),
        "scale1": [torch.empty(2, 4, dtype=torch.int64)],
        "scale2": [torch.empty(2, 8, dtype=torch.int64)],
        "bias1": [torch.empty(2, 4, dtype=torch.float32)],
        "bias2": [torch.empty(2, 8, dtype=torch.float32)],
        "probs": torch.empty(4, 2, dtype=torch.float32),
        "svdq": MoESVDQWeights(
            gate_up_svdq_l1=torch.empty(2, 5, 8, dtype=torch.bfloat16),
            gate_svdq_l2=torch.empty(2, 4, 3, dtype=torch.bfloat16),
            up_svdq_l2=torch.empty(2, 4, 2, dtype=torch.bfloat16),
            down_svdq_l1=torch.empty(2, 3, 4, dtype=torch.bfloat16),
            down_svdq_l2=torch.empty(2, 8, 3, dtype=torch.bfloat16),
            gate_rank=3,
            up_rank=2,
            down_rank=3,
            gate_rank_offset=0,
            up_rank_offset=3,
        ),
        "group": "dummy_group",
        "max_output_size": 128,
        "swiglu_limit": 0,
        "x_active_mask": None,
        "out": torch.empty_like(x),
        "expert_token_nums": torch.empty(2, dtype=torch.int32),
    }


def test_svdq_operator_abi_validates_explicit_five_factor_contract():
    validate_svdq_operator_abi(**_valid_inputs())

    assert "gate_up_svdq_l1" in SVDQ_OPERATOR_ARG_ORDER
    assert "gate_svdq_l2" in SVDQ_OPERATOR_ARG_ORDER
    assert "up_svdq_l2" in SVDQ_OPERATOR_ARG_ORDER
    assert "gate_up_svdq_l2" not in SVDQ_OPERATOR_ARG_ORDER


def test_svdq_operator_abi_rejects_bad_rank_offsets():
    inputs = _valid_inputs()
    inputs["svdq"] = replace(inputs["svdq"], up_rank_offset=4)

    with pytest.raises(ValueError, match="up_rank_offset"):
        validate_svdq_operator_abi(**inputs)


def test_svdq_operator_abi_rejects_shared_gate_up_l2_tensor():
    inputs = _valid_inputs()
    shared = torch.empty(2, 4, 3, dtype=torch.bfloat16)
    inputs["svdq"] = MoESVDQWeights(
        gate_up_svdq_l1=torch.empty(2, 6, 8, dtype=torch.bfloat16),
        gate_svdq_l2=shared,
        up_svdq_l2=shared,
        down_svdq_l1=torch.empty(2, 3, 4, dtype=torch.bfloat16),
        down_svdq_l2=torch.empty(2, 8, 3, dtype=torch.bfloat16),
        gate_rank=3,
        up_rank=3,
        down_rank=3,
        gate_rank_offset=0,
        up_rank_offset=3,
    )

    with pytest.raises(ValueError, match="separate tensors|share storage"):
        validate_svdq_operator_abi(**inputs)


def test_svdq_operator_registration_exists():
    import vllm_ascend.ops.register_custom_ops  # noqa: F401

    assert hasattr(torch.ops.vllm, "dispatch_ffn_combine_w4a8_svdq")


def test_fused_mc2_call_site_uses_dedicated_svdq_operator():
    source = inspect.getsource(FusedMC2CommImpl.fused_experts)

    assert "torch.ops._C_ascend.dispatch_ffn_combine_w4a8_svdq" in source
    assert "dispatch_ffn_combine_w4a8_svdq" in source
    for name in (
        "gate_up_svdq_l1",
        "gate_svdq_l2",
        "up_svdq_l2",
        "down_svdq_l1",
        "down_svdq_l2",
        "gate_rank_offset",
        "up_rank_offset",
    ):
        assert name in source


def test_svdq_csrc_torch_schema_meta_and_adapter_are_registered():
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
    adapter = (
        REPO_ROOT
        / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/dispatch_ffn_combine_w4_a8_svdq_torch_adpt.h"
    ).read_text()

    assert "dispatch_ffn_combine_w4a8_svdq(Tensor x" in binding
    assert "ops.impl(\"dispatch_ffn_combine_w4a8_svdq\", torch::kPrivateUse1" in binding
    assert "dispatch_ffn_combine_w4a8_svdq_meta" in meta
    assert "ops.impl(\"dispatch_ffn_combine_w4a8_svdq\"" in meta
    assert "dispatch_ffn_combine_w4a8_svdq(" in adapter
    assert "EXEC_NPU_CMD(" in adapter
    assert "aclnnDispatchFFNCombineW4A8SVDQ" in adapter
    assert "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.h" in adapter

    for name in (
        "gate_up_svdq_l1",
        "gate_svdq_l2",
        "up_svdq_l2",
        "down_svdq_l1",
        "down_svdq_l2",
        "gate_rank",
        "up_rank",
        "down_rank",
        "gate_rank_offset",
        "up_rank_offset",
    ):
        assert name in binding
        assert name in meta
        assert name in adapter

    assert "gate_up_svdq_l2" not in binding
    assert "gate_up_svdq_l2" not in meta
    assert "gate_up_svdq_l2" not in adapter

    for meta_check in (
        'x.scalar_type() == at::kBFloat16',
        'expert_idx.scalar_type() == at::kInt',
        'probs.scalar_type() == at::kFloat',
        'out.scalar_type() == at::kBFloat16',
        'expert_token_nums.scalar_type() == at::kInt',
        '!weight1.empty()',
        '!weight2.empty()',
        '!scale1.empty()',
        '!scale2.empty()',
        '!bias1.empty()',
        '!bias2.empty()',
        'weight1[0].scalar_type() == at::kInt',
        'weight2[0].scalar_type() == at::kInt',
        'scale1[0].scalar_type() == at::kLong',
        'scale2[0].scalar_type() == at::kLong',
        'bias1[0].scalar_type() == at::kFloat',
        'bias2[0].scalar_type() == at::kFloat',
        'hidden_size == x.size(1)',
        'gate_svdq_l2.size(0) == num_experts && gate_svdq_l2.size(2) == gate_rank',
        'up_svdq_l2.size(0) == num_experts && up_svdq_l2.size(1) == intermediate_size',
        'down_svdq_l1.size(0) == num_experts && down_svdq_l1.size(1) == down_rank',
        'down_svdq_l2.size(0) == num_experts && down_svdq_l2.size(1) == hidden_size',
        'weight1[0].size(0) == num_experts && weight2[0].size(0) == num_experts',
        'expert_token_nums.size(0) == num_experts',
        'x_active_mask.value().scalar_type() == at::kBool',
    ):
        assert meta_check in meta

    adapter_call_order = (
        "x,",
        "weight1,",
        "weight2,",
        "expert_idx,",
        "scale1,",
        "scale2,",
        "bias1,",
        "bias2,",
        "probs,",
        "gate_up_svdq_l1,",
        "gate_svdq_l2,",
        "up_svdq_l2,",
        "down_svdq_l1,",
        "down_svdq_l2,",
        "gate_rank,",
        "up_rank,",
        "down_rank,",
        "gate_rank_offset,",
        "up_rank_offset,",
        "x_active_mask.has_value()",
        "group_ep_ptr,",
        "max_output_size,",
        "swiglu_limit,",
        "out,",
        "expert_token_nums",
    )
    call_start = adapter.index("EXEC_NPU_CMD(")
    last = call_start
    for token in adapter_call_order:
        current = adapter.index(token, last)
        assert current >= last
        last = current


def test_svdq_cann_op_host_surface_uses_canonical_five_factor_abi():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    cmake = (op_root / "op_host/CMakeLists.txt").read_text()
    header = (op_root / "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.h").read_text()
    wrapper = (op_root / "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.cpp").read_text()
    op_def = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_def.cpp").read_text()
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()

    assert "DispatchFFNCombineW4A8SVDQ" in cmake
    assert "dispatch_ffn_combine_w4_a8_svdq" in cmake
    assert "aclnnDispatchFFNCombineW4A8SVDQGetWorkspaceSize" in header
    assert "aclnnInnerDispatchFFNCombineW4A8SVDQGetWorkspaceSize" in wrapper
    assert "OP_ADD(DispatchFFNCombineW4A8SVDQ)" in op_def
    assert "IMPL_OP_OPTILING(DispatchFFNCombineW4A8SVDQ)" in tiling
    assert "AscendC kernel is not implemented yet" in tiling

    canonical_order = (
        "gateUpSvdqL1",
        "gateSvdqL2",
        "upSvdqL2",
        "downSvdqL1",
        "downSvdqL2",
        "gateRank",
        "upRank",
        "downRank",
        "gateRankOffset",
        "upRankOffset",
    )
    for name in canonical_order:
        assert name in header
        assert name in wrapper
        assert name in op_def or name.upper() in tiling

    for source in (header, wrapper, op_def, tiling):
        assert "gateUpSvdqL2" not in source
        assert "gate_up_svdq_l2" not in source

    assert "constexpr uint32_t GATE_UP_SVDQ_L1_INDEX = 9" in tiling
    assert "constexpr uint32_t GATE_SVDQ_L2_INDEX = 10" in tiling
    assert "constexpr uint32_t UP_SVDQ_L2_INDEX = 11" in tiling
    assert "constexpr uint32_t DOWN_SVDQ_L1_INDEX = 12" in tiling
    assert "constexpr uint32_t DOWN_SVDQ_L2_INDEX = 13" in tiling


def test_svdq_cann_tiling_workspace_map_matches_required_dataflow():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()

    assert "SVDQ_WORKSPACE_REGION_COUNT = 12" in tiling_header
    assert "SVDQWorkspaceRegion workspaceRegions[SVDQ_WORKSPACE_REGION_COUNT]" in tiling_header
    assert "workspaceBytes" in tiling_header
    assert "BuildWorkspaceMap" in tiling
    assert "workSpaces[0] = SVDQ_SYSTEM_WORKSPACE + info.workspaceBytes" in tiling

    for region in (
        "SVDQ_REGION_EXPANDED_ROW_IDX",
        "SVDQ_REGION_ROUTED_X",
        "SVDQ_REGION_X_Q",
        "SVDQ_REGION_X_SCALE",
        "SVDQ_REGION_PROJECTION_1",
        "SVDQ_REGION_ACCUMULATOR_1",
        "SVDQ_REGION_HIDDEN",
        "SVDQ_REGION_HIDDEN_Q",
        "SVDQ_REGION_HIDDEN_SCALE",
        "SVDQ_REGION_PROJECTION_2",
        "SVDQ_REGION_ACCUMULATOR_2",
        "SVDQ_REGION_PEER_OUTPUT",
    ):
        assert region in tiling_header
        assert region in tiling

    for stage in (
        "SVDQ_STAGE_BF16_DISPATCH",
        "SVDQ_STAGE_QUANT_1",
        "SVDQ_STAGE_LOWRANK_1",
        "SVDQ_STAGE_W4A8_GEMM_1",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_QUANT_2",
        "SVDQ_STAGE_LOWRANK_2",
        "SVDQ_STAGE_W4A8_GEMM_2",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_STAGE_UNPERMUTE_COMBINE",
    ):
        assert stage in tiling_header
        assert stage in tiling

    assert "SVDQ_DTYPE_INT8" in tiling_header
    assert "SVDQ_DTYPE_INT32" in tiling_header
    assert "SVDQ_DTYPE_BF16" in tiling_header
    assert "SVDQ_DTYPE_FP32" in tiling_header
    assert "SVDQ_WORKSPACE_ALIGNMENT = 512" in tiling
    assert "routedRows * gateUpSize * BF16_BYTES" in tiling
    assert "routedRows * gateUpSize * INT32_BYTES" in tiling
    assert "routedRows * hiddenSize * BF16_BYTES" in tiling
    assert "AscendC kernel is not implemented yet" in tiling


def test_svdq_cann_tiling_validates_dtype_and_all_factor_shapes():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()

    assert "DispatchFFNCombineW4A8SVDQCheckDType" in tiling
    assert "CheckRequiredInputDType" in tiling
    assert "CheckDynamicInputDType" in tiling
    assert "CheckOutputDType" in tiling
    assert "DispatchFFNCombineW4A8SVDQCheckDType(context)" in tiling

    for expected_check in (
        'CheckRequiredInputDType(context, X_INDEX, "x", ge::DT_BF16)',
        'CheckDynamicInputDType(context, WEIGHT1_INDEX, "w1", ge::DT_INT32)',
        'CheckDynamicInputDType(context, WEIGHT2_INDEX, "w2", ge::DT_INT32)',
        'CheckRequiredInputDType(context, EXPERT_ID_INDEX, "expertIdx", ge::DT_INT32)',
        'CheckDynamicInputDType(context, SCALE1_INDEX, "scale1", ge::DT_INT64)',
        'CheckDynamicInputDType(context, SCALE2_INDEX, "scale2", ge::DT_INT64)',
        'CheckDynamicInputDType(context, BIAS1_INDEX, "bias1", ge::DT_FLOAT)',
        'CheckDynamicInputDType(context, BIAS2_INDEX, "bias2", ge::DT_FLOAT)',
        'CheckRequiredInputDType(context, PROBS_INDEX, "probs", ge::DT_FLOAT)',
        'CheckRequiredInputDType(context, GATE_UP_SVDQ_L1_INDEX, "gateUpSvdqL1", ge::DT_BF16)',
        'CheckRequiredInputDType(context, GATE_SVDQ_L2_INDEX, "gateSvdqL2", ge::DT_BF16)',
        'CheckRequiredInputDType(context, UP_SVDQ_L2_INDEX, "upSvdqL2", ge::DT_BF16)',
        'CheckRequiredInputDType(context, DOWN_SVDQ_L1_INDEX, "downSvdqL1", ge::DT_BF16)',
        'CheckRequiredInputDType(context, DOWN_SVDQ_L2_INDEX, "downSvdqL2", ge::DT_BF16)',
        'CheckOutputDType(context, OUT_INDEX, "out", ge::DT_BF16)',
        'CheckOutputDType(context, EXPERT_TOKEN_NUMS_INDEX, "expertTokenNums", ge::DT_INT32)',
        "xActiveMaskDesc->GetDataType() != ge::DT_BOOL",
    ):
        assert expected_check in tiling

    for expected_shape_check in (
        "gateL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.gateRank)",
        "upL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.intermediateSize)",
        "upL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.upRank)",
        "downL1Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.downRank)",
        "downL1Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.intermediateSize)",
        "downL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.hiddenSize)",
        "downL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.downRank)",
        "outShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.m)",
        "outShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.hiddenSize)",
        "expertTokenNumsShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.expertPerRank)",
    ):
        assert expected_shape_check in tiling


def test_svdq_cann_tiling_sync_flags_match_required_dataflow():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()

    assert "SVDQ_SYNC_FLAG_COUNT = 14" in tiling_header
    assert "uint32_t syncFlagCount" in tiling_header
    assert "SVDQSyncFlag syncFlags[SVDQ_SYNC_FLAG_COUNT]" in tiling_header
    assert "BuildSyncFlagTable" in tiling
    assert "SetSyncFlag" in tiling
    assert "tilingData->info.syncFlagCount = SVDQ_SYNC_FLAG_COUNT" in tiling
    assert "BuildSyncFlagTable(tilingData)" in tiling
    assert "AscendC kernel is not implemented yet" in tiling

    expected_flags = (
        (
            "SVDQ_SYNC_DISPATCH_TO_QUANT_1",
            "SVDQ_STAGE_BF16_DISPATCH",
            "SVDQ_STAGE_QUANT_1",
            "SVDQ_REGION_ROUTED_X",
            "0",
        ),
        (
            "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1",
            "SVDQ_STAGE_BF16_DISPATCH",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_ROUTED_X",
            "1",
        ),
        (
            "SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1",
            "SVDQ_STAGE_QUANT_1",
            "SVDQ_STAGE_W4A8_GEMM_1",
            "SVDQ_REGION_X_Q",
            "2",
        ),
        (
            "SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1",
            "SVDQ_STAGE_QUANT_1",
            "SVDQ_STAGE_MIXED_EPILOGUE_1",
            "SVDQ_REGION_X_SCALE",
            "3",
        ),
        (
            "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_STAGE_MIXED_EPILOGUE_1",
            "SVDQ_REGION_PROJECTION_1",
            "4",
        ),
        (
            "SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1",
            "SVDQ_STAGE_W4A8_GEMM_1",
            "SVDQ_STAGE_MIXED_EPILOGUE_1",
            "SVDQ_REGION_ACCUMULATOR_1",
            "5",
        ),
        (
            "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2",
            "SVDQ_STAGE_MIXED_EPILOGUE_1",
            "SVDQ_STAGE_QUANT_2",
            "SVDQ_REGION_HIDDEN",
            "6",
        ),
        (
            "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2",
            "SVDQ_STAGE_MIXED_EPILOGUE_1",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_REGION_HIDDEN",
            "7",
        ),
        (
            "SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2",
            "SVDQ_STAGE_QUANT_2",
            "SVDQ_STAGE_W4A8_GEMM_2",
            "SVDQ_REGION_HIDDEN_Q",
            "8",
        ),
        (
            "SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_STAGE_QUANT_2",
            "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_REGION_HIDDEN_SCALE",
            "9",
        ),
        (
            "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_REGION_PROJECTION_2",
            "10",
        ),
        (
            "SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_STAGE_W4A8_GEMM_2",
            "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_REGION_ACCUMULATOR_2",
            "11",
        ),
        (
            "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE",
            "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
            "SVDQ_STAGE_UNPERMUTE_COMBINE",
            "SVDQ_REGION_PEER_OUTPUT",
            "12",
        ),
        (
            "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE",
            "SVDQ_STAGE_BF16_DISPATCH",
            "SVDQ_STAGE_UNPERMUTE_COMBINE",
            "SVDQ_REGION_EXPANDED_ROW_IDX",
            "13",
        ),
    )
    for flag, producer, consumer, region, index in expected_flags:
        assert flag in tiling_header
        assert flag in tiling
        call = f"SetSyncFlag(tilingData, {flag}, {producer},\n        {consumer}, {region}, {index})"
        assert call in tiling


def test_svdq_cann_kernel_contract_resolves_factors_workspace_and_bf16_stages():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    kernel = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.cpp").read_text()
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()

    assert '#include "dispatch_ffn_combine_w4_a8_svdq.h"' in kernel
    assert "DispatchFFNCombineW4A8SVDQ op" in kernel
    assert "op.Init(x, w1, w2, expertId, scale1, scale2, bias1, bias2, probs, gateUpSvdqL1" in kernel
    assert "op.Process()" in kernel

    assert "SVDQ_FACTOR_COUNT = 5" in contract
    assert "SVDQ_BF16_STAGE_COUNT = 7" in contract
    assert "SVDQFactorGM" in contract
    assert "SVDQResidualGM" in contract
    assert "SVDQRuntimeGM" in contract
    assert "SVDQWorkspaceGM" in contract
    assert "SVDQBF16StageContract" in contract
    assert "GET_TILING_DATA(tilingData, tilingGM)" in contract
    assert "WorkspaceAddress(uint32_t regionId)" in contract
    assert "runtime_.workspace + tilingData_.workspaceRegions[regionId].offset" in contract
    assert "WorkspaceRegion(uint32_t regionId)" in contract
    assert "SyncFlag(uint32_t flagId)" in contract
    assert "HasCompleteTilingContract" in contract
    assert "tilingData_.info.syncFlagCount == SVDQ_SYNC_FLAG_COUNT" in contract
    assert "tilingData_.info.upRankOffset == tilingData_.info.gateRank" in contract

    for factor in (
        "gateUpSvdqL1",
        "gateSvdqL2",
        "upSvdqL2",
        "downSvdqL1",
        "downSvdqL2",
    ):
        assert factor in contract

    for region_field, region_id in (
        ("expandedRowIdx", "SVDQ_REGION_EXPANDED_ROW_IDX"),
        ("routedX", "SVDQ_REGION_ROUTED_X"),
        ("xQ", "SVDQ_REGION_X_Q"),
        ("xScale", "SVDQ_REGION_X_SCALE"),
        ("projection1", "SVDQ_REGION_PROJECTION_1"),
        ("accumulator1", "SVDQ_REGION_ACCUMULATOR_1"),
        ("hidden", "SVDQ_REGION_HIDDEN"),
        ("hiddenQ", "SVDQ_REGION_HIDDEN_Q"),
        ("hiddenScale", "SVDQ_REGION_HIDDEN_SCALE"),
        ("projection2", "SVDQ_REGION_PROJECTION_2"),
        ("accumulator2", "SVDQ_REGION_ACCUMULATOR_2"),
        ("peerOutput", "SVDQ_REGION_PEER_OUTPUT"),
    ):
        assert f"workspace_.{region_field} = WorkspaceAddress({region_id})" in contract

    expected_bf16_stages = (
        (
            "SVDQ_BF16_STAGE_ROUTING",
            "SVDQ_STAGE_BF16_DISPATCH",
            "SVDQ_INVALID_ID",
            "SVDQ_REGION_ROUTED_X",
            "SVDQ_INVALID_ID",
            "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_UP_L1_GEMM",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_ROUTED_X",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_UP_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_INVALID_ID",
            "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L1_GEMM",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_REGION_HIDDEN",
            "SVDQ_REGION_PROJECTION_2",
            "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_REGION_PROJECTION_2",
            "SVDQ_REGION_PROJECTION_2",
            "SVDQ_INVALID_ID",
            "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE",
        ),
    )
    for stage, tiling_stage, input_region, output_region, wait_flag, signal_flag in expected_bf16_stages:
        assert f"case {stage}:" in contract
        assert tiling_stage in contract
        assert input_region in contract
        assert output_region in contract
        assert wait_flag in contract
        assert signal_flag in contract

    assert "gateUpSvdqL2" not in contract
    assert "gate_up_svdq_l2" not in contract
