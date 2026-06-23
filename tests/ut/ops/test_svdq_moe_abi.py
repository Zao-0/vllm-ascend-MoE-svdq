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


def test_svdq_cann_op_is_selected_by_a3_aclnn_build_script():
    build_script = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    a3_branch = build_script[build_script.index('elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]];') :]
    a3_ops = a3_branch[: a3_branch.index('elif [[ "$SOC_VERSION" =~ ^ascend950 ]];')]

    assert '"dispatch_ffn_combine_w4_a8"' in a3_ops
    assert '"dispatch_ffn_combine_w4_a8_svdq"' in a3_ops
    assert a3_ops.index('"dispatch_ffn_combine_w4_a8"') < a3_ops.index('"dispatch_ffn_combine_w4_a8_svdq"')
    assert a3_ops.index('"dispatch_ffn_combine_w4_a8_svdq"') < a3_ops.index('"dispatch_ffn_combine_bf16"')


def test_cann_host_library_build_path_honors_soc_selection():
    build_script = (REPO_ROOT / "csrc/build.sh").read_text()
    create_lib_branch = build_script[build_script.index('elif [[ "$ENABLE_CREATE_LIB" == "TRUE" ]];') :]
    create_lib_branch = create_lib_branch[: create_lib_branch.index('elif [[ "$ENABLE_STATIC" == "TRUE" ]];')]

    assert create_lib_branch.index("set_compute_unit_option") < create_lib_branch.index("build_lib")


def test_cann_graph_symbol_generation_skips_empty_filtered_op_sets():
    symbol_cmake = (REPO_ROOT / "csrc/cmake/symbol.cmake").read_text()
    graph_symbol = symbol_cmake[symbol_cmake.index("function(gen_opgraph_symbol)") :]
    graph_symbol = graph_symbol[: graph_symbol.index("function(gen_opapi_symbol)")]

    assert "if(NOT TARGET ${GRAPH_PLUGIN_NAME}_obj)" in graph_symbol
    assert "get_target_property(GRAPH_PLUGIN_SRCS ${GRAPH_PLUGIN_NAME}_obj SOURCES)" in graph_symbol
    assert "if(NOT GRAPH_PLUGIN_SRCS)" in graph_symbol
    assert "$<TARGET_OBJECTS:${GRAPH_PLUGIN_NAME}_obj>" in graph_symbol


def test_svdq_cann_tiling_workspace_map_matches_required_dataflow():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()

    assert "SVDQ_WORKSPACE_REGION_COUNT = 14" in tiling_header
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
        "SVDQ_REGION_LOWRANK_ACCUMULATOR_1",
        "SVDQ_REGION_LOWRANK_ACCUMULATOR_2",
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
    assert "routedRows * gateUpSize * FP32_BYTES" in tiling
    assert "routedRows * gateUpSize * INT32_BYTES" in tiling
    assert "routedRows * hiddenSize * BF16_BYTES" in tiling
    assert "routedRows * hiddenSize * FP32_BYTES" in tiling
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


def test_svdq_cann_tiling_records_bf16_stage_shapes_and_factor_bindings():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()

    assert "SVDQ_BF16_STAGE_COUNT = 7" in tiling_header
    assert "SVDQ_INVALID_ID = 0xffffffffU" in tiling_header
    assert "SVDQFactorId" in tiling_header
    assert "SVDQBF16StageShape" in tiling_header
    assert "SVDQBF16StageShape bf16StageShapes[SVDQ_BF16_STAGE_COUNT]" in tiling_header
    assert "SetBF16StageShape" in tiling
    assert "BuildBF16StageShapeTable" in tiling
    assert "BuildBF16StageShapeTable(tilingData)" in tiling
    assert "const uint32_t gateUpRank = info.gateRank + info.upRank" in tiling
    assert "const uint32_t upOutputOffset = info.intermediateSize" in tiling

    for factor in (
        "SVDQ_FACTOR_GATE_UP_L1",
        "SVDQ_FACTOR_GATE_L2",
        "SVDQ_FACTOR_UP_L2",
        "SVDQ_FACTOR_DOWN_L1",
        "SVDQ_FACTOR_DOWN_L2",
    ):
        assert factor in tiling_header
        assert factor in tiling
        assert f"case {factor}:" in contract

    expected_stage_shapes = (
        (
            "SVDQ_BF16_STAGE_ROUTING",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
            "SVDQ_REGION_ROUTED_X",
            "routedRows",
            "info.hiddenSize",
            "info.hiddenSize",
            "0",
            "0",
            "0",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_UP_L1_GEMM",
            "SVDQ_FACTOR_GATE_UP_L1",
            "SVDQ_REGION_ROUTED_X",
            "SVDQ_REGION_PROJECTION_1",
            "routedRows",
            "info.hiddenSize",
            "gateUpRank",
            "0",
            "0",
            "0",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_L2_GEMM",
            "SVDQ_FACTOR_GATE_L2",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_REGION_PROJECTION_1",
            "routedRows",
            "info.gateRank",
            "info.intermediateSize",
            "info.gateRankOffset",
            "gateOutputOffset",
            "0",
        ),
        (
            "SVDQ_BF16_STAGE_UP_L2_GEMM",
            "SVDQ_FACTOR_UP_L2",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_REGION_PROJECTION_1",
            "routedRows",
            "info.upRank",
            "info.intermediateSize",
            "info.upRankOffset",
            "upOutputOffset",
            "0",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L1_GEMM",
            "SVDQ_FACTOR_DOWN_L1",
            "SVDQ_REGION_HIDDEN",
            "SVDQ_REGION_PROJECTION_2",
            "routedRows",
            "info.intermediateSize",
            "info.downRank",
            "0",
            "0",
            "0",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L2_GEMM",
            "SVDQ_FACTOR_DOWN_L2",
            "SVDQ_REGION_PROJECTION_2",
            "SVDQ_REGION_PROJECTION_2",
            "routedRows",
            "info.downRank",
            "info.hiddenSize",
            "0",
            "0",
            "0",
        ),
    )
    for stage, factor, input_region, output_region, m, k, n, input_offset, output_offset, factor_offset in expected_stage_shapes:
        assert stage in tiling
        assert factor in tiling
        assert input_region in tiling
        assert output_region in tiling
        assert m in tiling
        assert k in tiling
        assert n in tiling
        assert input_offset in tiling
        assert output_offset in tiling
        assert factor_offset in tiling

    assert "BF16StageShape(uint32_t stageId)" in contract
    assert "return tilingData_.bf16StageShapes[stageId]" in contract
    assert "FactorAddress(uint32_t factorId)" in contract


def test_svdq_cann_lowrank_down_up_component_contract_is_wired():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()
    lowrank_tiling = (op_root / "op_kernel/lowrank/svdq_fused_down_up_tiling.h").read_text()
    lowrank_header = (op_root / "op_kernel/lowrank/svdq_fused_down_up.hpp").read_text()
    lowrank_cpp = (op_root / "op_kernel/lowrank/svdq_fused_down_up.cpp").read_text()

    assert '#include "lowrank/svdq_fused_down_up_tiling.h"' in tiling_header
    assert '#include "lowrank/svdq_fused_down_up.hpp"' in contract
    assert '#include "svdq_fused_down_up.hpp"' in lowrank_cpp
    assert "SVDQ_LOWRANK_INVOCATION_COUNT = 2" in lowrank_tiling
    assert "SVDQ_LOWRANK_INVOCATION_GATE_UP" in lowrank_tiling
    assert "SVDQ_LOWRANK_INVOCATION_DOWN" in lowrank_tiling
    assert "SVDQFusedDownUpTiling" in lowrank_tiling
    assert "secondUpFactorId" in lowrank_tiling
    assert "secondRankColumns" in lowrank_tiling
    assert "secondInputColumnOffset" in lowrank_tiling
    assert "secondOutputColumnOffset" in lowrank_tiling
    assert "rowTile" in lowrank_tiling
    assert "outputColumnTile" in lowrank_tiling
    assert "kTile" in lowrank_tiling
    assert "coreCount" in lowrank_tiling
    assert "accumulatorRegionId" in lowrank_tiling
    assert "SVDQFusedDownUpArgs" in lowrank_header
    assert "SVDQFusedDownUp" in lowrank_header
    assert "SVDQ_LOWRANK_BF16_BYTES = 2" in lowrank_header
    assert "expertPerRank" in lowrank_header
    assert "HasIndependentSecondUp" in lowrank_header
    assert "SVDQLowRankStageKind" in lowrank_header
    assert "SVDQ_LOWRANK_STAGE_DOWN_PROJECT" in lowrank_header
    assert "SVDQ_LOWRANK_STAGE_UP_PROJECT" in lowrank_header
    assert "SVDQ_LOWRANK_STAGE_SECOND_UP_PROJECT" in lowrank_header
    assert "SVDQLowRankStagePlan" in lowrank_header
    assert "SVDQLowRankExpertPlan" in lowrank_header
    assert "SVDQLowRankCoreTileRange" in lowrank_header
    assert "SVDQLowRankTilePlan" in lowrank_header
    assert "SVDQLowRankTileTensorPlan" in lowrank_header
    assert "SVDQLowRankMmadTilePlan" in lowrank_header
    assert "SVDQLowRankMmadBufferPlan" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_M_TILE = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_N_TILE = 64" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_K_TILE = 64" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_M_ALIGNMENT = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_N_ALIGNMENT = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_K_ALIGNMENT = 16" in lowrank_header
    assert "TotalRankColumns() const" in lowrank_header
    assert "PrimaryOutputColumns() const" in lowrank_header
    assert "StageCount() const" in lowrank_header
    assert "ExpertCount() const" in lowrank_header
    assert "ExpertTokenCount(uint32_t expertId)" in lowrank_header
    assert "ExpertTokenStart(uint32_t expertId)" in lowrank_header
    assert "StagePlan(uint32_t stageIndex)" in lowrank_header
    assert "ExpertStagePlan(uint32_t stageIndex, uint32_t expertId)" in lowrank_header
    assert "StageColumnTileCount(const SVDQLowRankStagePlan& stage)" in lowrank_header
    assert "StageKTileCount(const SVDQLowRankStagePlan& stage)" in lowrank_header
    assert "ExpertRowTileCount(uint32_t tokenCount)" in lowrank_header
    assert "ExpertTileCount(uint32_t stageIndex, uint32_t expertId)" in lowrank_header
    assert "StageTileCount(uint32_t stageIndex)" in lowrank_header
    assert "InvocationTileCount() const" in lowrank_header
    assert "CoreTileRange(uint32_t coreIdx, uint32_t coreCount)" in lowrank_header
    assert "TilePlan(uint32_t tileId)" in lowrank_header
    assert "BuildTileTensorPlan(" in lowrank_header
    assert "BuildMmadTilePlan(" in lowrank_header
    assert "BuildMmadBufferPlan(" in lowrank_header
    assert "HasCompatibleShape() const" in lowrank_header
    assert "HasCompleteFootprint() const" in lowrank_header
    assert "InputElementOffset(" in lowrank_header
    assert "FactorElementOffset(" in lowrank_header
    assert "OutputElementOffset(" in lowrank_header
    assert "LoadInputBF16(" in lowrank_header
    assert "LoadFactorBF16(" in lowrank_header
    assert "LoadOutputBF16(" in lowrank_header
    assert "LoadAccumulatorFP32(" in lowrank_header
    assert "StoreOutputBF16(" in lowrank_header
    assert "StoreAccumulatorFP32(" in lowrank_header
    assert "AccumulateScalarBF16(" in lowrank_header
    assert "RunScalarTileBF16(" in lowrank_header
    assert "RunPlannedTileBF16(" in lowrank_header
    assert "BuildDownStagePlan() const" in lowrank_header
    assert "BuildPrimaryUpStagePlan() const" in lowrank_header
    assert "BuildSecondUpStagePlan() const" in lowrank_header
    assert "CeilDiv(uint32_t value, uint32_t divisor)" in lowrank_header
    assert "Min(uint32_t lhs, uint32_t rhs)" in lowrank_header
    assert "RoundUp(uint32_t value, uint32_t alignment)" in lowrank_header
    assert "MatrixAddress(" in lowrank_header
    assert "FactorAddress(const SVDQLowRankStagePlan& stage, uint32_t expertId)" in lowrank_header
    assert "AccumulatorAddress(" in lowrank_header
    assert "FactorTileAddress(" in lowrank_header
    assert "GM_ADDR inputBase = args_.input" in lowrank_header
    assert "if (stageIndex != 0)" in lowrank_header
    assert "inputBase = args_.output" in lowrank_header
    assert "IsImplemented() const" in lowrank_header
    assert "return false" in lowrank_header
    assert "args_.tiling.invocationId < SVDQ_LOWRANK_INVOCATION_COUNT" in lowrank_header
    assert "args_.expertPerRank > 0" in lowrank_header
    assert "args_.tiling.rowTile > 0" in lowrank_header
    assert "args_.tiling.outputColumnTile > 0" in lowrank_header
    assert "args_.tiling.kTile > 0" in lowrank_header
    assert "args_.tiling.coreCount > 0" in lowrank_header
    assert "args_.accumulator != nullptr" in lowrank_header
    assert "args_.tiling.secondInputColumnOffset + args_.tiling.secondRankColumns <= TotalRankColumns()" in lowrank_header
    assert "args_.tiling.outputColumnOffset < args_.tiling.secondOutputColumnOffset" in lowrank_header
    assert "for (uint32_t stageIndex = 0; stageIndex < StageCount(); ++stageIndex)" in lowrank_header
    assert "for (uint32_t expertId = 0; expertId < ExpertCount(); ++expertId)" in lowrank_header
    assert "stage.HasCompleteContract()" in lowrank_header
    assert "expertPlan.tokenCount > 0 && !expertPlan.HasWork()" in lowrank_header
    assert "expertTokenNums.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(args_.expertTokenNums))" in lowrank_header
    assert "static_cast<uint64_t>(row) * strideColumns + columnOffset" in lowrank_header
    assert "static_cast<uint64_t>(expertId) * stage.inputColumns * stage.outputColumns" in lowrank_header
    assert "static_cast<uint64_t>(outputColumnOffset) * expert.stage.inputColumns + kColumnOffset" in lowrank_header
    assert "MatrixAddress(expert.input, rowOffset, stage.inputStrideColumns, tilePlan.kColumnOffset)" in lowrank_header
    assert "FactorTileAddress(expert, tilePlan.outputColumnOffset, tilePlan.kColumnOffset)" in lowrank_header
    assert "MatrixAddress(expert.output, rowOffset, stage.outputStrideColumns, tilePlan.outputColumnOffset)" in lowrank_header
    assert "AccumulatorAddress(expert, rowOffset, tilePlan.outputColumnOffset)" in lowrank_header
    assert "const uint32_t mActual = tilePlan.tile.rowCount" in lowrank_header
    assert "const uint32_t nActual = tilePlan.tile.outputColumnCount" in lowrank_header
    assert "const uint32_t kActual = tilePlan.tile.kColumnCount" in lowrank_header
    assert "const uint32_t mRound = RoundUp(mActual, SVDQ_LOWRANK_MMAD_M_ALIGNMENT)" in lowrank_header
    assert "const uint32_t nRound = RoundUp(nActual, SVDQ_LOWRANK_MMAD_N_ALIGNMENT)" in lowrank_header
    assert "const uint32_t kRound = RoundUp(kActual, SVDQ_LOWRANK_MMAD_K_ALIGNMENT)" in lowrank_header
    assert "mActual <= SVDQ_LOWRANK_MMAD_M_TILE" in lowrank_header
    assert "nActual <= SVDQ_LOWRANK_MMAD_N_TILE" in lowrank_header
    assert "kActual <= SVDQ_LOWRANK_MMAD_K_TILE" in lowrank_header
    assert "mRound <= SVDQ_LOWRANK_MMAD_M_TILE" in lowrank_header
    assert "nRound <= SVDQ_LOWRANK_MMAD_N_TILE" in lowrank_header
    assert "kRound <= SVDQ_LOWRANK_MMAD_K_TILE" in lowrank_header
    assert "tile.inputStrideColumns >= tile.tile.kColumnOffset + kActual" in lowrank_header
    assert "tile.factorStrideColumns >= tile.tile.kColumnOffset + kActual" in lowrank_header
    assert "tile.outputStrideColumns >= tile.tile.outputColumnOffset + nActual" in lowrank_header
    assert "tile.accumulatorStrideColumns >= tile.tile.outputColumnOffset + nActual" in lowrank_header
    assert "inputElementCount == tile.mRound * tile.kRound" in lowrank_header
    assert "factorElementCount == tile.nRound * tile.kRound" in lowrank_header
    assert "outputElementCount == tile.mRound * tile.nRound" in lowrank_header
    assert "l1InputBytes == BytesForBF16Elements(inputElementCount)" in lowrank_header
    assert "l1FactorBytes == BytesForBF16Elements(factorElementCount)" in lowrank_header
    assert "l0ABytes == BytesForBF16Elements(inputElementCount)" in lowrank_header
    assert "l0BBytes == BytesForBF16Elements(factorElementCount)" in lowrank_header
    assert "l0CBytes == BytesForFP32Elements(outputElementCount)" in lowrank_header
    assert "storesAccumulator != storesOutput" in lowrank_header
    assert "return elementCount * SVDQ_LOWRANK_BF16_BYTES" in lowrank_header
    assert "return elementCount * sizeof(float)" in lowrank_header
    assert "const uint32_t inputElementCount = tilePlan.mRound * tilePlan.kRound" in lowrank_header
    assert "const uint32_t factorElementCount = tilePlan.nRound * tilePlan.kRound" in lowrank_header
    assert "const uint32_t outputElementCount = tilePlan.mRound * tilePlan.nRound" in lowrank_header
    assert "!tilePlan.tile.accumulatesLastKTile" in lowrank_header
    assert "tilePlan.tile.accumulatesLastKTile" in lowrank_header
    assert "input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.input))" in lowrank_header
    assert "factor.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.factor))" in lowrank_header
    assert "output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tilePlan.output))" in lowrank_header
    assert "accumulator.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tilePlan.accumulator))" in lowrank_header
    assert "output.SetValue(OutputElementOffset(tilePlan, rowOffset, outputOffset), value)" in lowrank_header
    assert "accumulator.SetValue(AccumulatorElementOffset(tilePlan, rowOffset, outputOffset), value)" in lowrank_header
    assert "tilePlan.accumulatesFirstKTile ? 0.0F" in lowrank_header
    assert "LoadAccumulatorFP32(tilePlan, rowOffset, outputOffset)" in lowrank_header
    assert "static_cast<float>(LoadInputBF16(tilePlan, rowOffset, kOffset))" in lowrank_header
    assert "static_cast<float>(LoadFactorBF16(tilePlan, outputOffset, kOffset))" in lowrank_header
    assert "for (uint32_t rowOffset = 0; rowOffset < tilePlan.tile.rowCount; ++rowOffset)" in lowrank_header
    assert "for (uint32_t outputOffset = 0; outputOffset < tilePlan.tile.outputColumnCount; ++outputOffset)" in lowrank_header
    assert "const float accumulator = AccumulateScalarBF16(tilePlan, rowOffset, outputOffset)" in lowrank_header
    assert "if (tilePlan.accumulatesLastKTile)" in lowrank_header
    assert "StoreOutputBF16(tilePlan, rowOffset, outputOffset, static_cast<bfloat16_t>(accumulator))" in lowrank_header
    assert "StoreAccumulatorFP32(tilePlan, rowOffset, outputOffset, accumulator)" in lowrank_header
    assert "const uint32_t coreIdx = AscendC::GetBlockIdx()" in lowrank_header
    assert "const uint32_t runtimeCoreCount = AscendC::GetBlockNum()" in lowrank_header
    assert "const SVDQLowRankCoreTileRange tileRange = CoreTileRange(coreIdx, scheduledCoreCount)" in lowrank_header
    assert "const SVDQLowRankTilePlan tilePlan = TilePlan(tileRange.tileStart + tileOffset)" in lowrank_header
    assert "const SVDQLowRankTileTensorPlan tileTensorPlan = BuildTileTensorPlan(tilePlan)" in lowrank_header
    assert "const SVDQLowRankMmadTilePlan mm" in lowrank_header
    assert "if (!mmadTilePlan.HasCompatibleShape())" in lowrank_header
    assert "const SVDQLowRankMmadBufferPlan bufferPlan = BuildMmadBufferPlan(mmadTilePlan)" in lowrank_header
    assert "if (!bufferPlan.HasCompleteFootprint())" in lowrank_header
    assert "if (!RunPlannedTileBF16(bufferPlan))" in lowrank_header
    assert "return RunScalarTileBF16(bufferPlan.tile.tile)" in lowrank_header
    assert "rowTiles * StageColumnTileCount(stage) * StageKTileCount(stage)" in lowrank_header
    assert "const uint32_t tilesPerRow = columnTiles * kTiles" in lowrank_header
    assert "Min(args_.tiling.outputColumnTile, stage.outputColumns - outputColumnOffset)" in lowrank_header
    assert "Min(args_.tiling.kTile, stage.inputColumns - kColumnOffset)" in lowrank_header
    assert "input BF16 -> down factor GEMM -> rank tile -> up factor GEMM -> projection BF16 GM" in lowrank_header
    assert " / 2" not in lowrank_header
    assert "total_rank" not in lowrank_header

    assert "lowRankCoreCount" in tiling_header
    assert "SVDQ_LOWRANK_ROW_TILE = 16" in tiling
    assert "SVDQ_LOWRANK_OUTPUT_COLUMN_TILE = 64" in tiling
    assert "SVDQ_LOWRANK_K_TILE = 64" in tiling
    assert "DispatchFFNCombineW4A8SVDQGetPlatformInfoAndSetTiling" in tiling
    assert "info.lowRankCoreCount = blockDim" in tiling
    assert "context->SetBlockDim(blockDim)" in tiling
    assert "context->SetTilingKey(1000000)" in tiling
    assert "invocation.rowTile = SVDQ_LOWRANK_ROW_TILE" in tiling
    assert "invocation.outputColumnTile = SVDQ_LOWRANK_OUTPUT_COLUMN_TILE" in tiling
    assert "invocation.kTile = SVDQ_LOWRANK_K_TILE" in tiling
    assert "invocation.coreCount = coreCount" in tiling
    assert "invocation.accumulatorRegionId = accumulatorRegionId" in tiling
    assert "SVDQFusedDownUpTiling" in tiling_header
    assert "lowRankInvocations[DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_COUNT]" in tiling_header
    assert "SetLowRankInvocation" in tiling
    assert "BuildLowRankInvocationTable" in tiling
    assert "BuildLowRankInvocationTable(tilingData)" in tiling
    assert "LowRankInvocation(uint32_t invocationId)" in contract
    assert "BuildLowRankArgs(uint32_t invocationId)" in contract
    assert "WorkspaceAddress(invocation.accumulatorRegionId)" in contract
    assert "FactorAddress(invocation.secondUpFactorId)" in contract
    assert "tilingData_.info.expertPerRank" in contract
    assert "LowRankInvocationReady(uint32_t invocationId)" in contract
    assert "ExecuteLowRankInvocation(uint32_t invocationId)" in contract
    assert "RunBF16LowRankStages()" in contract
    assert "SVDQFusedDownUp lowRankOp" in contract
    assert "lowRankOp.Init(BuildLowRankArgs(invocationId))" in contract
    assert "lowRankOp.HasCompleteContract() && lowRankOp.IsImplemented()" in contract
    assert "lowRankOp.Process()" in contract
    assert "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)" in contract
    assert "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_DOWN)" in contract
    assert "if (!RunBF16LowRankStages())" in contract

    gate_up_call = (
        "SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_GATE_UP,\n"
        "        SVDQ_REGION_ROUTED_X, SVDQ_REGION_PROJECTION_1, SVDQ_FACTOR_GATE_UP_L1, SVDQ_FACTOR_GATE_L2,\n"
        "        SVDQ_FACTOR_UP_L2, routedRows, info.hiddenSize, info.gateRank, info.upRank,\n"
        "        info.intermediateSize * 2, info.gateRankOffset, 0, info.upRankOffset, info.intermediateSize,\n"
        "        info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_1)"
    )
    down_call = (
        "SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_DOWN,\n"
        "        SVDQ_REGION_HIDDEN, SVDQ_REGION_PROJECTION_2, SVDQ_FACTOR_DOWN_L1, SVDQ_FACTOR_DOWN_L2,\n"
        "        SVDQ_INVALID_ID, routedRows, info.intermediateSize, info.downRank, 0, info.hiddenSize, 0, 0, 0, 0,\n"
        "        info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_2)"
    )
    assert gate_up_call in tiling
    assert down_call in tiling
    assert "gateUpSvdqL2" not in lowrank_header
    assert "gate_up_svdq_l2" not in lowrank_header
    assert "gateUpSvdqL2" not in lowrank_tiling
    assert "gate_up_svdq_l2" not in lowrank_tiling


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
        ("lowRankAccumulator1", "SVDQ_REGION_LOWRANK_ACCUMULATOR_1"),
        ("lowRankAccumulator2", "SVDQ_REGION_LOWRANK_ACCUMULATOR_2"),
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
