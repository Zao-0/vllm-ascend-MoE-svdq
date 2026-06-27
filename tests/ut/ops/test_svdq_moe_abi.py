#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import inspect
import json
from dataclasses import replace
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
        REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/dispatch_ffn_combine_w4_a8_svdq_torch_adpt.h"
    ).read_text()

    assert "dispatch_ffn_combine_w4a8_svdq(Tensor x" in binding
    assert 'ops.impl("dispatch_ffn_combine_w4a8_svdq", torch::kPrivateUse1' in binding
    assert "dispatch_ffn_combine_w4a8_svdq_meta" in meta
    assert 'ops.impl("dispatch_ffn_combine_w4a8_svdq"' in meta
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
        "x.scalar_type() == at::kBFloat16",
        "expert_idx.scalar_type() == at::kInt",
        "probs.scalar_type() == at::kFloat",
        "out.scalar_type() == at::kBFloat16",
        "expert_token_nums.scalar_type() == at::kInt",
        "!weight1.empty()",
        "!weight2.empty()",
        "!scale1.empty()",
        "!scale2.empty()",
        "!bias1.empty()",
        "!bias2.empty()",
        "weight1[0].scalar_type() == at::kInt",
        "weight2[0].scalar_type() == at::kInt",
        "scale1[0].scalar_type() == at::kLong",
        "scale2[0].scalar_type() == at::kLong",
        "bias1[0].scalar_type() == at::kFloat",
        "bias2[0].scalar_type() == at::kFloat",
        "hidden_size == x.size(1)",
        "gate_svdq_l2.size(0) == num_experts && gate_svdq_l2.size(2) == gate_rank",
        "up_svdq_l2.size(0) == num_experts && up_svdq_l2.size(1) == intermediate_size",
        "down_svdq_l1.size(0) == num_experts && down_svdq_l1.size(1) == down_rank",
        "down_svdq_l2.size(0) == num_experts && down_svdq_l2.size(1) == hidden_size",
        "weight1[0].size(0) == num_experts && weight2[0].size(0) == num_experts",
        "expert_token_nums.size(0) == num_experts",
        "x_active_mask.value().scalar_type() == at::kBool",
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


def test_svdq_mixed_epilogue_debug_torch_schema_meta_and_adapter_are_registered():
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
    adapter = (
        REPO_ROOT / "csrc/mc2/svdq_mixed_epilogue_debug_readback/"
        "svdq_mixed_epilogue_debug_readback_torch_adpt.h"
    ).read_text()

    assert "svdq_mixed_epilogue_debug_readback(Tensor residual_gate_up" in binding
    assert 'ops.impl("svdq_mixed_epilogue_debug_readback", torch::kPrivateUse1' in binding
    assert "svdq_mixed_epilogue_debug_readback_meta" in meta
    assert 'ops.impl("svdq_mixed_epilogue_debug_readback"' in meta
    assert "svdq_mixed_epilogue_debug_readback(" in adapter
    assert "EXEC_NPU_CMD(" in adapter
    assert "aclnnSVDQMixedEpilogueDebugReadback" in adapter
    assert "aclnn_svdq_mixed_epilogue_debug_readback.h" in adapter

    for name in (
        "residual_gate_up",
        "gate_up_low_rank",
        "residual_down",
        "down_low_rank",
        "swiglu_limit",
        "gate_up_total",
        "hidden_bf16",
        "hidden_int8",
        "hidden_int4_packed",
        "hidden_scale",
        "down_total",
        "out_bf16",
    ):
        assert name in binding
        assert name in meta
        assert name in adapter


def test_svdq_w4a8_gmm2_debug_torch_schema_meta_and_adapter_are_registered():
    binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
    adapter = (
        REPO_ROOT / "csrc/mc2/svdq_w4a8_gmm2_debug_readback/"
        "svdq_w4a8_gmm2_debug_readback_torch_adpt.h"
    ).read_text()
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8"
    cmake = (op_root / "op_host/CMakeLists.txt").read_text()
    op_def = (op_root / "op_host/svdqw4_a8_gmm2_debug_readback_def.cpp").read_text()
    kernel = (op_root / "op_kernel/svdqw4_a8_gmm2_debug_readback.cpp").read_text()
    official = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp").read_text()

    assert "svdq_w4a8_gmm2_debug_readback(Tensor x" in binding
    assert 'ops.impl("svdq_w4a8_gmm2_debug_readback", torch::kPrivateUse1' in binding
    assert "svdq_w4a8_gmm2_debug_readback_meta" in meta
    assert 'ops.impl("svdq_w4a8_gmm2_debug_readback"' in meta
    assert "svdq_w4a8_gmm2_debug_readback(" in adapter
    assert "EXEC_NPU_CMD(" in adapter
    assert "aclnnSVDQW4A8GMM2DebugReadback" in adapter
    assert "aclnn_svdq_w4a8_gmm2_debug_readback.h" in adapter

    for name in (
        "hidden_x_int4_packed",
        "hidden_x_scale",
        "external_expert_token_nums",
        "gmm2_post_dequant",
        "hidden_x_readback",
        "hidden_scale_readback",
        "gmm2_accumulator_int32",
    ):
        assert name in binding
        assert name in meta
        assert name in adapter

    assert "SVDQW4A8GMM2DebugReadback" in cmake
    assert "svdqw4_a8_gmm2_debug_readback" in cmake
    assert "OP_ADD(SVDQW4A8GMM2DebugReadback)" in op_def
    assert 'this->Output("hiddenXReadback")' in op_def
    assert 'this->Output("hiddenScaleReadback")' in op_def
    assert 'this->Output("gmm2AccumulatorInt32")' in op_def
    assert "InitGMM2OnlyFromPacked" in kernel
    assert "hiddenXReadback" in kernel
    assert "hiddenScaleReadback" in kernel
    assert "gmm2AccumulatorInt32" in kernel
    assert "GMM2OnlyFromPacked" in official
    assert "GMM2(params)" in official


def test_svdq_cann_op_host_surface_uses_canonical_five_factor_abi():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    cmake = (op_root / "op_host/CMakeLists.txt").read_text()
    header = (op_root / "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.h").read_text()
    wrapper = (op_root / "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.cpp").read_text()
    op_def = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_def.cpp").read_text()
    debug_op_def = (op_root / "op_host/svdq_low_rank_debug_readback_def.cpp").read_text()
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()

    assert "DispatchFFNCombineW4A8SVDQ" in cmake
    assert "dispatch_ffn_combine_w4_a8_svdq" in cmake
    assert "option(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in cmake
    assert "ON)" in cmake
    assert "set(_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS)" in cmake
    assert (
        "list(APPEND _DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS "
        "-DSVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)"
    ) in cmake
    dispatch_options = cmake[
        cmake.index("add_ops_compile_options(\n    OP_NAME DispatchFFNCombineW4A8SVDQ"):
        cmake.index("add_ops_compile_options(\n    OP_NAME SVDQLowRankDebugReadback")
    ]
    lowrank_debug_options = cmake[
        cmake.index("add_ops_compile_options(\n    OP_NAME SVDQLowRankDebugReadback"):
        cmake.index("add_ops_compile_options(\n    OP_NAME SVDQMixedEpilogueDebugReadback")
    ]
    assert "${_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS}" not in dispatch_options
    assert "${_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS}" in lowrank_debug_options
    assert (
        "OPTYPE dispatch_ffn_combine_w4_a8_svdq svdq_low_rank_debug_readback "
        "svdq_mixed_epilogue_debug_readback"
    ) in cmake
    assert "ACLNNTYPE aclnn_inner aclnn_inner aclnn_inner" in cmake
    assert "target_sources(op_host_aclnnInner PRIVATE" in cmake
    assert "svdq_low_rank_debug_readback_def.cpp" in cmake
    assert "svdq_mixed_epilogue_debug_readback_def.cpp" in cmake
    assert "SVDQMixedEpilogueDebugReadback" in cmake
    assert "aclnnDispatchFFNCombineW4A8SVDQGetWorkspaceSize" in header
    assert "aclnnInnerDispatchFFNCombineW4A8SVDQGetWorkspaceSize" in wrapper
    assert "OP_ADD(DispatchFFNCombineW4A8SVDQ)" in op_def
    assert "class SVDQLowRankDebugReadback" in debug_op_def
    assert "OP_ADD(SVDQLowRankDebugReadback)" in debug_op_def
    assert 'this->Output("gateUpOutput")' in debug_op_def
    assert 'this->Output("downOutput")' in debug_op_def
    assert 'this->Output("gateUpAccumulator")' in debug_op_def
    assert 'this->Output("downAccumulator")' in debug_op_def
    assert "IMPL_OP_OPTILING(DispatchFFNCombineW4A8SVDQ)" in tiling
    assert "DispatchFFNCombineW4A8SVDQ AscendC kernel is not implemented yet" not in tiling
    assert "return ge::GRAPH_SUCCESS;" in tiling

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

    for source in (header, wrapper, op_def, debug_op_def, tiling):
        assert "gateUpSvdqL2" not in source
        assert "gate_up_svdq_l2" not in source

    assert "constexpr uint32_t GATE_UP_SVDQ_L1_INDEX = 9" in tiling
    assert "constexpr uint32_t GATE_SVDQ_L2_INDEX = 10" in tiling
    assert "constexpr uint32_t UP_SVDQ_L2_INDEX = 11" in tiling
    assert "constexpr uint32_t DOWN_SVDQ_L1_INDEX = 12" in tiling
    assert "constexpr uint32_t DOWN_SVDQ_L2_INDEX = 13" in tiling


def test_svdq_host_tiling_builds_official_dispatch_routing_subtiling():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()

    assert (
        "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/"
        "moe_init_routing_quant_v2_tiling.h"
    ) in tiling_header
    assert "struct SVDQDispatchRoutingTiling" in tiling_header
    assert "uint64_t bf16RoutingTilingKey" in tiling_header
    assert "uint64_t bf16RoutingWorkspaceBytes" in tiling_header
    assert "uint64_t initRoutingQuantTilingKey" in tiling_header
    assert "uint64_t routingWorkspaceBytes" in tiling_header
    assert "uint32_t aivNum" in tiling_header
    assert "optiling::InnerMoeInitRoutingV2TilingData moeInitRoutingV2TilingData" in tiling_header
    assert "optiling::MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData" in tiling_header
    assert "SVDQDispatchRoutingTiling dispatchRouting" in tiling_header

    for token in (
        "production tiling is fail-closed",
        "constexpr uint32_t SVDQ_ROUTING_BLOCK_NUM = 20",
        "constexpr uint64_t SVDQ_ROUTING_UB_SIZE = 196352",
        "SVDQBF16RouteOnlyTilingBase routingBase",
        "dispatchRouting.bf16RoutingTilingKey = routingBase.tilingKey_",
        "dispatchRouting.bf16RoutingWorkspaceBytes = routingBase.workspaceSize_",
        "dispatchRouting.moeInitRoutingV2TilingData = routingBase.Data()",
        "MoeInitRoutingQuantV2TilingBase routingBase",
        "routingBase.DoTiling",
        "expertTokensCountOrCumsumFlag = 2",
        "quantMode = 1",
        "aivNumInitRouting = 2 * SVDQ_ROUTING_BLOCK_NUM",
        "info.expertPerRank) * static_cast<int64_t>(info.worldSize) + 1",
        "info.m) * static_cast<int64_t>(info.topK)",
        "dispatchRouting.initRoutingQuantTilingKey = routingBase.tilingKey_",
        "dispatchRouting.routingWorkspaceBytes = routingBase.workspaceSize_",
        "dispatchRouting.aivNum = aivNumInitRouting",
        "dispatchRouting.moeInitRoutingQuantV2TilingData = routingBase.quantTilingData",
        "vmsMiddleComputeParamsOp",
        "sortOutComputeParamsOp",
        "srcToDstComputeParamsOp",
        "srcToDstCapacityComputeParamsOp",
        "BuildDispatchRoutingTiling(tilingData)",
        "BuildBF16DispatchRoutingTiling(tilingData)",
        "workSpaces[0] = SVDQ_SYSTEM_WORKSPACE + info.workspaceBytes +",
        "tilingData->dispatchRouting.bf16RoutingWorkspaceBytes",
        "tilingData->dispatchRouting.routingWorkspaceBytes",
    ):
        assert token in tiling


def _build_aclnn_branch(build_script: str, start: str, end: str) -> str:
    branch = build_script[build_script.index(start) :]
    return branch[: branch.index(end)]


def _assert_svdq_ops_selected(build_branch: str):
    assert '"dispatch_ffn_combine_w4_a8"' in build_branch
    assert '"dispatch_ffn_combine_w4_a8_svdq"' in build_branch
    assert '"svdq_low_rank_debug_readback"' in build_branch
    assert '"svdq_mixed_epilogue_debug_readback"' in build_branch
    assert '"dispatch_ffn_combine_bf16"' in build_branch
    assert build_branch.index('"dispatch_ffn_combine_w4_a8"') < build_branch.index(
        '"dispatch_ffn_combine_w4_a8_svdq"'
    )
    assert build_branch.index('"dispatch_ffn_combine_w4_a8_svdq"') < build_branch.index(
        '"svdq_low_rank_debug_readback"'
    )
    assert build_branch.index('"svdq_low_rank_debug_readback"') < build_branch.index(
        '"svdq_mixed_epilogue_debug_readback"'
    )
    assert build_branch.index('"svdq_mixed_epilogue_debug_readback"') < build_branch.index(
        '"dispatch_ffn_combine_bf16"'
    )


def test_svdq_cann_ops_are_selected_by_a2_and_a3_aclnn_build_script():
    build_script = (REPO_ROOT / "csrc/build_aclnn.sh").read_text()
    a2_ops = _build_aclnn_branch(
        build_script,
        'elif [[ "$SOC_VERSION" =~ ^ascend910b ]];',
        'elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]];',
    )
    a3_ops = _build_aclnn_branch(
        build_script,
        'elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]];',
        'elif [[ "$SOC_VERSION" =~ ^ascend950 ]];',
    )

    _assert_svdq_ops_selected(a2_ops)
    _assert_svdq_ops_selected(a3_ops)


def test_official_w4a8_opdef_advertises_ascend910b_for_package_generation():
    op_def = (
        REPO_ROOT
        / "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_def.cpp"
    ).read_text()

    assert "OP_ADD(DispatchFFNCombineW4A8)" in op_def
    assert 'this->AICore().AddConfig("ascend910_93", aicore_config);' in op_def
    assert 'this->AICore().AddConfig("ascend910b", aicore_config);' in op_def
    assert op_def.index('this->AICore().AddConfig("ascend910_93", aicore_config);') < op_def.index(
        'this->AICore().AddConfig("ascend910b", aicore_config);'
    )


def test_official_w4a8_debug_readback_compile_flag_is_default_off():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8"
    cmake = (op_root / "op_host/CMakeLists.txt").read_text()
    kernel = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp").read_text()
    op_class = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8.h").read_text()
    debug_kernel = (op_root / "op_kernel/svdqw4_a8_debug_readback.cpp").read_text()
    debug_def = (op_root / "op_host/svdqw4_a8_debug_readback_def.cpp").read_text()
    debug_api = (op_root / "op_host/op_api/aclnn_svdq_w4a8_debug_readback.cpp").read_text()
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_tiling.cpp").read_text()
    gmm1_epilogue = (op_root / "op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp").read_text()
    gmm2_epilogue = (op_root / "op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp").read_text()

    assert "option(SVDQ_W4A8_DEBUG_READBACK" in cmake
    assert "Compile official W4A8 epilogues with FP32 GMM readback" in cmake
    assert "OFF)" in cmake
    assert "if(SVDQ_W4A8_DEBUG_READBACK)" in cmake
    assert "list(APPEND _DISPATCH_FFN_W4A8_DEBUG_OPTS -DW4A8_DEBUG)" in cmake
    assert "${_DISPATCH_FFN_W4A8_DEBUG_OPTS}" in cmake
    assert "OP_NAME SVDQW4A8DebugReadback" in cmake
    assert "-DW4A8_DEBUG" in cmake
    assert "svdqw4_a8_debug_readback" in cmake
    assert "get_filename_component(_DISPATCH_FFN_W4A8_ROOT" in cmake
    assert "set(svdqw4_a8_debug_readback_dir ${_DISPATCH_FFN_W4A8_ROOT}" in cmake

    assert "ptrCGMM1" in kernel
    assert "ptrCGMM1Hidden" in kernel
    assert "ptrCGMM2" in kernel
    assert "GM_ADDR ptrDebugRoutedX;" in kernel
    assert "GM_ADDR ptrDebugRoutedScale;" in kernel
    assert "GM_ADDR ptrDebugGMM1;" in kernel
    assert "GM_ADDR ptrDebugGMM1Hidden;" in kernel
    assert "GM_ADDR ptrDebugHiddenX;" in kernel
    assert "GM_ADDR ptrDebugHiddenScale;" in kernel
    assert "GM_ADDR ptrDebugGMM2;" in kernel
    assert "GM_ADDR symmetricPtr_ = nullptr, GM_ADDR ptrDebugRoutedX_ = nullptr" in kernel
    assert "GM_ADDR ptrDebugRoutedScale_ = nullptr" in kernel
    assert "GM_ADDR ptrDebugGMM1Hidden_ = nullptr" in kernel
    assert "GM_ADDR ptrDebugHiddenX_ = nullptr" in kernel
    assert "GM_ADDR ptrDebugHiddenScale_ = nullptr" in kernel
    assert "GM_ADDR ptrDebugGMM2_ = nullptr" in kernel
    assert "ptrDebugRoutedX(ptrDebugRoutedX_)" in kernel
    assert "ptrDebugRoutedScale(ptrDebugRoutedScale_)" in kernel
    assert "ptrDebugGMM1(ptrDebugGMM1_)" in kernel
    assert "ptrDebugGMM1Hidden(ptrDebugGMM1Hidden_)" in kernel
    assert "ptrDebugHiddenX(ptrDebugHiddenX_)" in kernel
    assert "ptrDebugHiddenScale(ptrDebugHiddenScale_)" in kernel
    assert "ptrDebugGMM2(ptrDebugGMM2_)" in kernel
    assert "#ifdef W4A8_DEBUG" in kernel
    assert "if (coreIdx == 0 && params.ptrDebugRoutedX != nullptr)" in kernel
    assert "if (coreIdx == 0 && params.ptrDebugRoutedScale != nullptr)" in kernel
    assert "if (params.ptrDebugGMM1 != nullptr)" in kernel
    assert "ptrCGMM1 = params.ptrDebugGMM1;" in kernel
    assert "workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(float);" in kernel
    assert "if (params.ptrDebugGMM1Hidden != nullptr)" in kernel
    assert "ptrCGMM1Hidden = params.ptrDebugGMM1Hidden;" in kernel
    assert "workspaceOffset += params.maxOutputSize * k2 * sizeof(float);" in kernel
    assert "if (params.ptrDebugHiddenX != nullptr)" in kernel
    assert "CopyGMToGM(hiddenXDebugGM, gmA2I4_I8" in kernel
    assert "if (params.ptrDebugHiddenScale != nullptr)" in kernel
    assert "CopyGMToGM(hiddenScaleDebugGM, gmPerTokenScale2" in kernel
    assert "if (params.ptrDebugGMM2 != nullptr)" in kernel
    assert "ptrCGMM2 = params.ptrDebugGMM2;" in kernel
    assert "workspaceOffset += params.maxOutputSize * n2 * sizeof(float);" in kernel
    assert "using CopyUbToGmGMM1 = typename TileCopyDebug::CopyUbToGmD;" in gmm1_epilogue
    assert "using CopyUbToGmGMM1Hidden = typename TileCopyDebug::CopyUbToGmD;" in gmm1_epilogue
    assert "layout::RowMajor layoutGMM1{1, blockN};" in gmm1_epilogue
    assert "copyUbToGmGMM1(gmTileGMM1, ubCFp32, layoutGMM1, layoutGMM1);" in gmm1_epilogue
    assert "layout::RowMajor layoutGMM1Hidden{1, ChunkTileLen};" in gmm1_epilogue
    assert "copyUbToGmGMM1Hidden(gmTileGMM1Hidden, ubCFp32ChunkN, layoutGMM1Hidden, layoutGMM1Hidden);" in gmm1_epilogue
    hidden_copy = gmm1_epilogue.index(
        "copyUbToGmGMM1Hidden(gmTileGMM1Hidden, ubCFp32ChunkN, layoutGMM1Hidden, layoutGMM1Hidden);"
    )
    quantization = gmm1_epilogue.index("// Quantization", hidden_copy)
    hidden_copy_block = gmm1_epilogue[hidden_copy:quantization]
    assert "AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);" in hidden_copy_block
    assert hidden_copy_block.count("AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);") >= 2
    assert "copyUbToGmGMM2(gmTileGMM2, ubFp32, layoutGM, layoutUB);" in gmm2_epilogue

    assert "GM_ADDR debugRoutedXGM = nullptr" in op_class
    assert "GM_ADDR debugRoutedScaleGM = nullptr" in op_class
    assert "GM_ADDR debugGMM1GM = nullptr" in op_class
    assert "GM_ADDR debugGMM1HiddenGM = nullptr" in op_class
    assert "GM_ADDR debugHiddenXGM = nullptr" in op_class
    assert "GM_ADDR debugHiddenScaleGM = nullptr" in op_class
    assert "GM_ADDR debugGMM2GM = nullptr" in op_class
    assert "debugRoutedXGM_" in op_class
    assert "debugRoutedScaleGM_" in op_class
    assert "debugGMM1GM_" in op_class
    assert "debugGMM1HiddenGM_" in op_class
    assert "debugHiddenXGM_" in op_class
    assert "debugHiddenScaleGM_" in op_class
    assert "debugGMM2GM_" in op_class
    assert "debugRoutedXGM_, debugRoutedScaleGM_, debugGMM1GM_, debugGMM1HiddenGM_, debugHiddenXGM_," in op_class
    assert "debugHiddenScaleGM_, debugGMM2GM_" in op_class

    assert "extern \"C\" __global__ __aicore__ void svdqw4_a8_debug_readback" in debug_kernel
    assert "KERNEL_TYPE_MIX_AIC_1_2" in debug_kernel
    assert "DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true> op" in debug_kernel
    assert "routedXInt8" in debug_kernel
    assert "routedXScale" in debug_kernel
    assert "gmm1PostDequant" in debug_kernel
    assert "gmm1HiddenPrequant" in debug_kernel
    assert "hiddenXInt4Packed" in debug_kernel
    assert "hiddenXScale" in debug_kernel
    assert "gmm2PostDequant" in debug_kernel

    assert "class SVDQW4A8DebugReadback" in debug_def
    assert 'this->Output("routedXInt8")' in debug_def
    assert 'this->Output("routedXScale")' in debug_def
    assert 'this->Output("gmm1PostDequant")' in debug_def
    assert 'this->Output("gmm1HiddenPrequant")' in debug_def
    assert 'this->Output("hiddenXInt4Packed")' in debug_def
    assert 'this->Output("hiddenXScale")' in debug_def
    assert 'this->Output("gmm2PostDequant")' in debug_def
    assert "OP_ADD(SVDQW4A8DebugReadback)" in debug_def
    assert "IMPL_OP_OPTILING(SVDQW4A8DebugReadback)" in tiling

    assert "aclnnSVDQW4A8DebugReadbackGetWorkspaceSize" in debug_api
    assert "aclnnInnerSVDQW4A8DebugReadbackGetWorkspaceSize" in debug_api
    assert "routedXInt8" in debug_api
    assert "routedXScale" in debug_api
    assert "gmm1PostDequant" in debug_api
    assert "gmm1HiddenPrequant" in debug_api
    assert "gmm2PostDequant" in debug_api


def test_svdq_w4a8_debug_calibration_probe_builds_nonzero_packed_int4_data():
    from tools.svdq_w4a8_debug_calibration_probe import (
        CalibrationShape,
        build_calibration_tensors,
        validate_calibration_tensors,
    )

    shape = CalibrationShape(
        num_experts=2,
        hidden_size=16,
        intermediate_size=16,
        group_size=8,
        num_tokens=4,
        top_k=1,
    )
    tensors = build_calibration_tensors(shape)
    validation = validate_calibration_tensors(tensors, shape)

    assert validation["passed"]
    assert not validation["public_grouped_matmul_used"]
    assert not validation["official_kernel_launched"]
    assert not validation["numerical_acceptance_claimed"]
    assert tensors["w13_weight"].shape == (2, 16, 4)
    assert tensors["w2_weight"].shape == (2, 16, 2)
    assert tensors["w13_weight"].dtype == torch.int32
    assert tensors["w2_weight"].dtype == torch.int32
    assert tensors["w13_weight_scale"].shape == (2, 2, 32)
    assert tensors["w2_weight_scale"].shape == (2, 2, 16)
    assert tensors["w13_weight_scale"].dtype == torch.int64
    assert tensors["w2_weight_scale"].dtype == torch.int64
    assert validation["packed_int4_checks"]["w13_weight"]["nonzero_nibbles"]
    assert validation["packed_int4_checks"]["w2_weight"]["nonzero_nibbles"]
    assert validation["expert_token_total"] == 4


def test_svdq_w4a8_debug_calibration_probe_does_not_use_public_grouped_matmul():
    probe = (REPO_ROOT / "tools/svdq_w4a8_debug_calibration_probe.py").read_text()

    assert "npu_grouped_matmul" not in probe
    assert "torch.ops._C_ascend.dispatch_ffn_combine" not in probe
    assert '"public_grouped_matmul_used": False' in probe
    assert '"numerical_acceptance_claimed": False' in probe


def test_svdq_lowrank_debug_probe_preflights_runtime_soc_package_support(tmp_path):
    from tools.svdq_lowrank_debug_readback_probe import (
        DEBUG_OP_NAME,
        PRODUCTION_OP_NAME,
        _custom_package_debug_op_support,
        _normalize_soc_name,
        _package_supports_runtime_soc,
    )

    config_root = tmp_path / "config"
    a3_config = config_root / "ascend910_93" / "binary_info_config.json"
    a3_config.parent.mkdir(parents=True)
    a3_config.write_text(json.dumps({DEBUG_OP_NAME: {}, PRODUCTION_OP_NAME: {}, "OtherOp": {}}), encoding="utf-8")
    a2_config = config_root / "ascend910b" / "binary_info_config.json"
    a2_config.parent.mkdir(parents=True)
    a2_config.write_text(json.dumps({DEBUG_OP_NAME: {}, PRODUCTION_OP_NAME: {}, "OtherOp": {}}), encoding="utf-8")

    support = _custom_package_debug_op_support(config_root)

    assert _normalize_soc_name("Ascend910B4") == "ascend910b"
    assert _normalize_soc_name("Ascend910_9391") == "ascend910_93"
    assert support["debug_op_supported_socs"] == ["ascend910_93", "ascend910b"]
    assert support["production_op_supported_socs"] == ["ascend910_93", "ascend910b"]
    assert support["by_soc"]["ascend910_93"]["has_debug_op"]
    assert support["by_soc"]["ascend910_93"]["has_production_op"]
    assert support["by_soc"]["ascend910b"]["has_debug_op"]
    assert support["by_soc"]["ascend910b"]["has_production_op"]
    assert _package_supports_runtime_soc(package_support=support, runtime_soc="ascend910_93")
    assert _package_supports_runtime_soc(package_support=support, runtime_soc="ascend910b")
    assert _package_supports_runtime_soc(
        package_support=support,
        runtime_soc="ascend910b",
        supported_socs_key="production_op_supported_socs",
    )


def test_svdq_lowrank_debug_probe_records_appendix4_bf16_boundary():
    probe = (REPO_ROOT / "tools/svdq_lowrank_debug_readback_probe.py").read_text()

    for token in (
        "gate_up_output_bf16",
        "down_output_bf16",
        "_bf16_output_boundary_check",
        "producer_bf16_to_accumulator_cast_error",
        "expected = accumulator.to(torch.bfloat16)",
        "actual_bf16_nonzero_active_rows",
        "actual_bf16_max_abs",
        "expected_bf16_from_accumulator_max_abs",
        "fp32_accumulator_max_abs",
        "producer_gm_base",
        "workspace_region_offset_bytes",
        "routed_row_offset",
        "row_stride_elements",
        "row_stride_bytes",
        "store_byte_count",
        "physical_byte_offset",
        "fp32_accumulator_value",
        "expected_bf16_value",
        "stored_bf16_value",
        "aiv_loaded_value",
        "copy-only AIV consumer readback operator is not implemented yet",
        "rank_workspace_non_overlap_verified",
        "bf16_output_boundary_passed",
    ):
        assert token in probe
    assert "finite_fp32_accumulator_readback_cast_to_bf16" not in probe


def test_svdq_lowrank_debug_tiling_launches_aic_only():
    tiling = (
        REPO_ROOT
        / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp"
    ).read_text()
    debug_start = tiling.index("static ge::graphStatus SVDQLowRankDebugReadbackCheckShapeAndSetTiling")
    debug_end = tiling.index("static ge::graphStatus SVDQLowRankDebugReadbackTilingFunc", debug_start)
    debug_tiling = tiling[debug_start:debug_end]

    assert "const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();" in debug_tiling
    assert "const uint32_t blockDim = aicNum;" in debug_tiling
    assert "context->SetBlockDim(blockDim);" in debug_tiling
    assert "context->SetTilingKey(1000000);" in debug_tiling
    assert "GetCoreNumAiv()" not in debug_tiling
    assert "CalcTschBlockDim" not in debug_tiling


def test_svdq_lowrank_debug_readback_excludes_scalar_bf16_fallback():
    fused = (
        REPO_ROOT
        / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_fused_down_up.hpp"
    ).read_text()
    scalar_start = fused.index("__aicore__ inline bfloat16_t LoadInputBF16")
    scalar_end = fused.index("__aicore__ inline bool RunMmadTileBF16", scalar_start)
    scalar_block = fused[scalar_start:scalar_end]
    run_planned_start = fused.index("__aicore__ inline bool RunPlannedTileBF16")
    run_planned_end = fused.index("__aicore__ inline bool IsImplemented", run_planned_start)
    run_planned_block = fused[run_planned_start:run_planned_end]

    assert "#ifndef SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in fused[:scalar_start]
    assert "static_cast<bfloat16_t>(accumulator)" in scalar_block
    assert "#elif defined(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)" in run_planned_block
    assert "return false;" in run_planned_block


def test_svdq_lowrank_debug_mmad_synchronizes_l0_reuse():
    fused = (
        REPO_ROOT
        / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_fused_down_up.hpp"
    ).read_text()
    mmad_start = fused.index("__aicore__ inline bool RunMmadTileBF16")
    mmad_end = fused.index("__aicore__ inline bool RunPlannedTileBF16", mmad_start)
    mmad_block = fused[mmad_start:mmad_end]

    assert "SVDQ_LOWRANK_MMAD_L0_STAGES = 2" in fused
    assert "SVDQ_LOWRANK_MMAD_L0A_EVENT_BASE" in fused
    assert "SVDQ_LOWRANK_MMAD_L0B_EVENT_BASE" in fused
    assert "SVDQ_LOWRANK_MMAD_M_EVENT" in fused
    assert "% SVDQ_LOWRANK_MMAD_L0_STAGES" in mmad_block
    assert "const int32_t l0AEvent = SVDQ_LOWRANK_MMAD_L0A_EVENT_BASE" in mmad_block
    assert "const int32_t l0BEvent = SVDQ_LOWRANK_MMAD_L0B_EVENT_BASE" in mmad_block
    assert "pipelinePlan.buffer.l0ABytes * l0Stage" in mmad_block
    assert "pipelinePlan.buffer.l0BBytes * l0Stage" in mmad_block
    assert "AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(" in mmad_block
    assert "AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0AEvent);" in mmad_block
    assert "AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0BEvent);" in mmad_block
    assert "AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(SVDQ_LOWRANK_MMAD_M_EVENT);" in mmad_block
    assert "AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(SVDQ_LOWRANK_MMAD_M_EVENT);" in mmad_block
    assert "AscendC::PipeBarrier<PIPE_M>();" in mmad_block
    process_start = fused.index("__aicore__ inline void Process()")
    process_end = fused.index("/*", process_start)
    process_block = fused[process_start:process_end]
    assert "SVDQOfficialBF16Resource resource;" in process_block
    assert "ExecuteStage(stageIndex, coreIdx, scheduledCoreCount, resource)" in process_block
    assert "InitializeMmadL0ReuseEvents();" not in process_block
    assert "DrainMmadL0ReuseEvents();" not in process_block
    assert "SVDQOfficialBF16BlockMmad blockMmad(resource);" in fused
    assert "SVDQ_LOWRANK_BF16_RANK_N_TILE = 64" in fused
    assert "SVDQLowRankBF16RankBlockMmad blockMmad(resource);" not in fused
    assert "stage.stageKind == SVDQ_LOWRANK_STAGE_DOWN_PROJECT" in fused
    assert "StageOutputColumnTile(stage)" in fused
    assert "RunOfficialOutputTileBF16(outputTilePlan, blockMmad)" in fused
    assert mmad_block.index("WaitFlag<AscendC::HardEvent::M_MTE1>") < mmad_block.index(
        "l1_to_l0_a<ArchType::ASCEND_V220"
    )
    assert mmad_block.index("WaitFlag<AscendC::HardEvent::MTE1_M>") < mmad_block.index(
        "mmad<ArchType::ASCEND_V220"
    )


def test_svdq_lowrank_debug_install_validator_checks_static_package_surfaces():
    validator = (REPO_ROOT / "tools/svdq_lowrank_debug_install_validate.py").read_text()

    assert "CUSTOM_LOWRANK_DEBUG_SOURCE_DIR" in validator
    assert "REQUIRED_PACKAGE_SOURCE_SNIPPETS" in validator
    assert "FORBIDDEN_PACKAGE_SOURCE_SNIPPETS" in validator
    assert "package_lowrank_debug_source" in validator
    assert "GM_ADDR rankWorkspace" in validator
    assert "runtime_.rankWorkspace = workspaceGM" in validator
    assert "args.rank = runtime_.rankWorkspace" in validator
    assert "downAccumulator, workspaceGM, tilingGM" in validator
    assert '"(void)workspaceGM"' in validator
    assert "REQUIRED_OPAPI_SYMBOLS" in validator
    assert "REQUIRED_PRODUCTION_OPAPI_SYMBOLS" in validator
    for symbol in (
        "aclnnSVDQLowRankDebugReadbackGetWorkspaceSize",
        "aclnnSVDQLowRankDebugReadback",
        "aclnnInnerSVDQLowRankDebugReadbackGetWorkspaceSize",
        "aclnnInnerSVDQLowRankDebugReadback",
        "aclnnDispatchFFNCombineW4A8SVDQGetWorkspaceSize",
        "aclnnDispatchFFNCombineW4A8SVDQ",
        "aclnnInnerDispatchFFNCombineW4A8SVDQGetWorkspaceSize",
        "aclnnInnerDispatchFFNCombineW4A8SVDQ",
    ):
        assert symbol in validator
    assert "svdq_low_rank_debug_readback" in validator
    assert "dispatch_ffn_combine_w4a8_svdq" in validator
    assert "--require-production-runtime-soc-support" in validator
    assert '"production_fail_closed"' in validator
    assert "_custom_package_debug_op_support()" in validator
    assert "_package_supports_runtime_soc(" in validator
    assert "require_runtime_soc_support" in validator
    assert "libcust_opapi.so" in validator


def test_svdq_lowrank_debug_install_validator_rejects_stale_package_source(tmp_path):
    from tools.svdq_lowrank_debug_install_validate import _package_lowrank_debug_source_status

    source_dir = tmp_path / "lowrank"
    source_dir.mkdir()
    (source_dir / "svdq_lowrank_debug_readback.h").write_text(
        """
        struct SVDQLowRankDebugRuntimeGM {};
        __aicore__ inline void Init(GM_ADDR tilingGM) {}
        """,
        encoding="utf-8",
    )
    (source_dir / "svdq_lowrank_debug_readback.cpp").write_text(
        """
        (void)workspaceGM;
        op.Init(routedX, hidden, gateUpSvdqL1, gateSvdqL2, upSvdqL2, downSvdqL1,
            downSvdqL2, expertTokenNums, gateUpOutput, downOutput, gateUpAccumulator,
            downAccumulator, tilingGM);
        """,
        encoding="utf-8",
    )

    stale_status = _package_lowrank_debug_source_status(source_dir)

    assert stale_status["checked"]
    assert not stale_status["passed"]
    assert not stale_status["required_snippets"]["svdq_lowrank_debug_readback.h"]["GM_ADDR rankWorkspace"]
    assert stale_status["forbidden_snippets"]["svdq_lowrank_debug_readback.cpp"]["(void)workspaceGM"]

    (source_dir / "svdq_lowrank_debug_readback.h").write_text(
        """
        GM_ADDR rankWorkspace;
        __aicore__ inline void Init(GM_ADDR workspaceGM, GM_ADDR tilingGM) {
            runtime_.rankWorkspace = workspaceGM;
        }
        args.rank = runtime_.rankWorkspace;
        """,
        encoding="utf-8",
    )
    (source_dir / "svdq_lowrank_debug_readback.cpp").write_text(
        """
        op.Init(routedX, hidden, gateUpSvdqL1, gateSvdqL2, upSvdqL2, downSvdqL1,
            downSvdqL2, expertTokenNums, gateUpOutput, downOutput, gateUpAccumulator,
            downAccumulator, workspaceGM, tilingGM);
        """,
        encoding="utf-8",
    )

    fixed_status = _package_lowrank_debug_source_status(source_dir)

    assert fixed_status["checked"]
    assert fixed_status["passed"]


def test_svdq_w4a8_activation_quant_probe_matches_residual_stage_contract():
    probe = (REPO_ROOT / "tools/svdq_w4a8_activation_quant_probe.py").read_text()

    assert "torch_npu.npu_dynamic_quant" in probe
    assert "routed_input" in probe
    assert "hidden_input" in probe
    assert "_cpu_dynamic_quant_reference" in probe
    assert ".amax(dim=1)" in probe
    assert "/ 127.0" in probe
    assert ".clamp(-127, 127)" in probe
    assert "--require-npu" in probe
    assert "phase_f_activation_quant_probe_summary.json" in probe
    assert '"q_exact_match"' in probe
    assert '"q_mismatch_count"' in probe
    assert '"scale_error"' in probe
    assert '"npu_dequant_vs_input_error"' in probe


def test_svdq_w4a8_residual_gmm_contract_probe_matches_official_aic_aiv_surface():
    probe = (REPO_ROOT / "tools/svdq_w4a8_residual_gmm_contract_probe.py").read_text()

    for token in (
        "official_mixed_aic_aiv_kernel_boundary",
        "official_aic_w4a8_gmm_contract",
        "official_aiv_gmm1_dequant_swiglu_hidden_quant_contract",
        "official_aiv_gmm2_dequant_final_output_contract",
        "KERNEL_TYPE_MIX_AIC_1_2",
        "CATLASS_DEVICE void operator()<AscendC::AIC>",
        "CATLASS_DEVICE void operator()<AscendC::AIV>",
        "GMM1(params);",
        "GMM2(params);",
        "DispatchAndCombine(params);",
        "BlockMmad",
        "EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant",
        "EpilogueAtlasA2W4A8PostPerTokenDequantV2",
        "production_host_tiling_expected",
        "fail_closed",
        "numerical_acceptance_claimed",
        "phase_an_official_w4a8_aic_aiv_contract_summary.json",
    ):
        assert token in probe


def test_svdq_w4a8_residual_gmm_device_probe_executes_official_stage_shapes():
    probe = (REPO_ROOT / "tools/svdq_w4a8_residual_gmm_device_probe.py").read_text()

    for token in (
        "torch_npu.npu_grouped_matmul",
        "_pack_modelslim_per_channel_scale",
        "w4a8_residual_gmm1",
        "w4a8_residual_gmm2",
        "(2 * intermediate_size) // 8",
        "hidden_size // 8",
        "group_list_type=1",
        "output_dtype=torch.bfloat16",
        "phase_f_residual_gmm_device_probe_summary.json",
        "--require-npu",
        "--real-checkpoint",
        "--calibrate-packed-int4",
        "_public_grouped_matmul_api_contract",
        "production_acceptance_surface",
        "official dispatch_ffn_combine_w4_a8 mixed AIC/AIV kernel",
        "_run_packed_int4_calibration",
        "packed_int4_all_ones_grouped_matmul_calibration",
        "gmm1_postload_scale_shape",
        "gmm2_postload_scale_shape",
        "public grouped-matmul packed-INT4 behavior does not match",
        "_load_real_residual_layer",
        "build_svdq_moe_layer_spec",
        "_make_official_w4a8_method",
        "_make_residual_validation_layer",
        "_load_residual_checkpoint_tensor",
        "_ensure_minimal_ascend_config_for_official_postload",
        "real_checkpoint_w4a8_residual_gmm1",
        "real_checkpoint_w4a8_residual_gmm2",
        "reference_formula",
        "signed_int4_weight",
        "scale_bias",
        "per_token_scale",
        "official_aic_aiv_validation",
        "appendix1_required_before_production_enablement",
    ):
        assert token in probe


def test_svdq_official_w4a8_dispatch_probe_uses_official_aclnn_path():
    probe = (REPO_ROOT / "tools/svdq_official_w4a8_dispatch_probe.py").read_text()

    for token in (
        "torch.ops._C_ascend.dispatch_ffn_combine",
        "aclnnDispatchFFNCombineW4A8",
        "OFFICIAL_W4A8_OPAPI_SYMBOLS",
        "official_w4a8_opapi_symbols",
        "aclnnInnerDispatchFFNCombineW4A8GetWorkspaceSize",
        "_load_real_residual_layer",
        "official_dispatch_ffn_combine_w4a8_zero_input_real_checkpoint",
        "official_aic_aiv_validation",
        "zero-input full official W4A8 op smoke",
        "nonzero staged GMM/dequant validation still required",
        "production_svdq_host_tiling_expected",
        "fail_closed",
        "weight1=[layer.w13_weight]",
        "weight2=[layer.w2_weight]",
        "scale1=[layer.w13_weight_scale]",
        "scale2=[layer.w2_weight_scale]",
        "bias1=[layer.w13_scale_bias]",
        "bias2=[layer.w2_scale_bias]",
        "--require-npu",
        "phase_as_official_w4a8_dispatch_probe_summary.json",
    ):
        assert token in probe


def test_svdq_w4a8_debug_readback_real_checkpoint_probe_uses_official_debug_path():
    probe = (REPO_ROOT / "tools/svdq_w4a8_debug_readback_real_checkpoint_probe.py").read_text()

    for token in (
        "torch.ops._C_ascend.svdq_w4a8_debug_readback",
        "aclnnSVDQW4A8DebugReadback",
        "_make_official_w4a8_method",
        "_make_residual_validation_layer",
        "_load_residual_checkpoint_tensor",
        "AscendW4A8DynamicFusedMoEMethod.process_weights_after_loading_modelslim",
        "real_checkpoint_w4a8_debug_readback_",
        "--input-mode",
        "linear",
        "input_health_gate_passed",
        "routed_quant_health",
        "routed_x_int8_active",
        "routed_x_scale_active",
        "gmm1_post_dequant_active",
        "gmm1_hidden_prequant_active",
        "gmm2_post_dequant_active",
        "--compare-gmm1-reference",
        "_official_gmm1_unfused_reference",
        "FetchAndPreprocessInt8ToInt4",
        "gmm1_real_checkpoint_numerical_gate",
        "gmm2_real_checkpoint_numerical_gate",
        "gmm2_unfused_reference",
        "--compare-gmm2-reference",
        "_official_gmm2_unfused_reference",
        "hidden_x_int4_packed_active",
        "hidden_x_scale_active",
        "hidden_quant_health",
        "public_grouped_matmul_used",
        "False",
        "nonzero_real_checkpoint_numerical_gate",
        "phase_bb_w4a8_debug_readback_real_checkpoint_summary.json",
        "--require-npu",
    ):
        assert token in probe
    assert "npu_grouped_matmul" not in probe


def test_svdq_w4a8_tap_mixed_epilogue_probe_uses_official_taps_and_records_limitations():
    probe = (REPO_ROOT / "tools/svdq_w4a8_tap_mixed_epilogue_probe.py").read_text()

    for token in (
        "torch.ops._C_ascend.svdq_w4a8_debug_readback",
        "_official_gmm1_unfused_reference",
        "_official_gmm2_unfused_reference",
        "_launch_lowrank_debug",
        "_run_npu_mixed_epilogue",
        "_run_official_gmm2_from_mixed_hidden",
        "torch.ops._C_ascend.svdq_w4a8_gmm2_debug_readback",
        "_has_registered_gmm2_debug_op",
        "build_svdq_mixed_epilogue_reference",
        "build_svdq_final_combine_reference",
        "_run_real_final_combine",
        "torch_npu.npu_moe_token_unpermute",
        "official_token_major_output_slots_to_permuted_input_rows",
        "pack_official_hidden_i4_reference",
        "stage2_3_real_checkpoint_same_route_w4a8_svdq_composition",
        "official_gmm2_from_svdq_hidden",
        "stage2_3_same_routing_manifest",
        "same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant",
        "same_source_token_payload_across_topk_slots_proven",
        "final_combine_consumes_mixed_down_peer_output",
        "final_combine_output_validated",
        "final_combine_output",
        "real_final_combine",
        "source_for_mixed_epilogue",
        "bf16_lowrank_output_readback",
        "bf16_output_used_for_mixed_epilogue",
        "True",
        "lowrank_output_health_passed",
        "_nonzero_finite(lowrank_stats[\"gate_up_output\"])",
        "_nonzero_finite(lowrank_stats[\"down_output\"])",
        "production_svdq_host_tiling_fail_closed",
        "public_grouped_matmul_used",
        "GMM2 residual tap comes from the official W4A8 debug path",
        "relaunch official GMM2 from the SVDQ-modified hidden activation",
        "hidden_q_mismatch_count_tolerance",
        "--hidden-q-mismatch-count-tol",
        "phase_w4a8_tap_mixed_epilogue_summary.json",
        "--require-npu",
    ):
        assert token in probe
    assert "finite_fp32_accumulator_readback_cast_to_bf16" not in probe
    assert "lowrank_accumulator_health_passed" not in probe
    assert "npu_grouped_matmul" not in probe


def test_svdq_final_combine_device_probe_executes_official_token_unpermute_surface():
    probe = (REPO_ROOT / "tools/svdq_final_combine_device_probe.py").read_text()

    for token in (
        "torch_npu.npu_moe_token_unpermute",
        "build_svdq_final_combine_reference",
        "permuted_tokens=routed_output.to(device=device)",
        "sorted_indices=expanded_row_idx.to(device=device)",
        "probs=topk_weights.to(device=device)",
        "expanded_row_idx = torch.arange(num_rows, dtype=torch.int32)",
        "num_rows = num_tokens * top_k",
        "expected = reference[\"stages\"][\"combined_output\"].to(torch.bfloat16)",
        "phase_h_final_combine_device_probe_summary.json",
        "--require-npu",
    ):
        assert token in probe


def test_svdq_mixed_epilogue_device_probe_matches_reference_oracle():
    probe = (REPO_ROOT / "tools/svdq_mixed_epilogue_device_probe.py").read_text()

    for token in (
        "build_svdq_mixed_epilogue_reference",
        "torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback",
        "bootstrap_custom_op_env(include_vendor_lib=True)",
        "CUSTOM_OPAPI_LIB",
        "_preload_custom_opapi()",
        "_PRELOADED_CUSTOM_OPAPI_GLOBAL",
        "_finish_probe",
        "SVDQ_MIXED_EPILOGUE_ALLOW_CANN_TEARDOWN",
        "os._exit(exit_code)",
        "enable_custom_op()",
        "gate_up_low_rank = torch.cat((inputs[\"gate_lowrank\"], inputs[\"up_lowrank\"]), dim=1).contiguous()",
        "gate_up_total,",
        "hidden_bf16,",
        "hidden_q,",
        "hidden_scale,",
        "down_total,",
        "out_bf16,",
        "_appendix3_case_inputs",
        "appendix3_gate_order",
        "residual_only",
        "svdq_only",
        "two_branch_nonzero",
        "appendix3_zero_branch_tests_are_isolation_only",
        "w4a8_fp32_gate_up_add_zero_exact",
        "svdq_bf16_gate_up_cast_to_fp32_exact",
        "\"stage\": \"mixed_epilogue_debug_readback\"",
        "\"debug_op\": \"torch.ops._C_ascend.svdq_mixed_epilogue_debug_readback\"",
        "hidden_q_exact_match",
        "phase_k_mixed_epilogue_device_probe_summary.json",
        "--require-npu",
    ):
        assert token in probe
    assert "npu_grouped_matmul" not in probe
    assert "torch_npu.npu_dynamic_quant" not in probe


def test_svdq_composed_pipeline_device_probe_validates_stage_order():
    probe = (REPO_ROOT / "tools/svdq_composed_pipeline_device_probe.py").read_text()

    for token in (
        "build_svdq_mixed_epilogue_reference",
        "build_svdq_final_combine_reference",
        "torch_npu.npu_dynamic_quant",
        "torch_npu.npu_moe_token_unpermute",
        "gate_rank_offset = 0",
        "up_rank_offset = gate_rank",
        "gate_up_svdq_l1 = torch.cat((gate_l1, up_l1), dim=0).contiguous()",
        "npu_gate_up = _npu_lowrank_gate_up(",
        "first_mixed_reference = build_svdq_mixed_epilogue_reference(",
        "hidden_q, hidden_scale = torch_npu.npu_dynamic_quant",
        "npu_down = _npu_down_lowrank(",
        "output_epilogue_reference = build_svdq_mixed_epilogue_reference(",
        "final_reference = build_svdq_final_combine_reference(",
        "final_output = torch_npu.npu_moe_token_unpermute(",
        "_same_routing_identity_manifest(",
        "routing_identity_manifest",
        "first_stage_w4a8_and_svdq_share_routed_x",
        "svdq_down_and_w4a8_hidden_quant_share_canonical_hidden",
        "final_combine_uses_mixed_down_peer_output",
        "expanded_row_idx_matches_arange",
        "hidden_q_mismatch_tolerance",
        "--hidden-q-mismatch-tol",
        "phase_l_composed_pipeline_device_probe_summary.json",
        "--require-npu",
    ):
        assert token in probe


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

    assert "SVDQ_WORKSPACE_REGION_COUNT = 16" in tiling_header
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
        "SVDQ_REGION_LOWRANK_RANK_1",
        "SVDQ_REGION_LOWRANK_RANK_2",
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
    assert "routedRows * hiddenSize * BF16_BYTES" in tiling
    assert "routedRows * hiddenSize * FP32_BYTES" in tiling
    assert "SVDQ_REGION_ACCUMULATOR_1, offset, routedRows * gateUpSize * BF16_BYTES" in tiling
    assert "SVDQ_REGION_ACCUMULATOR_2, offset, routedRows * hiddenSize * BF16_BYTES" in tiling
    assert "DispatchFFNCombineW4A8SVDQ AscendC kernel is not implemented yet" not in tiling
    assert "return ge::GRAPH_SUCCESS;" in tiling


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
    assert "DispatchFFNCombineW4A8SVDQ AscendC kernel is not implemented yet" not in tiling
    assert "return ge::GRAPH_SUCCESS;" in tiling

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
            "SVDQ_REGION_LOWRANK_RANK_1",
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
            "SVDQ_REGION_LOWRANK_RANK_1",
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
            "SVDQ_REGION_LOWRANK_RANK_1",
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
            "SVDQ_REGION_LOWRANK_RANK_2",
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
            "SVDQ_REGION_LOWRANK_RANK_2",
            "SVDQ_REGION_PROJECTION_2",
            "routedRows",
            "info.downRank",
            "info.hiddenSize",
            "0",
            "0",
            "0",
        ),
    )
    for (
        stage,
        factor,
        input_region,
        output_region,
        m,
        k,
        n,
        input_offset,
        output_offset,
        factor_offset,
    ) in expected_stage_shapes:
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


def test_svdq_cann_tiling_records_w4a8_residual_stage_contract():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()

    assert "SVDQ_RESIDUAL_STAGE_COUNT = 4" in tiling_header
    assert "SVDQ_RESIDUAL_GMM_COUNT = 2" in tiling_header
    assert "SVDQ_RESIDUAL_QUANT_COUNT = 2" in tiling_header
    assert "SVDQResidualStageId" in tiling_header
    assert "SVDQResidualStageShape" in tiling_header
    assert "SVDQResidualStageShape residualStageShapes[SVDQ_RESIDUAL_STAGE_COUNT]" in tiling_header
    assert "SVDQResidualQuantShape" in tiling_header
    assert "SVDQResidualQuantShape residualQuantShapes[SVDQ_RESIDUAL_QUANT_COUNT]" in tiling_header
    assert "SVDQResidualGmmShape" in tiling_header
    assert "SVDQResidualGmmShape residualGmmShapes[SVDQ_RESIDUAL_GMM_COUNT]" in tiling_header
    assert "SetResidualStageShape" in tiling
    assert "BuildResidualStageShapeTable" in tiling
    assert "BuildResidualStageShapeTable(tilingData)" in tiling
    assert "SetResidualQuantShape" in tiling
    assert "BuildResidualQuantShapeTable" in tiling
    assert "BuildResidualQuantShapeTable(tilingData)" in tiling
    assert "SetResidualGmmShape" in tiling
    assert "BuildResidualGmmShapeTable" in tiling
    assert "BuildResidualGmmShapeTable(tilingData)" in tiling
    assert "ResidualStageShape(uint32_t stageId)" in contract
    assert "return tilingData_.residualStageShapes[stageId]" in contract
    assert "ResidualQuantShape(uint32_t quantId)" in contract
    assert "return tilingData_.residualQuantShapes[quantId]" in contract
    assert "ResidualGmmShape(uint32_t gmmId)" in contract
    assert "return tilingData_.residualGmmShapes[gmmId]" in contract
    assert "SVDQResidualStageContract" in contract
    assert "ResidualStageContract(uint32_t stageId)" in contract
    assert "SVDQResidualExecutionPlan" in contract
    assert "SVDQResidualQuantLaunch" in contract
    assert "SVDQResidualGmmLaunch" in contract
    assert "ResidualExecutionPlan(uint32_t stageId)" in contract
    assert "ResidualQuantIdForStage(uint32_t stageId)" in contract
    assert "BuildResidualQuantLaunch(uint32_t stageId)" in contract
    assert "ResidualGmmIdForStage(uint32_t stageId)" in contract
    assert "BuildResidualGmmLaunch(uint32_t stageId)" in contract
    assert "ResidualStageReady(uint32_t stageId)" in contract
    assert "ResidualExecutionPlanReady(uint32_t stageId)" in contract
    assert "ResidualQuantLaunchReady(uint32_t stageId)" in contract
    assert "ResidualGmmLaunchReady(uint32_t stageId)" in contract
    assert "RunResidualDynamicQuantStage(uint32_t stageId)" in contract
    assert "RunResidualGmmStage(uint32_t stageId)" in contract
    assert "RunResidualStage(uint32_t stageId)" in contract
    assert "RunW4A8ResidualStages()" in contract

    for slot in (
        "SVDQ_RESIDUAL_WEIGHT1_SLOT = 1",
        "SVDQ_RESIDUAL_WEIGHT2_SLOT = 2",
        "SVDQ_RESIDUAL_SCALE1_SLOT = 4",
        "SVDQ_RESIDUAL_SCALE2_SLOT = 5",
        "SVDQ_RESIDUAL_BIAS1_SLOT = 6",
        "SVDQ_RESIDUAL_BIAS2_SLOT = 7",
    ):
        assert slot in contract
    assert "ResidualWeightAddress(uint32_t residualWeightSlot)" in contract
    assert "ResidualScaleAddress(uint32_t residualScaleSlot)" in contract
    assert "ResidualBiasAddress(uint32_t residualBiasSlot)" in contract
    assert "SVDQ_RESIDUAL_OP_DYNAMIC_QUANT" in contract
    assert "SVDQ_RESIDUAL_OP_W4A8_GMM" in contract
    assert "RunResidualDynamicQuantStage(stageId)" in contract
    assert "RunResidualGmmStage(stageId)" in contract
    assert "RunResidualStage(stageId)" in contract

    expected_stage_shapes = (
        (
            "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
            "SVDQ_REGION_ROUTED_X",
            "SVDQ_REGION_X_SCALE",
            "SVDQ_REGION_X_Q",
            "info.hiddenSize",
            "info.hiddenSize",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
            "SVDQ_REGION_X_Q",
            "SVDQ_REGION_X_SCALE",
            "SVDQ_REGION_ACCUMULATOR_1",
            "info.hiddenSize",
            "info.intermediateSize * 2",
            "weight1Slot",
            "scale1Slot",
        ),
        (
            "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
            "SVDQ_REGION_HIDDEN",
            "SVDQ_REGION_HIDDEN_SCALE",
            "SVDQ_REGION_HIDDEN_Q",
            "info.intermediateSize",
            "info.intermediateSize",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
            "SVDQ_REGION_HIDDEN_Q",
            "SVDQ_REGION_HIDDEN_SCALE",
            "SVDQ_REGION_ACCUMULATOR_2",
            "info.intermediateSize",
            "info.hiddenSize",
            "weight2Slot",
            "scale2Slot",
        ),
    )
    for stage, input_region, scale_region, output_region, k, n, weight_slot, scale_slot in expected_stage_shapes:
        assert stage in tiling_header
        assert stage in tiling
        assert input_region in tiling
        assert scale_region in tiling
        assert output_region in tiling
        assert "routedRows" in tiling
        assert k in tiling
        assert n in tiling
        assert weight_slot in tiling
        assert scale_slot in tiling
        assert f"case {stage}:" in contract
        assert input_region in contract
        assert scale_region in contract
        assert output_region in contract

    for token in (
        "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT:\n                return {stageId, SVDQ_RESIDUAL_OP_DYNAMIC_QUANT",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM1:\n                return {stageId, SVDQ_RESIDUAL_OP_W4A8_GMM",
        "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN:\n                return {stageId, SVDQ_RESIDUAL_OP_DYNAMIC_QUANT",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM2:\n                return {stageId, SVDQ_RESIDUAL_OP_W4A8_GMM",
        "SVDQ_RESIDUAL_BIAS1_SLOT",
        "SVDQ_RESIDUAL_BIAS2_SLOT",
    ):
        assert token in contract

    for token in (
        "quant.usesRouting = usesRouting",
        "quant.residualOnly = true",
        "SetResidualQuantShape(tilingData, 0, SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "SetResidualQuantShape(tilingData, 1, SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "routedRows, info.hiddenSize, routedRows, true",
        "routedRows, info.intermediateSize, routedRows, false",
    ):
        assert token in tiling

    residual_quant_descriptor_source = contract[
        contract.index("__aicore__ inline uint32_t ResidualQuantIdForStage") : contract.index(
            "__aicore__ inline uint32_t ResidualGmmIdForStage"
        )
    ]
    for token in (
        "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "WorkspaceAddress(shape.inputRegionId)",
        "WorkspaceAddress(shape.activationScaleRegionId)",
        "WorkspaceAddress(shape.outputRegionId)",
        "shape.usesRouting ? WorkspaceAddress(contract.routeIndexRegionId) : nullptr",
        "shape.usesRouting ? DispatchQuantRoutingTempWorkspace() : nullptr",
        "runtime_.expertTokenNums",
        "shape.scaleElements",
        "shape.usesRouting",
        "shape.residualOnly",
    ):
        assert token in residual_quant_descriptor_source

    for token in (
        "gmm.groupListType = 1",
        "gmm.groupType = 0",
        "gmm.splitItem = 2",
        "gmm.transB = false",
        "gmm.weightNz = true",
        "gmm.residualOnly = true",
        "SetResidualGmmShape(tilingData, 0, SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "SetResidualGmmShape(tilingData, 1, SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
        "bias1Slot",
        "bias2Slot",
        "info.expertPerRank",
    ):
        assert token in tiling

    residual_gmm_source = contract[
        contract.index("__aicore__ inline uint32_t ResidualGmmIdForStage") : contract.index(
            "__aicore__ inline bool ResidualStageReady"
        )
    ]
    for token in (
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
        "ResidualWeightAddress(shape.residualWeightSlot)",
        "ResidualScaleAddress(shape.residualScaleSlot)",
        "ResidualBiasAddress(shape.residualBiasSlot)",
        "runtime_.expertTokenNums",
        "shape.groupListType",
        "shape.groupType",
        "shape.splitItem",
        "shape.weightNz",
        "shape.residualOnly",
    ):
        assert token in residual_gmm_source

    gmm_ready_source = contract[
        contract.index("__aicore__ inline bool ResidualGmmLaunchReady") : contract.index(
            "__aicore__ inline bool RunResidualDynamicQuantStage"
        )
    ]
    for token in (
        "plan.opKind != SVDQ_RESIDUAL_OP_W4A8_GMM",
        "shape.inputRegionId == plan.inputRegionId",
        "shape.activationScaleRegionId == plan.activationScaleRegionId",
        "shape.outputRegionId == plan.outputRegionId",
        "shape.residualWeightSlot == plan.residualWeightSlot",
        "shape.residualScaleSlot == plan.residualScaleSlot",
        "shape.residualBiasSlot == plan.residualBiasSlot",
        "shape.listLen == tilingData_.info.expertPerRank",
        "shape.groupListType == 1",
        "shape.groupType == 0",
        "shape.splitItem == 2",
        "!shape.transB",
        "shape.weightNz",
        "launch.expertTokenNums != nullptr",
    ):
        assert token in gmm_ready_source

    gmm_execution_source = contract[
        contract.index("__aicore__ inline bool RunResidualGmmStage") : contract.index(
            "__aicore__ inline SVDQMixedEpilogueContract MixedEpilogueContract"
        )
    ]
    for token in (
        "plan.opKind != SVDQ_RESIDUAL_OP_W4A8_GMM",
        "!ResidualGmmLaunchReady(stageId)",
        "Residual W4A8 GMM must be implemented by the official AIC W4A8 kernel path.",
        "return false;",
    ):
        assert token in gmm_execution_source

    for token in (
        "RunResidualPackedW4A8ScalarGmmStage",
        "LoadResidualGmmExpertTokenCount",
        "ResolveResidualGmmExpert",
        "LoadResidualGmmInputINT8",
        "LoadResidualGmmActivationScale",
        "LoadResidualGmmWeightINT4",
        "LoadResidualGmmWeightScale",
        "LoadResidualGmmBias",
        "StoreResidualGmmOutputBF16",
        "UInt32BitsToFloat",
    ):
        assert token not in contract

    assert "stage.residualOnly = residualOnly" in tiling
    assert "true);" in tiling
    assert "DispatchFFNCombineW4A8SVDQ AscendC kernel is not implemented yet" not in tiling
    assert "return ge::GRAPH_SUCCESS;" in tiling

    assert (
        "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/moe_init_routing_quant_v2.cpp"
        in contract
    )
    assert "DispatchQuantRoutingTempWorkspace() const" in contract
    residual_quant_source = contract[
        contract.index("__aicore__ inline bool RunResidualDynamicQuantStage(uint32_t stageId)") : contract.index(
            "__aicore__ inline bool RunResidualGmmStage"
        )
    ]
    for token in (
        "SVDQResidualQuantLaunch launch = BuildResidualQuantLaunch(stageId)",
        "if (launch.usesRouting)",
        "moe_init_routing_quant_v2<bfloat16_t>",
        "launch.output",
        "launch.routeIndex",
        "launch.expertTokenNums",
        "launch.activationScale",
        "launch.workspace",
        "&routingTiling.moeInitRoutingQuantV2TilingData",
        "routingTiling.initRoutingQuantTilingKey",
        "return RunResidualHiddenQuantAIV(launch);",
    ):
        assert token in residual_quant_source

    for token in (
        "RunResidualHiddenQuantAIV(const SVDQResidualQuantLaunch& launch)",
        "QuantizeResidualHiddenRowAIV",
        "PackResidualHiddenOfficialI4AIV",
        "if (g_coreType == AIC)",
        "CopyInResidualQuantBf16(hiddenBf16, inputGm, row * launch.k + column,",
        "ReduceMax(reduceTmp, absHidden, scaleLocal, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "const float scale = maxAbs / 127.0f",
        "CopyOutResidualQuantScale(scaleGm, row, scaleLocal, 1)",
        "SetDeqScale(static_cast<half>(1.0f))",
        "Cast(hiddenI8, quantHalf, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "const uint32_t rowOffset = row * launch.k",
        "CopyOutResidualQuantI8(outputGm, rowOffset + packedOffset,",
        "CopyOutResidualQuantI8(outputGm, rowOffset + launch.k / 2 + packedOffset,",
    ):
        assert token in contract

    for token in (
        "RunResidualScalarDynamicQuantStage",
        "LoadResidualQuantInputBF16",
        "StoreResidualQuantOutputINT8",
        "StoreResidualQuantScaleFP32",
        "RoundQuantValue",
        "ClampInt8QuantValue",
    ):
        assert token not in contract

    quant_ready_source = contract[
        contract.index("__aicore__ inline bool ResidualQuantLaunchReady") : contract.index(
            "__aicore__ inline bool ResidualGmmLaunchReady"
        )
    ]
    for token in (
        "plan.opKind != SVDQ_RESIDUAL_OP_DYNAMIC_QUANT",
        "shape.inputRegionId != plan.inputRegionId",
        "shape.activationScaleRegionId != plan.activationScaleRegionId",
        "shape.outputRegionId != plan.outputRegionId",
        "shape.scaleElements != shape.m",
        "launch.expertTokenNums == nullptr",
        "if (shape.usesRouting)",
        "routingTiling.initRoutingQuantTilingKey != 0",
        "routingTiling.routingWorkspaceBytes > 0",
        "return launch.routeIndex == nullptr && launch.workspace == nullptr",
    ):
        assert token in quant_ready_source


def test_svdq_kernel_records_mixed_epilogue_and_final_combine_contracts():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()

    for token in (
        "SVDQ_MIXED_EPILOGUE_COUNT = 2",
        "SVDQ_MIXED_EPILOGUE_VECTOR_TILE = 64",
        "SVDQ_MIXED_EPILOGUE_UB_BYTES = 196352",
        "SVDQMixedEpilogueShape",
        "SVDQMixedEpilogueContract",
        "SVDQMixedEpilogueLaunch",
        "MixedEpilogueShape(uint32_t epilogueId)",
        "BuildMixedEpilogueLaunch(uint32_t epilogueId)",
        "MixedEpilogueContract(uint32_t epilogueId)",
        "MixedEpilogueReady(uint32_t epilogueId)",
        "RunMixedEpilogueStage(uint32_t epilogueId)",
        "RunMixedEpilogueStages()",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_REGION_ACCUMULATOR_1",
        "SVDQ_REGION_PROJECTION_1",
        "SVDQ_REGION_X_SCALE",
        "SVDQ_REGION_HIDDEN",
        "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2",
        "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_REGION_ACCUMULATOR_2",
        "SVDQ_REGION_PROJECTION_2",
        "SVDQ_REGION_HIDDEN_SCALE",
        "SVDQ_REGION_PEER_OUTPUT",
        "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE",
    ):
        assert token in contract

    for token in (
        "SVDQ_MIXED_EPILOGUE_COUNT = 2",
        "SVDQMixedEpilogueShape",
        "mixedEpilogueShapes[SVDQ_MIXED_EPILOGUE_COUNT]",
    ):
        assert token in tiling_header

    for token in (
        "SetMixedEpilogueShape",
        "BuildMixedEpilogueShapeTable",
        "BuildMixedEpilogueShapeTable(tilingData)",
        "epilogue.swigluLimit = tilingData->info.swigluLimit",
        "epilogue.appliesSwiGLU = appliesSwiGLU",
        "SetMixedEpilogueShape(tilingData, 0, SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_REGION_ACCUMULATOR_1, SVDQ_REGION_PROJECTION_1, SVDQ_REGION_X_SCALE, SVDQ_REGION_HIDDEN",
        "routedRows, info.intermediateSize * 2, info.intermediateSize * 2, info.intermediateSize",
        "0, info.intermediateSize, true",
        "SetMixedEpilogueShape(tilingData, 1, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_REGION_ACCUMULATOR_2, SVDQ_REGION_PROJECTION_2, SVDQ_REGION_HIDDEN_SCALE, SVDQ_REGION_PEER_OUTPUT",
        "routedRows, info.hiddenSize, info.hiddenSize, info.hiddenSize",
        "SVDQ_INVALID_ID, SVDQ_INVALID_ID, false",
    ):
        assert token in tiling

    epilogue_source = contract[
        contract.index("__aicore__ inline SVDQMixedEpilogueContract MixedEpilogueContract") : contract.index(
            "__aicore__ inline bool RunMixedEpilogueStages"
        )
    ]
    for token in (
        "BuildMixedEpilogueLaunch(epilogueId)",
        "WorkspaceAddress(shape.residualRegionId)",
        "WorkspaceAddress(shape.lowRankRegionId)",
        "WorkspaceAddress(shape.scaleRegionId)",
        "WorkspaceAddress(shape.outputRegionId)",
        "shape.residualColumns == tilingData_.info.intermediateSize * 2",
        "shape.lowRankColumns == tilingData_.info.intermediateSize * 2",
        "shape.outputColumns == tilingData_.info.intermediateSize",
        "shape.gateColumnOffset == 0",
        "shape.upColumnOffset == tilingData_.info.intermediateSize",
        "shape.residualColumns == tilingData_.info.hiddenSize",
        "shape.lowRankColumns == tilingData_.info.hiddenSize",
        "shape.outputColumns == tilingData_.info.hiddenSize",
        "shape.gateColumnOffset == SVDQ_INVALID_ID",
        "shape.upColumnOffset == SVDQ_INVALID_ID",
        "if (!MixedEpilogueReady(epilogueId))",
        "RunMixedOutputEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)",
        "RunMixedSwiGLUEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)",
        "CopyInMixedEpilogueBf16(residualBf16, residualGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "CopyInMixedEpilogueBf16(lowRankBf16, lowRankGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "Add(residual, residual, lowRank, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "CopyOutMixedEpilogueBf16(outputGm, offset, outputBf16, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "CopyInMixedEpilogueBf16(residualGateBf16, residualGm, gateOffset,",
        "CopyInMixedEpilogueBf16(lowRankGateBf16, lowRankGm, gateOffset,",
        "CopyInMixedEpilogueBf16(residualUpBf16, residualGm, upOffset,",
        "CopyInMixedEpilogueBf16(lowRankUpBf16, lowRankGm, upOffset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "Exp(tmp, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "Div(hidden, gate, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "Mul(hidden, hidden, up, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)",
        "return RunMixedSwiGLUEpilogueAIV(launch);",
        "return RunMixedOutputEpilogueAIV(launch);",
    ):
        assert token in epilogue_source

    for token in (
        "RunMixedSwiGLUEpilogueStage",
        "RunMixedOutputEpilogueStage",
        "LoadMixedEpilogueResidualBF16",
        "LoadMixedEpilogueLowRankBF16",
        "StoreMixedEpilogueOutputBF16",
        "SiluFloat",
        "ExpApproxFloat",
    ):
        assert token not in contract

    for token in (
        "SVDQFinalCombineContract",
        "SVDQFinalCombineShape",
        "SVDQFinalCombineLaunch",
        "FinalCombineShape() const",
        "BuildFinalCombineLaunch() const",
        "FinalCombineContract() const",
        "FinalCombineReady() const",
        "RunFinalCombine() const",
        "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "SVDQ_REGION_PEER_OUTPUT",
        "SVDQ_REGION_EXPANDED_ROW_IDX",
        "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE",
        "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE",
        "launch.output != nullptr",
        "launch.expertId != nullptr",
        "launch.probs != nullptr",
    ):
        assert token in contract

    for token in (
        "SVDQFinalCombineShape",
        "SVDQFinalCombineShape finalCombineShape",
        "SVDQFinalCombineTiling",
        "SVDQFinalCombineTiling finalCombine",
        "MoeTokenUnpermuteTilingData moeTokenUnpermuteTilingData",
    ):
        assert token in tiling_header

    for token in (
        "BuildFinalCombineShape",
        "BuildFinalCombineShape(tilingData)",
        "BuildFinalCombineTiling",
        "BuildFinalCombineTiling(tilingData)",
        "MoeTokenUnpermuteTiling(info.m * info.topK, info.hiddenSize, info.topK",
        "finalCombine.stageId = SVDQ_STAGE_UNPERMUTE_COMBINE",
        "finalCombine.inputRegionId = SVDQ_REGION_PEER_OUTPUT",
        "finalCombine.routeRegionId = SVDQ_REGION_EXPANDED_ROW_IDX",
        "finalCombine.m = info.m",
        "finalCombine.routedRows = info.maxOutputSize",
        "finalCombine.hiddenSize = info.hiddenSize",
        "finalCombine.topK = info.topK",
        "finalCombine.activeSlots = info.m * info.topK",
    ):
        assert token in tiling

    final_combine_source = contract[
        contract.index("__aicore__ inline SVDQFinalCombineContract FinalCombineContract") : contract.index(
            "private:"
        )
    ]
    for token in (
        "SVDQFinalCombineShape shape = FinalCombineShape()",
        "SVDQFinalCombineLaunch launch = BuildFinalCombineLaunch()",
        "WorkspaceAddress(shape.inputRegionId)",
        "WorkspaceAddress(shape.routeRegionId)",
        "runtime_.expertId",
        "runtime_.probs",
        "runtime_.out",
        "shape.stageId == contract.stageId",
        "shape.inputRegionId == contract.inputRegionId",
        "shape.routeRegionId == contract.routeRegionId",
        "shape.m == tilingData_.info.m",
        "shape.routedRows == tilingData_.info.maxOutputSize",
        "shape.hiddenSize == tilingData_.info.hiddenSize",
        "shape.topK == tilingData_.info.topK",
        "shape.activeSlots == tilingData_.info.m * tilingData_.info.topK",
        "shape.routedRows >= shape.activeSlots",
        "launch.expertId != nullptr",
        "launch.probs != nullptr",
        "finalCombineTiling.coreNum > 0",
        "finalCombineTiling.moeTokenUnpermuteTilingData.hidden_size == shape.hiddenSize",
        "finalCombineTiling.moeTokenUnpermuteTilingData.top_k == shape.topK",
        "finalCombineTiling.moeTokenUnpermuteTilingData.num_out_tokens == shape.activeSlots",
        "if (!FinalCombineReady())",
        "SVDQFinalCombineLaunch launch = BuildFinalCombineLaunch()",
        "KernelMoeTokenUnpermute<bfloat16_t, int32_t, float, true> kernelMoeTokenUnpermuteOp",
        "kernelMoeTokenUnpermuteOp.Init(launch.input, launch.routeIndex, launch.probs, launch.output",
        "&tilingData_.finalCombine.moeTokenUnpermuteTilingData",
        "kernelMoeTokenUnpermuteOp.Process()",
        "return true;",
    ):
        assert token in final_combine_source

    for token in (
        "LoadFinalCombineRouteIndex",
        "LoadFinalCombineProb",
        "LoadFinalCombineInput",
        "StoreFinalCombineOutput",
        "AccumulateFinalCombineOutput",
    ):
        assert token not in contract

    process = contract[
        contract.index("__aicore__ inline void Process()") : contract.index(
            "__aicore__ inline bool HasCompleteTilingContract()"
        )
    ]
    expected_process_order = (
        "RunDispatchRoutingStage()",
        "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)",
        "RunResidualStage(SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT)",
        "RunResidualStage(SVDQ_RESIDUAL_STAGE_W4A8_GMM1)",
        "RunMixedEpilogueStage(0)",
        "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_DOWN)",
        "RunResidualStage(SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN)",
        "RunResidualStage(SVDQ_RESIDUAL_STAGE_W4A8_GMM2)",
        "RunMixedEpilogueStage(1)",
        "RunFinalCombine()",
    )
    offset = 0
    for token in expected_process_order:
        index = process.find(token, offset)
        assert index >= 0, token
        offset = index + len(token)


def test_svdq_kernel_records_dispatch_routing_contract_before_lowrank():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()

    for token in (
        "SVDQDispatchRoutingContract",
        "DispatchRoutingContract() const",
        "DispatchRoutingReady() const",
        "RunDispatchRoutingStage() const",
        "DispatchRoutingTiling() const",
        "DispatchRoutingTempWorkspace() const",
        "tilingData_.dispatchRouting",
        "SVDQ_STAGE_BF16_DISPATCH",
        "SVDQ_REGION_ROUTED_X",
        "SVDQ_REGION_EXPANDED_ROW_IDX",
        "SVDQ_SYNC_DISPATCH_TO_QUANT_1",
        "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1",
        "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE",
    ):
        assert token in contract

    dispatch_source = contract[
        contract.index("__aicore__ inline bool RunDispatchRoutingStage() const") : contract.index(
            "__aicore__ inline SVDQResidualStageContract"
        )
    ]
    assert "svdq_moe_init_routing_v2<bfloat16_t>" in dispatch_source
    assert "WorkspaceAddress(contract.routedOutputRegionId)" in dispatch_source
    assert "WorkspaceAddress(contract.routeIndexRegionId)" in dispatch_source
    assert "&routingTiling.moeInitRoutingV2TilingData" in dispatch_source
    assert "routingTiling.bf16RoutingTilingKey" in dispatch_source
    assert "return true;" in dispatch_source
    dispatch_ready_source = contract[
        contract.index("__aicore__ inline bool DispatchRoutingReady() const") : contract.index(
            "__aicore__ inline bool RunDispatchRoutingStage() const"
        )
    ]
    assert "routingTiling.bf16RoutingTilingKey != 0" in dispatch_ready_source
    assert "routingTiling.bf16RoutingWorkspaceBytes > 0" in dispatch_ready_source

    process = contract[
        contract.index("__aicore__ inline void Process()") : contract.index(
            "__aicore__ inline bool HasCompleteTilingContract()"
        )
    ]
    assert process.index("RunDispatchRoutingStage()") < process.index(
        "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)"
    )


def test_svdq_cann_lowrank_down_up_component_contract_is_wired():
    op_root = REPO_ROOT / "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq"
    tiling = (op_root / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp").read_text()
    tiling_header = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h").read_text()
    contract = (op_root / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h").read_text()
    lowrank_tiling = (op_root / "op_kernel/lowrank/svdq_fused_down_up_tiling.h").read_text()
    lowrank_header = (op_root / "op_kernel/lowrank/svdq_fused_down_up.hpp").read_text()
    lowrank_cpp = (op_root / "op_kernel/lowrank/svdq_fused_down_up.cpp").read_text()
    lowrank_debug_tiling = (op_root / "op_kernel/lowrank/svdq_lowrank_debug_readback_tiling.h").read_text()
    lowrank_debug_header = (op_root / "op_kernel/lowrank/svdq_lowrank_debug_readback.h").read_text()
    lowrank_debug_kernel = (op_root / "op_kernel/lowrank/svdq_lowrank_debug_readback.cpp").read_text()
    lowrank_debug_alias_root = REPO_ROOT / "csrc/mc2/svdq_low_rank_debug_readback"
    lowrank_debug_alias_cmake = (lowrank_debug_alias_root / "op_host/CMakeLists.txt").read_text()
    lowrank_debug_alias_kernel = (lowrank_debug_alias_root / "svdq_low_rank_debug_readback.cpp").read_text()
    lowrank_debug_torch_adapter = (lowrank_debug_alias_root / "svdq_low_rank_debug_readback_torch_adpt.h").read_text()
    torch_binding = (REPO_ROOT / "csrc/torch_binding.cpp").read_text()
    torch_binding_meta = (REPO_ROOT / "csrc/torch_binding_meta.cpp").read_text()
    lowrank_debug_probe = (REPO_ROOT / "tools/svdq_lowrank_debug_readback_probe.py").read_text()

    assert '#include "lowrank/svdq_fused_down_up_tiling.h"' in tiling_header
    assert '#include "lowrank/svdq_fused_down_up.hpp"' in contract
    assert '#include "svdq_fused_down_up.hpp"' in lowrank_cpp
    for helper_include in (
        '#include "layout.h"',
        '#include "mem.h"',
        '#include "gm_to_l1_iterator.h"',
        '#include "l1_to_l0_iterator.h"',
        '#include "l0c_to_gm_iterator.h"',
        '#include "mma.h"',
    ):
        assert helper_include in lowrank_header
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
    assert "SVDQLowRankOutputTilePlan" in lowrank_header
    assert "SVDQLowRankTileTensorPlan" in lowrank_header
    assert "SVDQLowRankMmadTilePlan" in lowrank_header
    assert "SVDQLowRankMmadBufferPlan" in lowrank_header
    assert "SVDQLowRankMmadPipelinePlan" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_M_TILE = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_N_TILE = 64" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_K_TILE = 64" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_M_ALIGNMENT = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_N_ALIGNMENT = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_K_ALIGNMENT = 16" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_FORMAT_ND = 0" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_FORMAT_NZ = 1" in lowrank_header
    assert "SVDQ_LOWRANK_MMAD_FORMAT_ZN = 2" in lowrank_header
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
    assert "OutputTilePlan(uint32_t tileId)" in lowrank_header
    assert "OutputTileKTileCount(const SVDQLowRankOutputTilePlan& outputTilePlan)" in lowrank_header
    assert "KTilePlan(" in lowrank_header
    assert "BuildTileTensorPlan(" in lowrank_header
    assert "BuildMmadTilePlan(" in lowrank_header
    assert "BuildMmadBufferPlan(" in lowrank_header
    assert "BuildMmadPipelinePlan(" in lowrank_header
    assert "HasCompatibleShape() const" in lowrank_header
    assert "HasCompleteFootprint() const" in lowrank_header
    assert "HasCompletePipeline() const" in lowrank_header
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
    assert "RunMmadTileBF16(" in lowrank_header
    assert "RunPlannedTileBF16(" in lowrank_header
    assert "class SVDQLowRankDebugReadback" in lowrank_header
    assert "IsEnabled() const" in lowrank_header
    assert "#ifdef SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in lowrank_header
    assert "lowRankOp.HasCompleteContract()" in lowrank_header
    assert "lowRankOp.Process();" in lowrank_header
    assert '#include "svdq_lowrank_debug_readback_tiling.h"' in lowrank_debug_header
    assert "struct SVDQLowRankDebugTilingData" in lowrank_debug_tiling
    assert "SVDQFusedDownUpTiling gateUpInvocation" in lowrank_debug_tiling
    assert "SVDQFusedDownUpTiling downInvocation" in lowrank_debug_tiling
    assert "class SVDQLowRankDebugReadbackKernel" in lowrank_debug_header
    assert "BuildGateUpArgs() const" in lowrank_debug_header
    assert "BuildDownArgs() const" in lowrank_debug_header
    assert "GM_ADDR rankWorkspace" in lowrank_debug_header
    assert "runtime_.rankWorkspace = workspaceGM" in lowrank_debug_header
    assert "args.rank = runtime_.rankWorkspace" in lowrank_debug_header
    assert "runtime_.gateSvdqL2" in lowrank_debug_header
    assert "runtime_.upSvdqL2" in lowrank_debug_header
    assert "svdq_low_rank_debug_readback(" in lowrank_debug_kernel
    assert "(void)workspaceGM" not in lowrank_debug_kernel
    assert "downAccumulator, workspaceGM, tilingGM" in lowrank_debug_kernel
    assert "add_op_to_compiled_list()" in lowrank_debug_alias_cmake
    assert "svdq_low_rank_debug_readback(" in lowrank_debug_alias_kernel
    assert "svdq_lowrank_debug_readback.h" in lowrank_debug_alias_kernel
    assert "(void)workspaceGM" not in lowrank_debug_alias_kernel
    assert "downAccumulator, workspaceGM, tilingGM" in lowrank_debug_alias_kernel
    assert "GM_ADDR gateUpOutput" in lowrank_debug_kernel
    assert "GM_ADDR downOutput" in lowrank_debug_kernel
    assert "GM_ADDR gateUpAccumulator" in lowrank_debug_kernel
    assert "GM_ADDR downAccumulator" in lowrank_debug_kernel
    assert "SVDQLowRankDebugReadbackKernel op" in lowrank_debug_kernel
    assert "op.Process();" in lowrank_debug_kernel
    assert "aclnnSVDQLowRankDebugReadback" in lowrank_debug_torch_adapter
    assert "svdq_low_rank_debug_readback(" in lowrank_debug_torch_adapter
    assert "gate_up_output" in lowrank_debug_torch_adapter
    assert "down_output" in lowrank_debug_torch_adapter
    assert "gate_up_accumulator" in lowrank_debug_torch_adapter
    assert "down_accumulator" in lowrank_debug_torch_adapter
    assert "svdq_low_rank_debug_readback_torch_adpt.h" in torch_binding
    assert 'ops.def(\n        "svdq_low_rank_debug_readback' in torch_binding
    assert 'ops.impl("svdq_low_rank_debug_readback", torch::kPrivateUse1' in torch_binding
    assert "svdq_low_rank_debug_readback_meta" in torch_binding_meta
    assert (
        'ops.impl("svdq_low_rank_debug_readback", &vllm_ascend::meta::svdq_low_rank_debug_readback_meta)'
        in torch_binding_meta
    )
    assert "torch.ops._C_ascend.svdq_low_rank_debug_readback" in lowrank_debug_probe
    assert "_load_validation_layer(" in lowrank_debug_probe
    assert "build_svdq_bf16_stage_reference" in lowrank_debug_probe
    assert "--require-accumulator-readback" in lowrank_debug_probe
    assert 'DEFAULT_SUMMARY_NAME = "phase_j_lowrank_debug_readback_probe_summary.json"' in lowrank_debug_probe
    assert "_aggregate_stage_errors(results)" in lowrank_debug_probe
    assert '"aggregate_stage_errors": _aggregate_stage_errors(results)' in lowrank_debug_probe
    for aggregate_field in (
        "layer_count",
        "expert_count",
        "stage_comparison_count",
        "max_abs_by_stage",
        "mean_abs_by_stage",
        "all_stage_outputs_finite",
        "max_abs_overall",
        "mean_abs_overall",
    ):
        assert aggregate_field in lowrank_debug_probe
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
    assert "GM_ADDR rank;" in lowrank_header
    assert "args_.rank != nullptr" in lowrank_header
    assert "GM_ADDR inputBase = stageIndex == 0 ? args_.input : args_.rank;" in lowrank_header
    assert "GM_ADDR outputBase = stageIndex == 0 ? args_.rank : args_.output;" in lowrank_header
    assert "ExecuteStage(stageIndex, coreIdx, scheduledCoreCount, resource)" in lowrank_header
    assert "SVDQOfficialBF16Resource resource;" in lowrank_header
    assert "SVDQOfficialBF16BlockMmad blockMmad(resource);" in lowrank_header
    assert "SVDQLowRankBF16RankBlockMmad blockMmad(resource);" not in lowrank_header
    assert "stage.stageKind == SVDQ_LOWRANK_STAGE_DOWN_PROJECT" in lowrank_header
    assert "StageOutputColumnTile(stage)" in lowrank_header
    assert "SVDQOfficialBF16BlockScheduler blockScheduler;" in lowrank_header
    assert "blockScheduler.Update(" in lowrank_header
    assert "blockScheduler.GetCoreLoops()" in lowrank_header
    assert "blockScheduler.GetBlockCoord(loopIdx)" in lowrank_header
    assert "blockScheduler.GetActualBlockShape(blockCoord)" in lowrank_header
    assert "StageOutputTilePlan(stageIndex, tileRange.tileStart + tileOffset)" in lowrank_header
    assert "AscendC::SyncAll()" in lowrank_header
    assert "stage-ordered so L2 stages cannot read rank workspace" in lowrank_header
    assert "IsImplemented() const" in lowrank_header
    assert "return HasCompleteContract();" in lowrank_header
    assert "args_.tiling.invocationId < SVDQ_LOWRANK_INVOCATION_COUNT" in lowrank_header
    assert "args_.expertPerRank > 0" in lowrank_header
    assert "args_.tiling.rowTile > 0" in lowrank_header
    assert "args_.tiling.outputColumnTile > 0" in lowrank_header
    assert "args_.tiling.kTile > 0" in lowrank_header
    assert "args_.tiling.coreCount > 0" in lowrank_header
    assert "args_.accumulator != nullptr" in lowrank_header
    assert (
        "args_.tiling.secondInputColumnOffset + args_.tiling.secondRankColumns <= TotalRankColumns()" in lowrank_header
    )
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
    assert (
        "MatrixAddress(expert.output, rowOffset, stage.outputStrideColumns, tilePlan.outputColumnOffset)"
        in lowrank_header
    )
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
    assert "l1InputOffset == 0" in lowrank_header
    assert "l1FactorOffset == buffer.l1InputBytes" in lowrank_header
    assert "l0AOffset == 0" in lowrank_header
    assert "l0BOffset == 0" in lowrank_header
    assert "l0COffset == 0" in lowrank_header
    assert "inputGmFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND" in lowrank_header
    assert "factorGmFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND" in lowrank_header
    assert "inputL1Format == SVDQ_LOWRANK_MMAD_FORMAT_NZ" in lowrank_header
    assert "factorL1Format == SVDQ_LOWRANK_MMAD_FORMAT_NZ" in lowrank_header
    assert "l0AFormat == SVDQ_LOWRANK_MMAD_FORMAT_NZ" in lowrank_header
    assert "l0BFormat == SVDQ_LOWRANK_MMAD_FORMAT_ZN" in lowrank_header
    assert "outputFormat == SVDQ_LOWRANK_MMAD_FORMAT_ND" in lowrank_header
    assert "loadInputGmToL1 &&" in lowrank_header
    assert "loadFactorGmToL1 && loadInputL1ToL0A && loadFactorL1ToL0B" in lowrank_header
    assert "factorLoadsTransposed && runMmad && storeL0C" in lowrank_header
    assert "initAccumulator == buffer.tile.tile.accumulatesFirstKTile" in lowrank_header
    assert "storesAccumulator == buffer.storesAccumulator" in lowrank_header
    assert "storesOutput == buffer.storesOutput" in lowrank_header
    assert "bufferPlan.l1InputBytes" in lowrank_header
    assert "bufferPlan.tile.tile.accumulatesFirstKTile" in lowrank_header
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
    assert (
        "for (uint32_t outputOffset = 0; outputOffset < tilePlan.tile.outputColumnCount; ++outputOffset)"
        in lowrank_header
    )
    assert "const float accumulator = AccumulateScalarBF16(tilePlan, rowOffset, outputOffset)" in lowrank_header
    assert "if (tilePlan.accumulatesLastKTile)" in lowrank_header
    assert "StoreOutputBF16(tilePlan, rowOffset, outputOffset, static_cast<bfloat16_t>(accumulator))" in lowrank_header
    assert "StoreAccumulatorFP32(tilePlan, rowOffset, outputOffset, accumulator)" in lowrank_header
    assert "#ifdef __DAV_C220_CUBE__" in lowrank_header
    assert "AsdopsBuffer<ArchType::ASCEND_V220> buffers" in lowrank_header
    assert "inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.input))" in lowrank_header
    assert "factorGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.factor))" in lowrank_header
    assert "outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(tensorPlan.output))" in lowrank_header
    assert "accumulatorGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tensorPlan.accumulator))" in lowrank_header
    assert "buffers.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(pipelinePlan.l1InputOffset)" in lowrank_header
    assert "buffers.GetBuffer<BufferType::ASCEND_CB, bfloat16_t>(pipelinePlan.l1FactorOffset)" in lowrank_header
    assert "const uint32_t l0AOffset = pipelinePlan.l0AOffset + pipelinePlan.buffer.l0ABytes * l0Stage" in lowrank_header
    assert "const uint32_t l0BOffset = pipelinePlan.l0BOffset + pipelinePlan.buffer.l0BBytes * l0Stage" in lowrank_header
    assert "buffers.GetBuffer<BufferType::ASCEND_L0A, bfloat16_t>(l0AOffset)" in lowrank_header
    assert "buffers.GetBuffer<BufferType::ASCEND_L0B, bfloat16_t>(l0BOffset)" in lowrank_header
    assert "buffers.GetBuffer<BufferType::ASCEND_L0C, float>(pipelinePlan.l0COffset)" in lowrank_header
    assert "gm_to_l1<ArchType::ASCEND_V220, bfloat16_t, DataFormatT::ND, DataFormatT::NZ>" in lowrank_header
    assert "gm_to_l1<ArchType::ASCEND_V220, bfloat16_t, DataFormatT::ND, DataFormatT::ZN>" in lowrank_header
    assert "l1_to_l0_a<ArchType::ASCEND_V220, bfloat16_t, false, DataFormatT::NZ, DataFormatT::ZZ>" in lowrank_header
    assert "l1_to_l0_b<ArchType::ASCEND_V220, bfloat16_t, true, DataFormatT::ZN, DataFormatT::NZ>" in lowrank_header
    assert "mmad<ArchType::ASCEND_V220, bfloat16_t, bfloat16_t, float, false>" in lowrank_header
    assert "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in lowrank_header
    assert "l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, float, float>" in lowrank_header
    assert "accumulatorGm, l0C, tile.mActual, tile.nActual, tile.nRound" in lowrank_header
    assert "l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, bfloat16_t, float>" in lowrank_header
    assert "Production cube execution keeps partial K-loop sums resident in" in lowrank_header
    assert "mirrors the" in lowrank_header
    assert "current FP32 accumulator to GM for host-readable validation" in lowrank_header
    assert "if (!pipelinePlan.storesOutput)" in lowrank_header
    assert "return true" in lowrank_header
    assert "AscendC::PipeBarrier<PIPE_MTE2>()" in lowrank_header
    assert "AscendC::PipeBarrier<PIPE_MTE1>()" in lowrank_header
    assert "AscendC::PipeBarrier<PIPE_M>()" in lowrank_header
    assert "const bool initC = pipelinePlan.initAccumulator" in lowrank_header
    assert "return RunMmadTileBF16(pipelinePlan)" in lowrank_header
    assert "return false" in lowrank_header
    assert "const uint32_t coreIdx = AscendC::GetBlockIdx()" in lowrank_header
    assert "const uint32_t runtimeCoreCount = AscendC::GetBlockNum()" in lowrank_header
    assert "const SVDQLowRankCoreTileRange tileRange = CoreTileRange(coreIdx, scheduledCoreCount)" in lowrank_header
    assert (
        "const SVDQLowRankOutputTilePlan outputTilePlan = OutputTilePlan(tileRange.tileStart + tileOffset)"
        in lowrank_header
    )
    assert "const uint32_t kTileCount = OutputTileKTileCount(outputTilePlan)" in lowrank_header
    assert "for (uint32_t kTileIndex = 0; kTileIndex < kTileCount; ++kTileIndex)" in lowrank_header
    assert "const SVDQLowRankTilePlan tilePlan = KTilePlan(outputTilePlan, kTileIndex)" in lowrank_header
    assert "const SVDQLowRankTileTensorPlan tileTensorPlan = BuildTileTensorPlan(tilePlan)" in lowrank_header
    assert "const SVDQLowRankMmadTilePlan mm" in lowrank_header
    assert "if (!mmadTilePlan.HasCompatibleShape())" in lowrank_header
    assert "const SVDQLowRankMmadBufferPlan bufferPlan = BuildMmadBufferPlan(mmadTilePlan)" in lowrank_header
    assert "if (!bufferPlan.HasCompleteFootprint())" in lowrank_header
    assert "const SVDQLowRankMmadPipelinePlan pipelinePlan = BuildMmadPipelinePlan(bufferPlan)" in lowrank_header
    assert "if (!pipelinePlan.HasCompletePipeline())" in lowrank_header
    assert "if (!RunPlannedTileBF16(pipelinePlan))" in lowrank_header
    assert "return RunScalarTileBF16(pipelinePlan.buffer.tile.tile)" in lowrank_header
    assert "rowTiles * StageColumnTileCount(stage)" in lowrank_header
    assert "const uint32_t tilesPerRow = columnTiles" in lowrank_header
    assert "Min(outputColumnTile, stage.outputColumns - outputColumnOffset)" in lowrank_header
    assert "Min(args_.tiling.kTile, stage.inputColumns - kColumnOffset)" in lowrank_header
    assert "kColumnOffset >= stage.inputColumns" in lowrank_header
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
    assert "WorkspaceAddress(invocation.rankRegionId)" in contract
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
    process = contract[
        contract.index("__aicore__ inline void Process()") : contract.index(
            "__aicore__ inline bool HasCompleteTilingContract()"
        )
    ]
    assert "if (!RunBF16LowRankStages())" not in process
    assert process.index("ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)") < process.index(
        "RunMixedEpilogueStage(0)"
    )
    assert process.index("RunMixedEpilogueStage(0)") < process.index(
        "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_DOWN)"
    )

    gate_up_call = (
        "SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_GATE_UP,\n"
        "        SVDQ_REGION_ROUTED_X, SVDQ_REGION_LOWRANK_RANK_1, SVDQ_REGION_PROJECTION_1, SVDQ_FACTOR_GATE_UP_L1,\n"
        "        SVDQ_FACTOR_GATE_L2, SVDQ_FACTOR_UP_L2, routedRows, info.hiddenSize, info.gateRank, info.upRank,\n"
        "        info.intermediateSize * 2, info.gateRankOffset, 0, info.upRankOffset, info.intermediateSize,\n"
        "        info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_1)"
    )
    down_call = (
        "SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_DOWN,\n"
        "        SVDQ_REGION_HIDDEN, SVDQ_REGION_LOWRANK_RANK_2, SVDQ_REGION_PROJECTION_2, SVDQ_FACTOR_DOWN_L1,\n"
        "        SVDQ_FACTOR_DOWN_L2, SVDQ_INVALID_ID, routedRows, info.intermediateSize, info.downRank, 0,\n"
        "        info.hiddenSize, 0, 0, 0, 0, info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_2)"
    )
    assert gate_up_call in tiling
    assert down_call in tiling
    assert "gateUpSvdqL2" not in lowrank_header
    assert "gate_up_svdq_l2" not in lowrank_header
    assert "gateUpSvdqL2" not in lowrank_tiling
    assert "gate_up_svdq_l2" not in lowrank_tiling
    assert "gateUpSvdqL2" not in lowrank_debug_tiling
    assert "gate_up_svdq_l2" not in lowrank_debug_tiling
    assert "gateUpSvdqL2" not in lowrank_debug_header
    assert "gate_up_svdq_l2" not in lowrank_debug_header
    assert "gateUpSvdqL2" not in lowrank_debug_kernel
    assert "gate_up_svdq_l2" not in lowrank_debug_kernel
    assert "gateUpSvdqL2" not in lowrank_debug_alias_kernel
    assert "gate_up_svdq_l2" not in lowrank_debug_alias_kernel
    assert "gateUpSvdqL2" not in lowrank_debug_torch_adapter
    assert "gate_up_svdq_l2" not in lowrank_debug_torch_adapter
    assert "gateUpSvdqL2" not in lowrank_debug_probe
    assert "gate_up_svdq_l2" not in lowrank_debug_probe


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
    assert "SynchronizeStageBoundary(" in contract
    assert "ValidateSyncFlag(" in contract
    assert "HasCompleteSyncFlagTable() const" in contract
    assert "AscendC::SyncAll();" in contract
    assert "flag.producerSignalIndex == flagId" in contract
    assert "flag.consumerWaitIndex == flagId" in contract
    assert "HasCompleteTilingContract" in contract
    assert "tilingData_.info.syncFlagCount == SVDQ_SYNC_FLAG_COUNT" in contract
    assert "HasCompleteSyncFlagTable()" in contract
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
            "SVDQ_REGION_LOWRANK_RANK_1",
            "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_LOWRANK_RANK_1",
            "SVDQ_REGION_LOWRANK_RANK_1",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_GATE_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_LOWRANK_RANK_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_INVALID_ID",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_UP_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_1",
            "SVDQ_REGION_LOWRANK_RANK_1",
            "SVDQ_REGION_PROJECTION_1",
            "SVDQ_INVALID_ID",
            "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L1_GEMM",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_REGION_HIDDEN",
            "SVDQ_REGION_LOWRANK_RANK_2",
            "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2",
            "SVDQ_INVALID_ID",
        ),
        (
            "SVDQ_BF16_STAGE_DOWN_L2_GEMM",
            "SVDQ_STAGE_LOWRANK_2",
            "SVDQ_REGION_LOWRANK_RANK_2",
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

    process_source = contract[
        contract.index("__aicore__ inline void Process()") : contract.index(
            "__aicore__ inline bool HasCompleteTilingContract()"
        )
    ]
    for boundary in (
        "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_STAGE_BF16_DISPATCH",
        "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_LOWRANK_1",
        "SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_STAGE_QUANT_1",
        "SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_W4A8_GEMM_1",
        "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_LOWRANK_2",
        "SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_STAGE_QUANT_2",
        "SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_W4A8_GEMM_2",
        "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE",
    ):
        assert boundary in process_source

    assert "gateUpSvdqL2" not in contract
    assert "gate_up_svdq_l2" not in contract


def test_svdq_kernel_contract_manifest_documents_workspace_sync_and_stage_map(tmp_path):
    from tools.svdq_kernel_contract_manifest import build_manifest

    manifest = build_manifest(REPO_ROOT)
    output = tmp_path / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["operator"] == "DispatchFFNCombineW4A8SVDQ"
    assert loaded["source_files"]["op_cmake"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/CMakeLists.txt"
    )
    assert loaded["source_files"]["debug_op_def"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/svdq_low_rank_debug_readback_def.cpp"
    )
    assert loaded["source_files"]["debug_op_api_header"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/op_api/aclnn_svdq_lowrank_debug_readback.h"
    )
    assert loaded["source_files"]["debug_op_api_wrapper"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_host/op_api/aclnn_svdq_lowrank_debug_readback.cpp"
    )
    assert loaded["source_files"]["lowrank_debug_tiling"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback_tiling.h"
    )
    assert loaded["source_files"]["lowrank_debug_header"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback.h"
    )
    assert loaded["source_files"]["lowrank_debug_kernel"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8_svdq/op_kernel/lowrank/svdq_lowrank_debug_readback.cpp"
    )
    assert loaded["source_files"]["lowrank_debug_alias_cmake"] == (
        "csrc/mc2/svdq_low_rank_debug_readback/op_host/CMakeLists.txt"
    )
    assert loaded["source_files"]["lowrank_debug_alias_kernel"] == (
        "csrc/mc2/svdq_low_rank_debug_readback/svdq_low_rank_debug_readback.cpp"
    )
    assert loaded["source_files"]["lowrank_debug_torch_adapter"] == (
        "csrc/mc2/svdq_low_rank_debug_readback/svdq_low_rank_debug_readback_torch_adpt.h"
    )
    assert loaded["source_files"]["torch_binding"] == "csrc/torch_binding.cpp"
    assert loaded["source_files"]["torch_binding_meta"] == "csrc/torch_binding_meta.cpp"
    assert loaded["source_files"]["lowrank_debug_probe"] == "tools/svdq_lowrank_debug_readback_probe.py"
    assert loaded["source_files"]["lowrank_debug_install_validate"] == (
        "tools/svdq_lowrank_debug_install_validate.py"
    )
    assert loaded["source_files"]["build_aclnn"] == "csrc/build_aclnn.sh"
    assert loaded["source_files"]["official_w4a8_cmake"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/CMakeLists.txt"
    )
    assert loaded["source_files"]["official_w4a8_debug_def"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/svdqw4_a8_debug_readback_def.cpp"
    )
    assert loaded["source_files"]["official_w4a8_debug_api_header"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_debug_readback.h"
    )
    assert loaded["source_files"]["official_w4a8_debug_api_wrapper"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_debug_readback.cpp"
    )
    assert loaded["source_files"]["official_w4a8_host_tiling"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_host/dispatch_ffn_combine_w4_a8_tiling.cpp"
    )
    assert loaded["source_files"]["official_w4a8_kernel"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp"
    )
    assert loaded["source_files"]["official_w4a8_kernel_entry"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.cpp"
    )
    assert loaded["source_files"]["official_w4a8_debug_kernel_entry"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/svdqw4_a8_debug_readback.cpp"
    )
    assert loaded["source_files"]["w4a8_debug_torch_adapter"] == (
        "csrc/mc2/svdq_w4a8_debug_readback/svdq_w4a8_debug_readback_torch_adpt.h"
    )
    assert loaded["source_files"]["w4a8_debug_probe"] == "tools/svdq_w4a8_debug_readback_probe.py"
    assert loaded["source_files"]["official_w4a8_op"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h"
    )
    assert loaded["source_files"]["official_w4a8_gmm1_epilogue"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
        "block_epilogue_w4a8post_pertoken_swiglu.hpp"
    )
    assert loaded["source_files"]["official_w4a8_gmm2_epilogue"] == (
        "csrc/mc2/dispatch_ffn_combine_w4_a8/op_kernel/utils/"
        "block_epilogue_w4a8post_pertoken_v2.hpp"
    )
    assert loaded["counts"] == {
        "factor_abi": 5,
        "workspace_regions": 16,
        "sync_flags": 14,
        "bf16_stages": 7,
        "residual_stages": 4,
        "residual_quant_launches": 2,
        "residual_gmm_launches": 2,
        "mixed_epilogue_launches": 2,
        "final_combine_launches": 1,
        "lowrank_invocations": 2,
    }
    assert [factor["operator_tensor"] for factor in loaded["factor_abi"]] == [
        "gate_up_svdq_l1",
        "gate_svdq_l2",
        "up_svdq_l2",
        "down_svdq_l1",
        "down_svdq_l2",
    ]
    assert "gate_up_svdq_l2" not in json.dumps(loaded)
    assert loaded["rank_split_contract"]["split_source"] == "explicit gateRank/upRank offsets"
    assert loaded["rank_split_contract"]["up_rank_offset"] == "gateRank"
    assert loaded["production_fail_closed"]["host_tiling_returns_graph_failed"]
    assert not loaded["production_fail_closed"]["host_tiling_success_enabled"]
    assert loaded["production_fail_closed"]["sync_handoff_source_enabled"]
    assert loaded["production_fail_closed"]["lowrank_is_implemented_uses_complete_contract"]
    assert loaded["production_fail_closed"]["dispatch_routing_execution_enabled"]
    assert loaded["production_fail_closed"]["residual_routed_input_quant_execution_enabled"]
    assert loaded["production_fail_closed"]["residual_hidden_quant_execution_enabled"]
    assert loaded["production_fail_closed"]["residual_hidden_quant_scalar_helpers_absent"]
    assert loaded["production_fail_closed"]["residual_quant_launch_descriptor_recorded"]
    assert loaded["production_fail_closed"]["residual_gmm_launch_descriptor_recorded"]
    assert not loaded["production_fail_closed"]["residual_gmm_execution_enabled"]
    assert loaded["production_fail_closed"]["residual_gmm_scalar_helpers_absent"]
    assert loaded["production_fail_closed"]["mixed_epilogue_launch_descriptor_recorded"]
    assert loaded["production_fail_closed"]["mixed_output_epilogue_execution_enabled"]
    assert loaded["production_fail_closed"]["mixed_swiglu_epilogue_execution_enabled"]
    assert loaded["production_fail_closed"]["mixed_epilogue_scalar_helpers_absent"]
    assert loaded["production_fail_closed"]["final_combine_launch_descriptor_recorded"]
    assert loaded["production_fail_closed"]["final_combine_execution_enabled"]
    assert loaded["production_fail_closed"]["final_combine_scalar_helpers_absent"]
    assert loaded["production_fail_closed"]["w4a8_residual_execution_fail_closed"]
    assert loaded["production_fail_closed"]["mixed_epilogue_execution_fail_closed"]
    assert loaded["production_fail_closed"]["final_combine_execution_fail_closed"]
    assert loaded["production_fail_closed"]["w4a8_residual_contract_recorded"]
    assert loaded["production_fail_closed"]["mixed_epilogue_contract_recorded"]
    assert loaded["production_fail_closed"]["final_combine_contract_recorded"]
    assert loaded["production_admission"]["host_tiling_must_remain_fail_closed"]
    assert not loaded["production_admission"]["production_enable_allowed"]
    assert "stage2_2_official_gmm2_gate" in loaded["production_admission"]
    assert loaded["production_admission"]["stage2_2_official_gmm2_gate_passed"]
    assert loaded["production_admission"]["stage2_2_official_gmm2_gate"]["status"] == "passed"
    assert not loaded["production_admission"]["stage2_2_official_gmm2_gate"]["stage2_3_and_later_blocked"]
    assert "stage2_3_real_checkpoint_composition_gate" in loaded["production_admission"]
    assert loaded["production_admission"]["stage2_3_isolated_gate_passed"]
    assert loaded["production_admission"]["remaining_execution_requirements"] == {
        "dispatch_routing_execution_enabled": True,
        "residual_hidden_quant_execution_enabled": True,
        "residual_w4a8_gmm_execution_enabled": False,
        "mixed_swiglu_epilogue_execution_enabled": True,
        "mixed_output_epilogue_execution_enabled": True,
        "final_combine_execution_enabled": True,
        "four_npu_target_model_e2e_validated": False,
    }
    assert loaded["source_proof"]["kernel_resolves_rank_workspace_regions"]
    assert loaded["source_proof"]["kernel_records_dispatch_routing_contract"]
    assert loaded["source_proof"]["host_tiling_builds_dispatch_routing_subtiling"]
    assert loaded["source_proof"]["kernel_tiling_contains_dispatch_routing_subtiling"]
    assert loaded["source_proof"]["kernel_dispatch_routing_uses_official_tiling_contract"]
    assert loaded["source_proof"]["kernel_dispatch_routing_calls_official_bf16_helper"]
    assert loaded["source_proof"]["kernel_dispatch_routing_execution_enabled"]
    assert loaded["source_proof"]["kernel_process_orders_svdq_data_dependencies"]
    assert loaded["source_proof"]["kernel_validates_complete_sync_flag_table"]
    assert loaded["source_proof"]["kernel_synchronizes_stage_boundaries"]
    assert loaded["source_proof"]["kernel_binds_residual_weight_scale_slots"]
    assert loaded["source_proof"]["kernel_residual_dispatches_dynamic_quant_and_gmm"]
    assert loaded["source_proof"]["kernel_residual_quant_launch_descriptor_recorded"]
    assert loaded["source_proof"]["kernel_residual_gmm_launch_descriptor_recorded"]
    assert not loaded["source_proof"]["kernel_residual_gmm_scalar_execution_enabled"]
    assert loaded["source_proof"]["kernel_residual_gmm_scalar_helpers_absent"]
    assert loaded["source_proof"]["kernel_residual_routed_input_quant_execution_enabled"]
    assert loaded["source_proof"]["kernel_residual_hidden_quant_aiv_execution_enabled"]
    assert not loaded["source_proof"]["kernel_residual_hidden_quant_scalar_execution_enabled"]
    assert loaded["source_proof"]["kernel_residual_hidden_quant_scalar_helpers_absent"]
    assert loaded["source_proof"]["kernel_residual_execution_dispatch_enabled"]
    assert loaded["source_proof"]["kernel_records_mixed_epilogue_contracts"]
    assert loaded["source_proof"]["kernel_mixed_epilogue_launch_descriptor_recorded"]
    assert loaded["source_proof"]["kernel_mixed_output_epilogue_aiv_execution_enabled"]
    assert loaded["source_proof"]["kernel_mixed_swiglu_epilogue_aiv_execution_enabled"]
    assert not loaded["source_proof"]["kernel_mixed_output_epilogue_scalar_execution_enabled"]
    assert not loaded["source_proof"]["kernel_mixed_swiglu_epilogue_scalar_execution_enabled"]
    assert loaded["source_proof"]["kernel_mixed_epilogue_scalar_helpers_absent"]
    assert loaded["source_proof"]["kernel_records_final_combine_contract"]
    assert loaded["source_proof"]["kernel_final_combine_launch_descriptor_recorded"]
    assert loaded["source_proof"]["kernel_final_combine_official_unpermute_execution_enabled"]
    assert not loaded["source_proof"]["kernel_final_combine_scalar_execution_enabled"]
    assert loaded["source_proof"]["kernel_final_combine_scalar_helpers_absent"]
    assert loaded["source_proof"]["kernel_mixed_final_execution_dispatch_enabled"]
    assert loaded["source_proof"]["lowrank_helper_enabled_by_contract"]
    assert loaded["source_proof"]["lowrank_helper_uses_separate_rank_workspace"]
    assert loaded["source_proof"]["lowrank_helper_stage_orders_rank_consumers"]
    assert [region["name"] for region in loaded["workspace_regions"][-2:]] == [
        "SVDQ_REGION_LOWRANK_RANK_1",
        "SVDQ_REGION_LOWRANK_RANK_2",
    ]
    assert [stage["name"] for stage in loaded["residual_stages"]] == [
        "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
    ]
    assert all(stage["residual_only"] for stage in loaded["residual_stages"])
    assert [launch["name"] for launch in loaded["residual_quant_launches"]] == [
        "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
    ]
    assert loaded["residual_quant_launches"][0] == {
        "id": 0,
        "name": "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "input_region": "SVDQ_REGION_ROUTED_X",
        "activation_scale_region": "SVDQ_REGION_X_SCALE",
        "output_region": "SVDQ_REGION_X_Q",
        "m": "routedRows",
        "k": "info.hiddenSize",
        "scale_elements": "routedRows",
        "uses_routing": True,
        "residual_only": True,
    }
    assert loaded["residual_quant_launches"][1] == {
        "id": 1,
        "name": "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "input_region": "SVDQ_REGION_HIDDEN",
        "activation_scale_region": "SVDQ_REGION_HIDDEN_SCALE",
        "output_region": "SVDQ_REGION_HIDDEN_Q",
        "m": "routedRows",
        "k": "info.intermediateSize",
        "scale_elements": "routedRows",
        "uses_routing": False,
        "residual_only": True,
    }
    assert [launch["name"] for launch in loaded["residual_gmm_launches"]] == [
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
    ]
    assert loaded["residual_gmm_launches"][0] == {
        "id": 0,
        "name": "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "input_region": "SVDQ_REGION_X_Q",
        "activation_scale_region": "SVDQ_REGION_X_SCALE",
        "output_region": "SVDQ_REGION_ACCUMULATOR_1",
        "m": "routedRows",
        "k": "info.hiddenSize",
        "n": "info.intermediateSize * 2",
        "residual_weight_slot": "weight1",
        "residual_scale_slot": "scale1",
        "residual_bias_slot": "bias1",
        "list_len": "info.expertPerRank",
        "group_list_type": 1,
        "group_type": 0,
        "split_item": 2,
        "trans_b": False,
        "weight_nz": True,
        "residual_only": True,
    }
    assert loaded["residual_gmm_launches"][1]["input_region"] == "SVDQ_REGION_HIDDEN_Q"
    assert loaded["residual_gmm_launches"][1]["activation_scale_region"] == "SVDQ_REGION_HIDDEN_SCALE"
    assert loaded["residual_gmm_launches"][1]["output_region"] == "SVDQ_REGION_ACCUMULATOR_2"
    assert loaded["residual_gmm_launches"][1]["residual_weight_slot"] == "weight2"
    assert loaded["residual_gmm_launches"][1]["residual_scale_slot"] == "scale2"
    assert loaded["residual_gmm_launches"][1]["residual_bias_slot"] == "bias2"
    assert all(launch["residual_only"] for launch in loaded["residual_gmm_launches"])
    assert loaded["mixed_epilogue_launches"][0] == {
        "id": 0,
        "name": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "residual_region": "SVDQ_REGION_ACCUMULATOR_1",
        "lowrank_region": "SVDQ_REGION_PROJECTION_1",
        "scale_region": "SVDQ_REGION_X_SCALE",
        "output_region": "SVDQ_REGION_HIDDEN",
        "m": "routedRows",
        "residual_columns": "info.intermediateSize * 2",
        "lowrank_columns": "info.intermediateSize * 2",
        "output_columns": "info.intermediateSize",
        "gate_column_offset": 0,
        "up_column_offset": "info.intermediateSize",
        "applies_swiglu": True,
    }
    assert loaded["mixed_epilogue_launches"][1] == {
        "id": 1,
        "name": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "residual_region": "SVDQ_REGION_ACCUMULATOR_2",
        "lowrank_region": "SVDQ_REGION_PROJECTION_2",
        "scale_region": "SVDQ_REGION_HIDDEN_SCALE",
        "output_region": "SVDQ_REGION_PEER_OUTPUT",
        "m": "routedRows",
        "residual_columns": "info.hiddenSize",
        "lowrank_columns": "info.hiddenSize",
        "output_columns": "info.hiddenSize",
        "gate_column_offset": "SVDQ_INVALID_ID",
        "up_column_offset": "SVDQ_INVALID_ID",
        "applies_swiglu": False,
    }
    assert loaded["final_combine_launch"] == {
        "name": "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "input_region": "SVDQ_REGION_PEER_OUTPUT",
        "route_region": "SVDQ_REGION_EXPANDED_ROW_IDX",
        "m": "info.m",
        "routed_rows": "info.maxOutputSize",
        "hidden_size": "info.hiddenSize",
        "top_k": "info.topK",
        "active_slots": "info.m * info.topK",
        "uses_expert_idx": True,
        "uses_probs": True,
    }
    expected_false_source_proofs = {
        "host_tiling_graph_success_enabled",
        "kernel_residual_gmm_scalar_execution_enabled",
        "kernel_residual_hidden_quant_scalar_execution_enabled",
        "kernel_mixed_output_epilogue_scalar_execution_enabled",
        "kernel_mixed_swiglu_epilogue_scalar_execution_enabled",
        "kernel_final_combine_scalar_execution_enabled",
    }
    assert {
        name for name, passed in loaded["source_proof"].items() if not passed
    } == expected_false_source_proofs
    assert loaded["debug_readback_contract"] == {
        "compile_macro": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
        "cmake_option": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
        "default_enabled": True,
        "production_abi_changed": False,
        "readback_region": "lowRankAccumulator region selected by invocation.accumulatorRegionId",
        "readback_dtype": "FP32",
        "readback_source": "L0C accumulator after each MMAD K tile",
        "final_tile_semantics": "final full-K FP32 accumulator is mirrored before BF16 output conversion",
        "partial_tile_semantics": "non-final K-tile partial sums are mirrored for host-readable debug validation",
        "source_proof": [
            "op_cmake_has_local_debug_readback_option",
            "op_cmake_debug_readback_defaults_on",
            "op_cmake_scopes_debug_readback_to_lowrank_debug_op",
            "lowrank_mmad_debug_readback_macro",
            "lowrank_mmad_debug_readback_uses_fp32_l0c_to_gm",
            "lowrank_mmad_debug_readback_targets_accumulator_gm",
            "lowrank_debug_runner_exists",
            "lowrank_debug_runner_macro_gated",
            "lowrank_debug_runner_bypasses_production_is_implemented_gate",
        ],
    }
    assert loaded["debug_launch_contract"] == {
        "op_name": "SVDQLowRankDebugReadback",
        "aclnn_get_workspace": "aclnnSVDQLowRankDebugReadbackGetWorkspaceSize",
        "aclnn_launch": "aclnnSVDQLowRankDebugReadback",
        "kernel_symbol": "svdq_low_rank_debug_readback",
        "production_abi_changed": False,
        "input_tensors": [
            "routed_x",
            "hidden",
            "gate_up_svdq_l1",
            "gate_svdq_l2",
            "up_svdq_l2",
            "down_svdq_l1",
            "down_svdq_l2",
            "expert_token_nums",
        ],
        "readback_tensors": [
            "gate_up_output_bf16",
            "down_output_bf16",
            "gate_up_accumulator_fp32",
            "down_accumulator_fp32",
        ],
        "tiling_contract": "SVDQLowRankDebugTilingData with gate/up and down SVDQFusedDownUpTiling records",
        "source_proof": [
            "lowrank_debug_kernel_exists",
            "lowrank_debug_kernel_uses_debug_runner",
            "lowrank_debug_kernel_exposes_readback_buffers",
            "lowrank_debug_kernel_uses_separate_tiling_contract",
            "lowrank_debug_kernel_preserves_branch_separation",
            "lowrank_debug_op_has_compile_options",
            "lowrank_debug_op_inner_aclnn_linked",
            "lowrank_debug_op_registered",
            "lowrank_debug_op_tiling_registered",
            "lowrank_debug_op_public_aclnn_wrapper",
            "lowrank_debug_op_exposes_same_readback_outputs",
            "lowrank_debug_alias_source_root",
            "lowrank_debug_torch_adapter_registered",
            "lowrank_debug_meta_registered",
            "lowrank_debug_probe_launches_real_op",
            "lowrank_debug_probe_preflights_runtime_soc_package",
            "lowrank_debug_install_validator_checks_schema_symbols_and_soc",
            "svdq_ops_in_a2_a3_aclnn_package",
        ],
    }
    assert loaded["w4a8_debug_readback_contract"] == {
        "cann_operator_surface_wired": True,
        "launch_operator_wired": True,
        "acceptance_gate_claimed": False,
        "public_grouped_matmul_allowed": False,
        "required_compile_option": "SVDQ_W4A8_DEBUG_READBACK",
        "official_compile_macro": "W4A8_DEBUG",
        "default_enabled": False,
        "production_abi_changed": False,
        "official_kernel": "DispatchFFNCombineW4A8",
        "official_kernel_symbol": "dispatch_ffn_combine_w4_a8",
        "kernel_type": "KERNEL_TYPE_MIX_AIC_1_2",
        "aic_entry": "operator()<AscendC::AIC> -> GMM1(params); GMM2(params);",
        "aiv_entry": "operator()<AscendC::AIV> -> DispatchAndCombine(params);",
        "gmm1_readback": {
            "workspace_ptr": "ptrCGMM1",
            "source": "block_epilogue_w4a8post_pertoken_swiglu.hpp",
            "dtype": "FP32",
            "semantic_point": "post-dequant pre-SwiGLU GMM1",
            "copy_token": "copyUbToGmGMM1(gmTileGMM1, ubCFp32, layoutGMM1, layoutGMM1);",
        },
        "gmm1_hidden_prequant_readback": {
            "workspace_ptr": "ptrCGMM1Hidden",
            "source": "block_epilogue_w4a8post_pertoken_swiglu.hpp",
            "dtype": "FP32",
            "semantic_point": "post-SwiGLU pre-hidden-quant GMM1",
            "copy_token": (
                "copyUbToGmGMM1Hidden(gmTileGMM1Hidden, ubCFp32ChunkN, "
                "layoutGMM1Hidden, layoutGMM1Hidden);"
            ),
        },
        "gmm2_readback": {
            "workspace_ptr": "ptrCGMM2",
            "source": "block_epilogue_w4a8post_pertoken_v2.hpp",
            "dtype": "FP32",
            "semantic_point": "post-dequant GMM2 before BF16/output copy",
            "copy_token": "copyUbToGmGMM2(gmTileGMM2, ubFp32, layoutGM, layoutUB);",
        },
        "debug_op_abi": {
            "op_name": "SVDQW4A8DebugReadback",
            "aclnn_get_workspace": "aclnnSVDQW4A8DebugReadbackGetWorkspaceSize",
            "aclnn_launch": "aclnnSVDQW4A8DebugReadback",
            "kernel_symbol": "svdqw4_a8_debug_readback",
            "readback_tensors": [
                "routed_x_int8",
                "routed_x_scale_fp32",
                "gmm1_post_dequant_fp32",
                "gmm1_hidden_prequant_fp32",
                "hidden_x_int4_packed",
                "hidden_x_scale_fp32",
                "gmm2_post_dequant_fp32",
            ],
            "input_surface": "official DispatchFFNCombineW4A8 inputs plus readback outputs",
            "debug_output_pointer_hook": (
                "MatmulKernel::Params ptrDebugRoutedX/ptrDebugRoutedScale/"
                "ptrDebugGMM1/ptrDebugGMM1Hidden/ptrDebugHiddenX/"
                "ptrDebugHiddenScale/ptrDebugGMM2"
            ),
            "must_reuse": [
                "DispatchFFNCombineW4A8Kernel",
                "BlockMmad",
                "EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant",
                "EpilogueAtlasA2W4A8PostPerTokenDequantV2",
            ],
        },
        "source_proof": [
            "official_w4a8_debug_option_default_off",
            "official_w4a8_debug_macro_scoped",
            "official_w4a8_mixed_aic_aiv_kernel",
            "official_w4a8_aic_calls_gmm1_gmm2",
            "official_w4a8_aiv_calls_dispatch_and_combine",
            "official_w4a8_workspace_has_ptr_cgmm1_cgmm2",
            "official_w4a8_debug_output_pointer_hook",
            "official_w4a8_debug_op_surface_wired",
            "official_w4a8_debug_op_aclnn_wrapper",
            "official_w4a8_debug_kernel_reuses_official_path",
            "official_w4a8_debug_torch_adapter_registered",
            "official_w4a8_debug_probe_launches_real_op",
            "official_w4a8_kernel_binds_block_mmad_and_epilogues",
            "official_w4a8_gmm1_epilogue_debug_copies_fp32",
            "official_w4a8_gmm2_epilogue_debug_copies_fp32",
        ],
    }

    stage_names = [stage["name"] for stage in loaded["bf16_stages"]]
    assert stage_names == [
        "SVDQ_BF16_STAGE_ROUTING",
        "SVDQ_BF16_STAGE_GATE_UP_L1_GEMM",
        "SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT",
        "SVDQ_BF16_STAGE_GATE_L2_GEMM",
        "SVDQ_BF16_STAGE_UP_L2_GEMM",
        "SVDQ_BF16_STAGE_DOWN_L1_GEMM",
        "SVDQ_BF16_STAGE_DOWN_L2_GEMM",
    ]

    gate_up_invocation, down_invocation = loaded["lowrank_invocations"]
    assert gate_up_invocation["second_up_factor"] == "SVDQ_FACTOR_UP_L2"
    assert gate_up_invocation["second_input_column_offset"] == "info.upRankOffset"
    assert gate_up_invocation["second_output_column_offset"] == "info.intermediateSize"
    assert down_invocation["second_up_factor"] == "SVDQ_INVALID_ID"


def test_svdq_kernel_contract_manifest_keeps_stage2_3_evidence_blocked_by_gmm2_gate(tmp_path):
    from tools.svdq_kernel_contract_manifest import build_manifest

    evidence_path = tmp_path / "stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json"
    evidence_path.parent.mkdir(parents=True)
    zero_error = {
        "actual_finite": True,
        "expected_finite": True,
        "diff_finite": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
    }
    evidence_path.write_text(
        json.dumps(
            {
                "passed": True,
                "stage": {
                    "passed": True,
                    "public_grouped_matmul_used": False,
                    "production_svdq_host_tiling_fail_closed": True,
                    "shape": {
                        "num_tokens": 4,
                        "top_k": 8,
                        "active_rows": 32,
                        "hidden_size": 2048,
                        "intermediate_size": 512,
                    },
                    "rank_metadata": {
                        "gate_rank": 64,
                        "up_rank": 64,
                        "down_rank": 64,
                        "gate_rank_offset": 0,
                        "up_rank_offset": 64,
                    },
                    "stage_passed": {
                        "first_mixed_epilogue": True,
                        "official_gmm2_from_svdq_hidden": True,
                        "gate_mixed": True,
                        "up_mixed": True,
                        "hidden_bf16": True,
                        "hidden_scale": True,
                        "hidden_q": True,
                        "down_mixed": True,
                        "out_bf16": True,
                        "final_combine_output": True,
                        "same_routing_identity": True,
                    },
                    "stage2_3_same_routing_manifest": {
                        "checks": {
                            "expert_token_total_matches_active_rows": True,
                            "same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant": True,
                            "official_gmm2_output_feeds_final_mixed_residual_down": True,
                            "final_mixed_output_is_final_combine_input": True,
                            "final_combine_consumes_mixed_down_peer_output": True,
                            "final_combine_output_validated": True,
                            "same_source_token_payload_across_topk_slots_proven": True,
                        }
                    },
                    "official_gmm2_from_svdq_hidden": {
                        "checks": {
                            "official_gmm2_entry_reached": True,
                            "gate_a_input_boundary_passed": True,
                            "official_gmm2_post_dequant_finite": True,
                            "official_gmm2_post_dequant_nonzero": True,
                            "official_gmm2_post_dequant_reference_passed": True,
                            "official_gmm2_numerical_gate_passed": True,
                        },
                        "unfused_reference": {"error": zero_error},
                    },
                    "stage_errors": {
                        "down_mixed": zero_error,
                        "out_bf16": zero_error,
                        "final_combine_output": zero_error,
                    },
                    "real_final_combine": {
                        "official_surface": "torch_npu.npu_moe_token_unpermute",
                        "reference": "build_svdq_final_combine_reference",
                        "index_semantics": "official_token_major_output_slots_to_permuted_input_rows",
                        "passed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(REPO_ROOT, evidence_dir=tmp_path)
    gate = manifest["production_admission"]["stage2_3_real_checkpoint_composition_gate"]

    assert gate["evidence_found"]
    assert not gate["passed"]
    assert gate["historical_passed_under_superseded_contract"]
    assert gate["status"] == "blocked_by_stage2_2_official_gmm2_gate"
    assert gate["blocking_gate"] == "stage2_2_modified_hidden_official_w4a8_gmm2"
    assert not manifest["production_admission"]["stage2_2_official_gmm2_gate_passed"]
    assert not manifest["production_admission"]["stage2_3_isolated_gate_passed"]
    assert not manifest["production_admission"]["production_enable_allowed"]
    assert manifest["production_admission"]["host_tiling_must_remain_fail_closed"]


def test_svdq_kernel_contract_manifest_accepts_current_stage2_2_recheck_and_stage2_3_evidence(tmp_path):
    from tools.svdq_kernel_contract_manifest import build_manifest

    zero_error = {
        "actual_finite": True,
        "expected_finite": True,
        "diff_finite": True,
        "max_abs": 0.0,
        "mean_abs": 0.0,
    }
    stage2_2_path = tmp_path / "stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json"
    stage2_2_path.parent.mkdir(parents=True)
    stage2_2_path.write_text(
        json.dumps(
            {
                "passed": True,
                "skipped": False,
                "stage": {
                    "passed": True,
                    "real_checkpoint_validation": True,
                    "public_grouped_matmul_used": False,
                    "production_svdq_host_tiling_fail_closed": True,
                    "shape": {
                        "top_k": 1,
                        "active_rows": 16,
                        "hidden_size": 2048,
                        "intermediate_size": 512,
                    },
                    "routing_identity": {
                        "expert_token_total_matches_active_rows": True,
                        "reference_group_counts_match_expert_token_nums": True,
                        "manifest_scope": {
                            "same_source_token_payload_across_topk_slots_proven": True,
                        },
                        "tp_ep_mapping": {
                            "local_expert_id_equals_global_expert_id": True,
                        },
                    },
                    "official_lifecycle_debug_contract": {
                        "mode": "official DispatchAndCombine lifecycle with external hidden/scale overlay",
                        "gmm2_only_from_packed": False,
                        "forbidden_paths": {
                            "custom_v2c_or_c2v_signal_substitute_used": False,
                        },
                    },
                    "checks": {
                        "official_gmm2_entry_reached": True,
                        "official_gmm2_loop_count": 8,
                        "official_gmm2_active_tile_count": 8,
                        "official_gmm2_loop_stats_valid": True,
                        "official_gmm2_active_tile_count_nonzero": True,
                        "official_gmm2_aic_raw_output_finite": True,
                        "official_gmm2_aic_raw_output_nonzero": True,
                        "official_gmm2_aic_reference_passed": True,
                        "official_gmm2_accumulator_int32_reference_passed": True,
                        "official_gmm2_c2v_handoff_verified": True,
                        "official_gmm2_post_dequant_finite": True,
                        "official_gmm2_post_dequant_nonzero": True,
                        "official_gmm2_post_dequant_reference_passed": True,
                        "official_gmm2_numerical_gate_passed": True,
                        "gate_a_input_boundary_passed": True,
                        "hidden_packed_exact": True,
                        "hidden_packed_mismatch_count_zero": True,
                        "hidden_post_override_readback_exact": True,
                        "hidden_scale_post_override_readback_exact": True,
                        "hidden_scale_finite": True,
                        "hidden_scale_nonzero": True,
                        "canonical_hidden_finite": True,
                        "canonical_hidden_nonzero": True,
                    },
                    "gmm2": {
                        "int32_accumulator_readback_reference": {
                            "passed": True,
                            "actual_nonzero": True,
                        },
                        "actual_d2_post_dequant_reconstruction": {
                            "passed_with_gate_tolerance": True,
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    stage2_3_path = tmp_path / "stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json"
    stage2_3_path.parent.mkdir(parents=True, exist_ok=True)
    stage2_3_path.write_text(
        json.dumps(
            {
                "passed": True,
                "stage": {
                    "passed": True,
                    "public_grouped_matmul_used": False,
                    "production_svdq_host_tiling_fail_closed": True,
                    "shape": {
                        "num_tokens": 4,
                        "top_k": 8,
                        "active_rows": 32,
                        "hidden_size": 2048,
                        "intermediate_size": 512,
                    },
                    "rank_metadata": {
                        "gate_rank": 64,
                        "up_rank": 64,
                        "down_rank": 64,
                        "gate_rank_offset": 0,
                        "up_rank_offset": 64,
                    },
                    "stage_passed": {
                        "first_mixed_epilogue": True,
                        "official_gmm2_from_svdq_hidden": True,
                        "gate_mixed": True,
                        "up_mixed": True,
                        "hidden_bf16": True,
                        "hidden_scale": True,
                        "hidden_q": True,
                        "down_mixed": True,
                        "out_bf16": True,
                        "final_combine_output": True,
                        "same_routing_identity": True,
                    },
                    "stage2_3_same_routing_manifest": {
                        "checks": {
                            "expert_token_total_matches_active_rows": True,
                            "same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant": True,
                            "official_gmm2_output_feeds_final_mixed_residual_down": True,
                            "final_mixed_output_is_final_combine_input": True,
                            "final_combine_consumes_mixed_down_peer_output": True,
                            "final_combine_output_validated": True,
                            "same_source_token_payload_across_topk_slots_proven": True,
                        }
                    },
                    "official_gmm2_from_svdq_hidden": {
                        "checks": {
                            "official_gmm2_entry_reached": True,
                            "gate_a_input_boundary_passed": True,
                            "official_gmm2_post_dequant_finite": True,
                            "official_gmm2_post_dequant_nonzero": True,
                            "official_gmm2_post_dequant_reference_passed": True,
                            "official_gmm2_numerical_gate_passed": True,
                        },
                        "unfused_reference": {"error": zero_error},
                    },
                    "stage_errors": {
                        "down_mixed": zero_error,
                        "out_bf16": zero_error,
                        "final_combine_output": zero_error,
                    },
                    "real_final_combine": {
                        "official_surface": "torch_npu.npu_moe_token_unpermute",
                        "reference": "build_svdq_final_combine_reference",
                        "index_semantics": "official_token_major_output_slots_to_permuted_input_rows",
                        "passed": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(REPO_ROOT, evidence_dir=tmp_path)
    stage2_2 = manifest["production_admission"]["stage2_2_official_gmm2_gate"]
    stage2_3 = manifest["production_admission"]["stage2_3_real_checkpoint_composition_gate"]

    assert stage2_2["status"] == "passed"
    assert stage2_2["evidence_status"] == "current_recheck_passed"
    assert not stage2_2["historical_passed_under_superseded_contract"]
    assert stage2_2["passed"]
    assert not stage2_2["stage2_3_and_later_blocked"]
    assert stage2_3["status"] == "passed"
    assert stage2_3["passed"]
    assert stage2_3["blocking_gate"] is None
    assert manifest["production_admission"]["stage2_2_official_gmm2_gate_passed"]
    assert manifest["production_admission"]["stage2_3_isolated_gate_passed"]
    assert not manifest["production_admission"]["production_enable_allowed"]
    assert manifest["production_admission"]["host_tiling_must_remain_fail_closed"]


def test_svdq_bf16_routing_stage_probe_cpu_golden_uses_official_count_layout():
    from tools.svdq_bf16_routing_stage_probe import _cpu_routing_golden

    x = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    expert_idx = torch.tensor([[0], [1], [0], [1], [0], [1]], dtype=torch.int32)

    golden = _cpu_routing_golden(x, expert_idx, expert_num=2, active_expert_range=(0, 2))

    assert golden["expanded_x"].tolist() == [
        [0.0, 1.0, 2.0, 3.0],
        [8.0, 9.0, 10.0, 11.0],
        [16.0, 17.0, 18.0, 19.0],
        [4.0, 5.0, 6.0, 7.0],
        [12.0, 13.0, 14.0, 15.0],
        [20.0, 21.0, 22.0, 23.0],
    ]
    assert golden["expanded_row_idx"].tolist() == [0, 2, 4, 1, 3, 5]
    assert golden["expert_tokens"].tolist() == [3, 3]
