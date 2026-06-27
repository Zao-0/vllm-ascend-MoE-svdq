#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Emit the W4A8-SVDQ kernel workspace, sync, and BF16-stage contract manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_DIR = Path("/root/workspace/lza/svdq_clean_evidence")
STAGE2_2_GMM2_OFFICIAL_EVIDENCE = Path("stage2/phase_stage2_gmm2_current_recheck_top1_expert0.json")
STAGE2_2_GMM2_HISTORICAL_EVIDENCE = Path("phase_stage2_gmm2_trunc13_reference_true_top1_expert0.json")
STAGE2_3_REAL_COMPOSITION_EVIDENCE = Path(
    "stage2/phase_stage2_real_composition_topk8_finalcombine_fixedidx_experts0_7.json"
)
STAGE2_2_GMM2_OFFICIAL_PATH_APPENDIX = (
    "/root/workspace/lza/svdq_qwen35_moe_clean_implementation_stage2_appendix_gmm2_official_path.md"
)
STAGE2_2_GMM2_OFFICIAL_PATH_RESET_REVISION = (
    "stage2_appendix_gmm2_official_path_mandatory_official_path_correction_20260627"
)
STAGE2_2_GMM2_OFFICIAL_PATH_CURRENT_REQUIREMENTS_REVISION = (
    "stage2_appendix_gmm2_official_path_current_authoritative_fail_in_progress_20260627"
)

OP_ROOT = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq")
OP_CMAKE = OP_ROOT / "op_host/CMakeLists.txt"
OP_DEF = OP_ROOT / "op_host/dispatch_ffn_combine_w4_a8_svdq_def.cpp"
DEBUG_OP_DEF = OP_ROOT / "op_host/svdq_low_rank_debug_readback_def.cpp"
OP_PROTO = OP_ROOT / "op_host/dispatch_ffn_combine_w4_a8_svdq_proto.cpp"
DEBUG_OP_API_HEADER = OP_ROOT / "op_host/op_api/aclnn_svdq_lowrank_debug_readback.h"
DEBUG_OP_API_WRAPPER = OP_ROOT / "op_host/op_api/aclnn_svdq_lowrank_debug_readback.cpp"
HOST_TILING = OP_ROOT / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp"
KERNEL_CONTRACT = OP_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h"
KERNEL_TILING = OP_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h"
LOWRANK_HEADER = OP_ROOT / "op_kernel/lowrank/svdq_fused_down_up.hpp"
LOWRANK_DEBUG_TILING = OP_ROOT / "op_kernel/lowrank/svdq_lowrank_debug_readback_tiling.h"
LOWRANK_DEBUG_HEADER = OP_ROOT / "op_kernel/lowrank/svdq_lowrank_debug_readback.h"
LOWRANK_DEBUG_KERNEL = OP_ROOT / "op_kernel/lowrank/svdq_lowrank_debug_readback.cpp"
LOWRANK_DEBUG_ALIAS_ROOT = Path("csrc/mc2/svdq_low_rank_debug_readback")
LOWRANK_DEBUG_ALIAS_CMAKE = LOWRANK_DEBUG_ALIAS_ROOT / "op_host/CMakeLists.txt"
LOWRANK_DEBUG_ALIAS_KERNEL = LOWRANK_DEBUG_ALIAS_ROOT / "svdq_low_rank_debug_readback.cpp"
LOWRANK_DEBUG_TORCH_ADAPTER = LOWRANK_DEBUG_ALIAS_ROOT / "svdq_low_rank_debug_readback_torch_adpt.h"
TORCH_BINDING = Path("csrc/torch_binding.cpp")
TORCH_BINDING_META = Path("csrc/torch_binding_meta.cpp")
LOWRANK_DEBUG_PROBE = Path("tools/svdq_lowrank_debug_readback_probe.py")
LOWRANK_DEBUG_INSTALL_VALIDATE = Path("tools/svdq_lowrank_debug_install_validate.py")
W4A8_DEBUG_TORCH_ADAPTER = Path("csrc/mc2/svdq_w4a8_debug_readback/svdq_w4a8_debug_readback_torch_adpt.h")
W4A8_DEBUG_PROBE = Path("tools/svdq_w4a8_debug_readback_probe.py")
BUILD_ACLNN = Path("csrc/build_aclnn.sh")

OFFICIAL_W4A8_ROOT = Path("csrc/mc2/dispatch_ffn_combine_w4_a8")
OFFICIAL_W4A8_CMAKE = OFFICIAL_W4A8_ROOT / "op_host/CMakeLists.txt"
OFFICIAL_W4A8_DEBUG_DEF = OFFICIAL_W4A8_ROOT / "op_host/svdqw4_a8_debug_readback_def.cpp"
OFFICIAL_W4A8_DEBUG_API_HEADER = (
    OFFICIAL_W4A8_ROOT / "op_host/op_api/aclnn_svdq_w4a8_debug_readback.h"
)
OFFICIAL_W4A8_DEBUG_API_WRAPPER = (
    OFFICIAL_W4A8_ROOT / "op_host/op_api/aclnn_svdq_w4a8_debug_readback.cpp"
)
OFFICIAL_W4A8_HOST_TILING = OFFICIAL_W4A8_ROOT / "op_host/dispatch_ffn_combine_w4_a8_tiling.cpp"
OFFICIAL_W4A8_KERNEL = OFFICIAL_W4A8_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8_kernel.hpp"
OFFICIAL_W4A8_KERNEL_ENTRY = OFFICIAL_W4A8_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8.cpp"
OFFICIAL_W4A8_DEBUG_KERNEL_ENTRY = (
    OFFICIAL_W4A8_ROOT / "op_kernel/svdqw4_a8_debug_readback.cpp"
)
OFFICIAL_W4A8_OP = OFFICIAL_W4A8_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8.h"
OFFICIAL_W4A8_GMM1_EPILOGUE = (
    OFFICIAL_W4A8_ROOT / "op_kernel/utils/block_epilogue_w4a8post_pertoken_swiglu.hpp"
)
OFFICIAL_W4A8_GMM2_EPILOGUE = (
    OFFICIAL_W4A8_ROOT / "op_kernel/utils/block_epilogue_w4a8post_pertoken_v2.hpp"
)

FACTOR_ABI = [
    {"id": 0, "name": "SVDQ_FACTOR_GATE_UP_L1", "operator_tensor": "gate_up_svdq_l1"},
    {"id": 1, "name": "SVDQ_FACTOR_GATE_L2", "operator_tensor": "gate_svdq_l2"},
    {"id": 2, "name": "SVDQ_FACTOR_UP_L2", "operator_tensor": "up_svdq_l2"},
    {"id": 3, "name": "SVDQ_FACTOR_DOWN_L1", "operator_tensor": "down_svdq_l1"},
    {"id": 4, "name": "SVDQ_FACTOR_DOWN_L2", "operator_tensor": "down_svdq_l2"},
]

WORKSPACE_REGIONS = [
    {
        "id": 0,
        "name": "SVDQ_REGION_EXPANDED_ROW_IDX",
        "dtype": "SVDQ_DTYPE_INT32",
        "size_expr": "m * topK * INT32_BYTES",
        "producer_stage": "SVDQ_STAGE_BF16_DISPATCH",
        "consumer_stage": "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "lifetime_id": 1,
        "purpose": "routing metadata for final unpermute/combine",
    },
    {
        "id": 1,
        "name": "SVDQ_REGION_ROUTED_X",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * hiddenSize * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_BF16_DISPATCH",
        "consumer_stage": "SVDQ_STAGE_QUANT_1",
        "lifetime_id": 2,
        "purpose": "BF16 routed activations for W4A8 quantization and gate/up low-rank L1",
    },
    {
        "id": 2,
        "name": "SVDQ_REGION_X_Q",
        "dtype": "SVDQ_DTYPE_INT8",
        "size_expr": "maxOutputSize * hiddenSize * INT8_BYTES",
        "producer_stage": "SVDQ_STAGE_QUANT_1",
        "consumer_stage": "SVDQ_STAGE_W4A8_GEMM_1",
        "lifetime_id": 3,
        "purpose": "future quantized routed input for W4A8 GMM1",
    },
    {
        "id": 3,
        "name": "SVDQ_REGION_X_SCALE",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_QUANT_1",
        "consumer_stage": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "lifetime_id": 4,
        "purpose": "future routed input activation scale for mixed gate/up epilogue",
    },
    {
        "id": 4,
        "name": "SVDQ_REGION_PROJECTION_1",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * intermediateSize * 2 * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_1",
        "consumer_stage": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "lifetime_id": 5,
        "purpose": "gate and up BF16 low-rank outputs in separate halves",
    },
    {
        "id": 5,
        "name": "SVDQ_REGION_ACCUMULATOR_1",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * intermediateSize * 2 * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_1",
        "consumer_stage": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "lifetime_id": 6,
        "purpose": "W4A8 gate/up residual FP32 post-dequant boundary for mixed SwiGLU epilogue",
    },
    {
        "id": 6,
        "name": "SVDQ_REGION_HIDDEN",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * intermediateSize * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "consumer_stage": "SVDQ_STAGE_QUANT_2",
        "lifetime_id": 7,
        "purpose": "future SwiGLU hidden activations and down low-rank input",
    },
    {
        "id": 7,
        "name": "SVDQ_REGION_HIDDEN_Q",
        "dtype": "SVDQ_DTYPE_INT8",
        "size_expr": "maxOutputSize * intermediateSize * INT8_BYTES",
        "producer_stage": "SVDQ_STAGE_QUANT_2",
        "consumer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "lifetime_id": 8,
        "purpose": "future quantized hidden input for W4A8 GMM2",
    },
    {
        "id": 8,
        "name": "SVDQ_REGION_HIDDEN_SCALE",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_QUANT_2",
        "consumer_stage": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "lifetime_id": 9,
        "purpose": "future hidden activation scale for mixed down epilogue",
    },
    {
        "id": 9,
        "name": "SVDQ_REGION_PROJECTION_2",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * hiddenSize * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_2",
        "consumer_stage": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "lifetime_id": 10,
        "purpose": "down BF16 low-rank output",
    },
    {
        "id": 10,
        "name": "SVDQ_REGION_ACCUMULATOR_2",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * hiddenSize * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "consumer_stage": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "lifetime_id": 11,
        "purpose": "W4A8 down residual FP32 post-dequant boundary for mixed output epilogue",
    },
    {
        "id": 11,
        "name": "SVDQ_REGION_LOWRANK_ACCUMULATOR_1",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * intermediateSize * 2 * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_1",
        "consumer_stage": "SVDQ_STAGE_LOWRANK_1",
        "lifetime_id": 12,
        "purpose": "gate/up low-rank scalar fallback accumulator",
    },
    {
        "id": 12,
        "name": "SVDQ_REGION_LOWRANK_ACCUMULATOR_2",
        "dtype": "SVDQ_DTYPE_FP32",
        "size_expr": "maxOutputSize * hiddenSize * FP32_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_2",
        "consumer_stage": "SVDQ_STAGE_LOWRANK_2",
        "lifetime_id": 13,
        "purpose": "down low-rank scalar fallback accumulator",
    },
    {
        "id": 13,
        "name": "SVDQ_REGION_PEER_OUTPUT",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * hiddenSize * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "consumer_stage": "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "lifetime_id": 14,
        "purpose": "future per-peer output before final unpermute/combine",
    },
    {
        "id": 14,
        "name": "SVDQ_REGION_LOWRANK_RANK_1",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * (gateRank + upRank) * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_1",
        "consumer_stage": "SVDQ_STAGE_LOWRANK_1",
        "lifetime_id": 15,
        "purpose": "gate/up BF16 rank-state buffer between fused L1 and independent L2 stages",
    },
    {
        "id": 15,
        "name": "SVDQ_REGION_LOWRANK_RANK_2",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "maxOutputSize * downRank * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_LOWRANK_2",
        "consumer_stage": "SVDQ_STAGE_LOWRANK_2",
        "lifetime_id": 16,
        "purpose": "down BF16 rank-state buffer between down L1 and down L2 stages",
    },
    {
        "id": 16,
        "name": "SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT",
        "dtype": "SVDQ_DTYPE_BF16",
        "size_expr": "m * topK * hiddenSize * BF16_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "consumer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "lifetime_id": 17,
        "purpose": "scratch destination for ordinary official W4A8 final-combine output; never accepted as SVDQ output",
    },
    {
        "id": 17,
        "name": "SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE",
        "dtype": "SVDQ_DTYPE_INT8",
        "size_expr": "ResidualW4A8OfficialWorkspaceBytes(info)",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_1",
        "consumer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "lifetime_id": 18,
        "purpose": "private workspace for official W4A8 wrapper layout; starts at official params.ptrWorkspace",
    },
    {
        "id": 18,
        "name": "SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR",
        "dtype": "SVDQ_DTYPE_INT32",
        "size_expr": "maxOutputSize * hiddenSize * 2 * INT32_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "consumer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "lifetime_id": 19,
        "purpose": "official GMM2 pre-Fixpipe int32 accumulator tap with doubled high/low C2 rows",
    },
]

SYNC_FLAGS = [
    (0, "SVDQ_SYNC_DISPATCH_TO_QUANT_1", "SVDQ_STAGE_BF16_DISPATCH", "SVDQ_STAGE_QUANT_1", "SVDQ_REGION_ROUTED_X"),
    (1, "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1", "SVDQ_STAGE_BF16_DISPATCH", "SVDQ_STAGE_LOWRANK_1", "SVDQ_REGION_ROUTED_X"),
    (2, "SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1", "SVDQ_STAGE_QUANT_1", "SVDQ_STAGE_W4A8_GEMM_1", "SVDQ_REGION_X_Q"),
    (
        3,
        "SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_QUANT_1",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_REGION_X_SCALE",
    ),
    (
        4,
        "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_LOWRANK_1",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_REGION_PROJECTION_1",
    ),
    (
        5,
        "SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_W4A8_GEMM_1",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_REGION_ACCUMULATOR_1",
    ),
    (
        6,
        "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_QUANT_2",
        "SVDQ_REGION_HIDDEN",
    ),
    (
        7,
        "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2",
        "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "SVDQ_STAGE_LOWRANK_2",
        "SVDQ_REGION_HIDDEN",
    ),
    (8, "SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2", "SVDQ_STAGE_QUANT_2", "SVDQ_STAGE_W4A8_GEMM_2", "SVDQ_REGION_HIDDEN_Q"),
    (
        9,
        "SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_STAGE_QUANT_2",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_REGION_HIDDEN_SCALE",
    ),
    (
        10,
        "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_STAGE_LOWRANK_2",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_REGION_PROJECTION_2",
    ),
    (
        11,
        "SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_STAGE_W4A8_GEMM_2",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_REGION_ACCUMULATOR_2",
    ),
    (
        12,
        "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE",
        "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "SVDQ_REGION_PEER_OUTPUT",
    ),
    (
        13,
        "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE",
        "SVDQ_STAGE_BF16_DISPATCH",
        "SVDQ_STAGE_UNPERMUTE_COMBINE",
        "SVDQ_REGION_EXPANDED_ROW_IDX",
    ),
]

BF16_STAGES = [
    (
        0,
        "SVDQ_BF16_STAGE_ROUTING",
        "SVDQ_INVALID_ID",
        "SVDQ_INVALID_ID",
        "SVDQ_REGION_ROUTED_X",
        "routedRows",
        "info.hiddenSize",
        "info.hiddenSize",
        "0",
        "0",
    ),
    (
        1,
        "SVDQ_BF16_STAGE_GATE_UP_L1_GEMM",
        "SVDQ_FACTOR_GATE_UP_L1",
        "SVDQ_REGION_ROUTED_X",
        "SVDQ_REGION_LOWRANK_RANK_1",
        "routedRows",
        "info.hiddenSize",
        "gate_up_rank_columns",
        "0",
        "0",
    ),
    (
        2,
        "SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT",
        "SVDQ_INVALID_ID",
        "SVDQ_REGION_LOWRANK_RANK_1",
        "SVDQ_REGION_LOWRANK_RANK_1",
        "routedRows",
        "gate_up_rank_columns",
        "gate_up_rank_columns",
        "0",
        "0",
    ),
    (
        3,
        "SVDQ_BF16_STAGE_GATE_L2_GEMM",
        "SVDQ_FACTOR_GATE_L2",
        "SVDQ_REGION_LOWRANK_RANK_1",
        "SVDQ_REGION_PROJECTION_1",
        "routedRows",
        "info.gateRank",
        "info.intermediateSize",
        "info.gateRankOffset",
        "0",
    ),
    (
        4,
        "SVDQ_BF16_STAGE_UP_L2_GEMM",
        "SVDQ_FACTOR_UP_L2",
        "SVDQ_REGION_LOWRANK_RANK_1",
        "SVDQ_REGION_PROJECTION_1",
        "routedRows",
        "info.upRank",
        "info.intermediateSize",
        "info.upRankOffset",
        "info.intermediateSize",
    ),
    (
        5,
        "SVDQ_BF16_STAGE_DOWN_L1_GEMM",
        "SVDQ_FACTOR_DOWN_L1",
        "SVDQ_REGION_HIDDEN",
        "SVDQ_REGION_LOWRANK_RANK_2",
        "routedRows",
        "info.intermediateSize",
        "info.downRank",
        "0",
        "0",
    ),
    (
        6,
        "SVDQ_BF16_STAGE_DOWN_L2_GEMM",
        "SVDQ_FACTOR_DOWN_L2",
        "SVDQ_REGION_LOWRANK_RANK_2",
        "SVDQ_REGION_PROJECTION_2",
        "routedRows",
        "info.downRank",
        "info.hiddenSize",
        "0",
        "0",
    ),
]

RESIDUAL_STAGES = [
    {
        "id": 0,
        "name": "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT",
        "input_region": "SVDQ_REGION_ROUTED_X",
        "scale_region": "SVDQ_REGION_X_SCALE",
        "output_region": "SVDQ_REGION_X_Q",
        "m": "routedRows",
        "k": "info.hiddenSize",
        "n": "info.hiddenSize",
        "residual_weight_slot": "SVDQ_INVALID_ID",
        "residual_scale_slot": "SVDQ_INVALID_ID",
        "residual_only": True,
    },
    {
        "id": 1,
        "name": "SVDQ_RESIDUAL_STAGE_W4A8_GMM1",
        "input_region": "SVDQ_REGION_X_Q",
        "scale_region": "SVDQ_REGION_X_SCALE",
        "output_region": "SVDQ_REGION_ACCUMULATOR_1",
        "m": "routedRows",
        "k": "info.hiddenSize",
        "n": "info.intermediateSize * 2",
        "residual_weight_slot": "weight1",
        "residual_scale_slot": "scale1",
        "residual_only": True,
    },
    {
        "id": 2,
        "name": "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN",
        "input_region": "SVDQ_REGION_HIDDEN",
        "scale_region": "SVDQ_REGION_HIDDEN_SCALE",
        "output_region": "SVDQ_REGION_HIDDEN_Q",
        "m": "routedRows",
        "k": "info.intermediateSize",
        "n": "info.intermediateSize",
        "residual_weight_slot": "SVDQ_INVALID_ID",
        "residual_scale_slot": "SVDQ_INVALID_ID",
        "residual_only": True,
    },
    {
        "id": 3,
        "name": "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
        "input_region": "SVDQ_REGION_HIDDEN_Q",
        "scale_region": "SVDQ_REGION_HIDDEN_SCALE",
        "output_region": "SVDQ_REGION_ACCUMULATOR_2",
        "m": "routedRows",
        "k": "info.intermediateSize",
        "n": "info.hiddenSize",
        "residual_weight_slot": "weight2",
        "residual_scale_slot": "scale2",
        "residual_only": True,
    },
]

RESIDUAL_QUANT_LAUNCHES = [
    {
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
    },
    {
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
    },
]

RESIDUAL_GMM_LAUNCHES = [
    {
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
    },
    {
        "id": 1,
        "name": "SVDQ_RESIDUAL_STAGE_W4A8_GMM2",
        "input_region": "SVDQ_REGION_HIDDEN_Q",
        "activation_scale_region": "SVDQ_REGION_HIDDEN_SCALE",
        "output_region": "SVDQ_REGION_ACCUMULATOR_2",
        "m": "routedRows",
        "k": "info.intermediateSize",
        "n": "info.hiddenSize",
        "residual_weight_slot": "weight2",
        "residual_scale_slot": "scale2",
        "residual_bias_slot": "bias2",
        "list_len": "info.expertPerRank",
        "group_list_type": 1,
        "group_type": 0,
        "split_item": 2,
        "trans_b": False,
        "weight_nz": True,
        "residual_only": True,
    },
]

MIXED_EPILOGUE_LAUNCHES = [
    {
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
    },
    {
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
    },
]

FINAL_COMBINE_LAUNCH = {
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

LOWRANK_INVOCATIONS = [
    {
        "id": 0,
        "name": "SVDQ_LOWRANK_INVOCATION_GATE_UP",
        "input_region": "SVDQ_REGION_ROUTED_X",
        "rank_region": "SVDQ_REGION_LOWRANK_RANK_1",
        "output_region": "SVDQ_REGION_PROJECTION_1",
        "down_factor": "SVDQ_FACTOR_GATE_UP_L1",
        "up_factor": "SVDQ_FACTOR_GATE_L2",
        "second_up_factor": "SVDQ_FACTOR_UP_L2",
        "input_columns": "info.hiddenSize",
        "primary_rank_columns": "info.gateRank",
        "second_rank_columns": "info.upRank",
        "output_columns": "info.intermediateSize * 2",
        "primary_input_column_offset": "info.gateRankOffset",
        "primary_output_column_offset": "0",
        "second_input_column_offset": "info.upRankOffset",
        "second_output_column_offset": "info.intermediateSize",
        "accumulator_region": "SVDQ_REGION_LOWRANK_ACCUMULATOR_1",
    },
    {
        "id": 1,
        "name": "SVDQ_LOWRANK_INVOCATION_DOWN",
        "input_region": "SVDQ_REGION_HIDDEN",
        "rank_region": "SVDQ_REGION_LOWRANK_RANK_2",
        "output_region": "SVDQ_REGION_PROJECTION_2",
        "down_factor": "SVDQ_FACTOR_DOWN_L1",
        "up_factor": "SVDQ_FACTOR_DOWN_L2",
        "second_up_factor": "SVDQ_INVALID_ID",
        "input_columns": "info.intermediateSize",
        "primary_rank_columns": "info.downRank",
        "second_rank_columns": "0",
        "output_columns": "info.hiddenSize",
        "primary_input_column_offset": "0",
        "primary_output_column_offset": "0",
        "second_input_column_offset": "0",
        "second_output_column_offset": "0",
        "accumulator_region": "SVDQ_REGION_LOWRANK_ACCUMULATOR_2",
    },
]


def _read_sources(repo_root: Path) -> dict[str, str]:
    return {
        "op_cmake": (repo_root / OP_CMAKE).read_text(encoding="utf-8"),
        "op_def": (repo_root / OP_DEF).read_text(encoding="utf-8"),
        "debug_op_def": (repo_root / DEBUG_OP_DEF).read_text(encoding="utf-8"),
        "op_proto": (repo_root / OP_PROTO).read_text(encoding="utf-8"),
        "debug_op_api_header": (repo_root / DEBUG_OP_API_HEADER).read_text(encoding="utf-8"),
        "debug_op_api_wrapper": (repo_root / DEBUG_OP_API_WRAPPER).read_text(encoding="utf-8"),
        "host_tiling": (repo_root / HOST_TILING).read_text(encoding="utf-8"),
        "kernel_contract": (repo_root / KERNEL_CONTRACT).read_text(encoding="utf-8"),
        "kernel_tiling": (repo_root / KERNEL_TILING).read_text(encoding="utf-8"),
        "lowrank_header": (repo_root / LOWRANK_HEADER).read_text(encoding="utf-8"),
        "lowrank_debug_tiling": (repo_root / LOWRANK_DEBUG_TILING).read_text(encoding="utf-8"),
        "lowrank_debug_header": (repo_root / LOWRANK_DEBUG_HEADER).read_text(encoding="utf-8"),
        "lowrank_debug_kernel": (repo_root / LOWRANK_DEBUG_KERNEL).read_text(encoding="utf-8"),
        "lowrank_debug_alias_cmake": (repo_root / LOWRANK_DEBUG_ALIAS_CMAKE).read_text(encoding="utf-8"),
        "lowrank_debug_alias_kernel": (repo_root / LOWRANK_DEBUG_ALIAS_KERNEL).read_text(encoding="utf-8"),
        "lowrank_debug_torch_adapter": (repo_root / LOWRANK_DEBUG_TORCH_ADAPTER).read_text(encoding="utf-8"),
        "w4a8_debug_torch_adapter": (repo_root / W4A8_DEBUG_TORCH_ADAPTER).read_text(encoding="utf-8"),
        "torch_binding": (repo_root / TORCH_BINDING).read_text(encoding="utf-8"),
        "torch_binding_meta": (repo_root / TORCH_BINDING_META).read_text(encoding="utf-8"),
        "lowrank_debug_probe": (repo_root / LOWRANK_DEBUG_PROBE).read_text(encoding="utf-8"),
        "w4a8_debug_probe": (repo_root / W4A8_DEBUG_PROBE).read_text(encoding="utf-8"),
        "lowrank_debug_install_validate": (repo_root / LOWRANK_DEBUG_INSTALL_VALIDATE).read_text(
            encoding="utf-8"
        ),
        "build_aclnn": (repo_root / BUILD_ACLNN).read_text(encoding="utf-8"),
        "official_w4a8_cmake": (repo_root / OFFICIAL_W4A8_CMAKE).read_text(encoding="utf-8"),
        "official_w4a8_debug_def": (repo_root / OFFICIAL_W4A8_DEBUG_DEF).read_text(encoding="utf-8"),
        "official_w4a8_debug_api_header": (repo_root / OFFICIAL_W4A8_DEBUG_API_HEADER).read_text(
            encoding="utf-8"
        ),
        "official_w4a8_debug_api_wrapper": (
            repo_root / OFFICIAL_W4A8_DEBUG_API_WRAPPER
        ).read_text(encoding="utf-8"),
        "official_w4a8_host_tiling": (repo_root / OFFICIAL_W4A8_HOST_TILING).read_text(
            encoding="utf-8"
        ),
        "official_w4a8_kernel": (repo_root / OFFICIAL_W4A8_KERNEL).read_text(encoding="utf-8"),
        "official_w4a8_kernel_entry": (repo_root / OFFICIAL_W4A8_KERNEL_ENTRY).read_text(
            encoding="utf-8"
        ),
        "official_w4a8_debug_kernel_entry": (
            repo_root / OFFICIAL_W4A8_DEBUG_KERNEL_ENTRY
        ).read_text(encoding="utf-8"),
        "official_w4a8_op": (repo_root / OFFICIAL_W4A8_OP).read_text(encoding="utf-8"),
        "official_w4a8_gmm1_epilogue": (repo_root / OFFICIAL_W4A8_GMM1_EPILOGUE).read_text(
            encoding="utf-8"
        ),
        "official_w4a8_gmm2_epilogue": (repo_root / OFFICIAL_W4A8_GMM2_EPILOGUE).read_text(
            encoding="utf-8"
        ),
    }


def _sync_flag_records() -> list[dict[str, Any]]:
    return [
        {
            "id": flag_id,
            "name": name,
            "producer_stage": producer,
            "consumer_stage": consumer,
            "workspace_region": region,
            "producer_signal_index": flag_id,
            "consumer_wait_index": flag_id,
        }
        for flag_id, name, producer, consumer, region in SYNC_FLAGS
    ]


def _bf16_stage_records() -> list[dict[str, Any]]:
    return [
        {
            "id": stage_id,
            "name": name,
            "factor": factor,
            "input_region": input_region,
            "output_region": output_region,
            "m": m,
            "k": k,
            "n": n,
            "input_column_offset": input_offset,
            "output_column_offset": output_offset,
        }
        for stage_id, name, factor, input_region, output_region, m, k, n, input_offset, output_offset in BF16_STAGES
    ]


def _residual_stage_records() -> list[dict[str, Any]]:
    return [dict(stage) for stage in RESIDUAL_STAGES]


def _tokens_in_order(source: str, tokens: list[str]) -> bool:
    offset = 0
    for token in tokens:
        index = source.find(token, offset)
        if index < 0:
            return False
        offset = index + len(token)
    return True


def _production_host_tiling_source(sources: dict[str, str]) -> str:
    host_tiling_start = sources["host_tiling"].find(
        "static ge::graphStatus DispatchFFNCombineW4A8SVDQTilingFunc(gert::TilingContext* context)"
    )
    host_tiling_end = sources["host_tiling"].find(
        "struct DispatchFFNCombineW4A8SVDQCompileInfo", host_tiling_start
    )
    if host_tiling_start < 0 or host_tiling_end < 0:
        return ""
    return sources["host_tiling"][host_tiling_start:host_tiling_end]


def _build_aclnn_branch(source: str, start: str, end: str) -> str:
    if start not in source or end not in source:
        return ""
    branch = source[source.index(start) :]
    return branch[: branch.index(end)]


def _svdq_ops_selected_in_build_branch(branch: str) -> bool:
    required_ops = (
        '"dispatch_ffn_combine_w4_a8"',
        '"dispatch_ffn_combine_w4_a8_svdq"',
        '"svdq_low_rank_debug_readback"',
        '"dispatch_ffn_combine_bf16"',
    )
    return (
        all(op in branch for op in required_ops)
        and branch.index(required_ops[0]) < branch.index(required_ops[1])
        and branch.index(required_ops[1]) < branch.index(required_ops[2])
        and branch.index(required_ops[2]) < branch.index(required_ops[3])
    )


def _source_proof(sources: dict[str, str]) -> dict[str, bool]:
    a2_aclnn_branch = _build_aclnn_branch(
        sources["build_aclnn"],
        'elif [[ "$SOC_VERSION" =~ ^ascend910b ]];',
        'elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]];',
    )
    a3_aclnn_branch = _build_aclnn_branch(
        sources["build_aclnn"],
        'elif [[ "$SOC_VERSION" =~ ^ascend910_93 ]];',
        'elif [[ "$SOC_VERSION" =~ ^ascend950 ]];',
    )
    process_start = sources["kernel_contract"].find("__aicore__ inline void Process()")
    process_end = sources["kernel_contract"].find("__aicore__ inline bool HasCompleteTilingContract()")
    process_source = sources["kernel_contract"][process_start:process_end] if process_start >= 0 else ""
    dispatch_start = sources["kernel_contract"].find("__aicore__ inline bool RunDispatchRoutingStage() const")
    dispatch_end = sources["kernel_contract"].find(
        "__aicore__ inline SVDQResidualStageContract", dispatch_start
    )
    dispatch_source = sources["kernel_contract"][dispatch_start:dispatch_end] if dispatch_start >= 0 else ""
    residual_quant_start = sources["kernel_contract"].find(
        "__aicore__ inline bool RunResidualDynamicQuantStage(uint32_t stageId)"
    )
    residual_quant_end = sources["kernel_contract"].find(
        "__aicore__ inline bool RunResidualGmmStage", residual_quant_start
    )
    residual_quant_source = (
        sources["kernel_contract"][residual_quant_start:residual_quant_end]
        if residual_quant_start >= 0
        else ""
    )
    host_tiling_source = _production_host_tiling_source(sources)
    host_bridge_build_index = host_tiling_source.find("BuildResidualW4A8OfficialTiling(tilingData);")
    host_workspace_index = host_tiling_source.find("workSpaces[0] = SVDQ_SYSTEM_WORKSPACE + info.workspaceBytes +")
    host_fail_closed_log_index = host_tiling_source.find("production tiling is fail-closed after metadata construction")
    host_fail_closed_return_index = (
        host_tiling_source.find("return ge::GRAPH_FAILED;", host_fail_closed_log_index)
        if host_fail_closed_log_index >= 0
        else -1
    )
    host_fail_closed_after_metadata = (
        0 <= host_bridge_build_index < host_workspace_index < host_fail_closed_log_index < host_fail_closed_return_index
    )
    dispatch_compile_options = _build_aclnn_branch(
        sources["op_cmake"],
        "add_ops_compile_options(\n    OP_NAME DispatchFFNCombineW4A8SVDQ",
        "add_ops_compile_options(\n    OP_NAME SVDQLowRankDebugReadback",
    )
    lowrank_debug_compile_options = _build_aclnn_branch(
        sources["op_cmake"],
        "add_ops_compile_options(\n    OP_NAME SVDQLowRankDebugReadback",
        "add_ops_compile_options(\n    OP_NAME SVDQMixedEpilogueDebugReadback",
    )
    return {
        "host_tiling_builds_workspace_map": "BuildWorkspaceMap(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_sync_flags": "BuildSyncFlagTable(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_bf16_stage_shapes": "BuildBF16StageShapeTable(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_residual_stage_shapes": (
            "BuildResidualStageShapeTable(tilingData)" in sources["host_tiling"]
        ),
        "host_tiling_builds_lowrank_invocations": "BuildLowRankInvocationTable(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_dispatch_routing_subtiling": (
            "BuildDispatchRoutingTiling(tilingData)" in sources["host_tiling"]
            and "MoeInitRoutingQuantV2TilingBase routingBase" in sources["host_tiling"]
            and "routingBase.DoTiling" in sources["host_tiling"]
            and "routingBase.tilingKey_" in sources["host_tiling"]
            and "routingBase.workspaceSize_" in sources["host_tiling"]
            and "routingBase.quantTilingData" in sources["host_tiling"]
            and "SVDQ_ROUTING_BLOCK_NUM" in sources["host_tiling"]
            and "SVDQ_ROUTING_UB_SIZE" in sources["host_tiling"]
            and "tilingData->dispatchRouting.routingWorkspaceBytes" in sources["host_tiling"]
        ),
        "host_tiling_fail_closed_after_metadata_construction": host_fail_closed_after_metadata,
        "op_cmake_has_local_debug_readback_option": (
            "option(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in sources["op_cmake"]
        ),
        "op_cmake_debug_readback_defaults_on": (
            "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)" in sources["op_cmake"]
            and "ON)" in sources["op_cmake"]
        ),
        "op_cmake_scopes_debug_readback_to_lowrank_debug_op": (
            "${_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS}" in lowrank_debug_compile_options
            and "${_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS}" not in dispatch_compile_options
        ),
        "kernel_resolves_workspace_addresses": "WorkspaceAddress(uint32_t regionId)" in sources["kernel_contract"],
        "kernel_resolves_rank_workspace_regions": (
            "workspace_.lowRankRank1 = WorkspaceAddress(SVDQ_REGION_LOWRANK_RANK_1)" in sources["kernel_contract"]
            and "workspace_.lowRankRank2 = WorkspaceAddress(SVDQ_REGION_LOWRANK_RANK_2)" in sources["kernel_contract"]
        ),
        "kernel_exposes_bf16_stage_contracts": "BF16StageContract(uint32_t stageId)" in sources["kernel_contract"],
        "kernel_records_dispatch_routing_contract": (
            "SVDQDispatchRoutingContract" in sources["kernel_contract"]
            and "DispatchRoutingContract() const" in sources["kernel_contract"]
            and "DispatchRoutingReady() const" in sources["kernel_contract"]
            and "RunDispatchRoutingStage() const" in sources["kernel_contract"]
            and "SVDQ_STAGE_BF16_DISPATCH" in sources["kernel_contract"]
            and "SVDQ_REGION_ROUTED_X" in sources["kernel_contract"]
            and "SVDQ_REGION_EXPANDED_ROW_IDX" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_TO_QUANT_1" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE" in sources["kernel_contract"]
        ),
        "kernel_tiling_contains_dispatch_routing_subtiling": (
            (
                '#include "../../dispatch_ffn_combine_w4_a8/op_kernel/moe_init_routing_quant_v2/'
                'moe_init_routing_quant_v2_tiling.h"'
            )
            in sources["kernel_tiling"]
            and "struct SVDQDispatchRoutingTiling" in sources["kernel_tiling"]
            and "bf16RoutingTilingKey" in sources["kernel_tiling"]
            and "bf16RoutingWorkspaceBytes" in sources["kernel_tiling"]
            and "InnerMoeInitRoutingV2TilingData moeInitRoutingV2TilingData" in sources["kernel_tiling"]
            and "initRoutingQuantTilingKey" in sources["kernel_tiling"]
            and "routingWorkspaceBytes" in sources["kernel_tiling"]
            and "MoeInitRoutingQuantV2TilingData moeInitRoutingQuantV2TilingData" in sources["kernel_tiling"]
            and "SVDQDispatchRoutingTiling dispatchRouting" in sources["kernel_tiling"]
        ),
        "kernel_dispatch_routing_uses_official_tiling_contract": (
            "DispatchRoutingTiling() const" in sources["kernel_contract"]
            and "DispatchRoutingTempWorkspace() const" in sources["kernel_contract"]
            and "tilingData_.dispatchRouting" in sources["kernel_contract"]
            and "routingTiling.bf16RoutingTilingKey != 0" in sources["kernel_contract"]
            and "routingTiling.bf16RoutingWorkspaceBytes > 0" in sources["kernel_contract"]
            and "routingTiling.aivNum > 0" in sources["kernel_contract"]
        ),
        "kernel_dispatch_routing_calls_official_bf16_helper": (
            "svdq_moe_init_routing_v2<bfloat16_t>" in sources["kernel_contract"]
            and "MoeInitRoutingQuantV2::MoeV2FullLoad<DTYPE_X>" in sources["kernel_contract"]
            and "MoeInitRoutingQuantV2::MoeV2GatherOut<DTYPE_X>" in sources["kernel_contract"]
            and "WorkspaceAddress(contract.routedOutputRegionId)" in sources["kernel_contract"]
            and "WorkspaceAddress(contract.routeIndexRegionId)" in sources["kernel_contract"]
            and "runtime_.expertTokenNums" in sources["kernel_contract"]
            and "DispatchRoutingTempWorkspace()" in sources["kernel_contract"]
            and "&routingTiling.moeInitRoutingV2TilingData" in sources["kernel_contract"]
            and "routingTiling.bf16RoutingTilingKey" in sources["kernel_contract"]
        ),
        "kernel_dispatch_routing_execution_enabled": (
            "RunDispatchRoutingStage() const" in sources["kernel_contract"]
            and "DispatchRoutingReady()" in sources["kernel_contract"]
            and "svdq_moe_init_routing_v2<bfloat16_t>" in dispatch_source
            and "return true;" in dispatch_source
            and "ExecuteLowRankInvocation(SVDQ_LOWRANK_INVOCATION_GATE_UP)" in process_source
        ),
        "kernel_process_orders_svdq_data_dependencies": _tokens_in_order(
            process_source,
            [
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
            ],
        ),
        "kernel_validates_complete_sync_flag_table": (
            "HasCompleteSyncFlagTable() const" in sources["kernel_contract"]
            and "ValidateSyncFlag(" in sources["kernel_contract"]
            and "flag.producerSignalIndex == flagId" in sources["kernel_contract"]
            and "flag.consumerWaitIndex == flagId" in sources["kernel_contract"]
            and "HasCompleteSyncFlagTable();" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_TO_QUANT_1" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE" in sources["kernel_contract"]
        ),
        "kernel_synchronizes_stage_boundaries": (
            "SynchronizeStageBoundary(" in sources["kernel_contract"]
            and "AscendC::SyncAll();" in sources["kernel_contract"]
            and "SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_STAGE_BF16_DISPATCH" in process_source
            and "SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_LOWRANK_1" in process_source
            and "SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_STAGE_QUANT_1" in process_source
            and "SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_W4A8_GEMM_1"
            in process_source
            and "SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_STAGE_MIXED_EPILOGUE_1"
            in process_source
            and "SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_LOWRANK_2"
            in process_source
            and "SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_STAGE_QUANT_2" in process_source
            and "SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_W4A8_GEMM_2"
            in process_source
            and "SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE" in process_source
        ),
        "kernel_exposes_residual_stage_contracts": (
            "ResidualStageShape(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualStageContract(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualExecutionPlan(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualStageReady(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualExecutionPlanReady(uint32_t stageId)" in sources["kernel_contract"]
        ),
        "kernel_binds_residual_weight_scale_slots": (
            "SVDQ_RESIDUAL_WEIGHT1_SLOT = 1" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_WEIGHT2_SLOT = 2" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_SCALE1_SLOT = 4" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_SCALE2_SLOT = 5" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_BIAS1_SLOT = 6" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_BIAS2_SLOT = 7" in sources["kernel_contract"]
            and "ResidualWeightAddress(uint32_t residualWeightSlot)" in sources["kernel_contract"]
            and "ResidualScaleAddress(uint32_t residualScaleSlot)" in sources["kernel_contract"]
            and "ResidualBiasAddress(uint32_t residualBiasSlot)" in sources["kernel_contract"]
        ),
        "kernel_residual_dispatches_dynamic_quant_and_gmm": (
            "SVDQResidualOpKind" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_OP_DYNAMIC_QUANT" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_OP_W4A8_GMM" in sources["kernel_contract"]
            and "RunResidualDynamicQuantStage(uint32_t stageId)" in sources["kernel_contract"]
            and "RunResidualGmmStage(uint32_t stageId)" in sources["kernel_contract"]
            and "RunResidualStage(uint32_t stageId)" in sources["kernel_contract"]
            and "RunResidualDynamicQuantStage(stageId)" in sources["kernel_contract"]
            and "RunResidualGmmStage(stageId)" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_BIAS1_SLOT" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_BIAS2_SLOT" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_launch_descriptor_recorded": (
            "SVDQ_RESIDUAL_GMM_COUNT = 2" in sources["kernel_tiling"]
            and "struct SVDQResidualGmmShape" in sources["kernel_tiling"]
            and "SVDQResidualGmmShape residualGmmShapes[SVDQ_RESIDUAL_GMM_COUNT]" in sources["kernel_tiling"]
            and "SetResidualGmmShape" in sources["host_tiling"]
            and "BuildResidualGmmShapeTable" in sources["host_tiling"]
            and "BuildResidualGmmShapeTable(tilingData)" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_W4A8_GMM1" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_W4A8_GMM2" in sources["host_tiling"]
            and "gmm.groupListType = 1" in sources["host_tiling"]
            and "gmm.groupType = 0" in sources["host_tiling"]
            and "gmm.splitItem = 2" in sources["host_tiling"]
            and "gmm.weightNz = true" in sources["host_tiling"]
            and "SVDQResidualGmmLaunch" in sources["kernel_contract"]
            and "ResidualGmmShape(uint32_t gmmId)" in sources["kernel_contract"]
            and "ResidualGmmIdForStage(uint32_t stageId)" in sources["kernel_contract"]
            and "BuildResidualGmmLaunch(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualGmmLaunchReady(uint32_t stageId)" in sources["kernel_contract"]
            and "BuildResidualGmmLaunch(stageId)" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_bridge_contract_recorded": (
            "SVDQResidualGmmOfficialBridgeContract" in sources["kernel_contract"]
            and "ResidualGmmOfficialBridgeContract(" in sources["kernel_contract"]
            and "ResidualGmmOfficialBridgeReady(uint32_t stageId) const" in sources["kernel_contract"]
            and "SVDQ_OFFICIAL_W4A8_KERNEL_DISPATCH_FFN_COMBINE" in sources["kernel_contract"]
            and "SVDQ_OFFICIAL_W4A8_AIC_GMM" in sources["kernel_contract"]
            and "SVDQ_OFFICIAL_W4A8_AIV_DEQUANT" in sources["kernel_contract"]
            and "SVDQ_REGION_X_Q, SVDQ_REGION_X_SCALE" in sources["kernel_contract"]
            and "SVDQ_REGION_HIDDEN_Q, SVDQ_REGION_HIDDEN_SCALE" in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_1, true, true, true, true, true" in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_2, true, true, true, true, true" in sources["kernel_contract"]
            and "bridge.requiresPackedW4Weights && bridge.requiresOfficialAicAccumulator" in sources[
                "kernel_contract"
            ]
            and "bridge.requiresOfficialC2VHandoff && bridge.requiresOfficialAivDequant" in sources[
                "kernel_contract"
            ]
            and "bridge.producesFP32Residual && launch.weightNz && launch.residualOnly" in sources[
                "kernel_contract"
            ]
            and "Execution remains fail-closed until official W4A8 producer segments can be interleaved"
            in sources["kernel_contract"]
            and "without accepting the ordinary W4A8" in sources["kernel_contract"]
            and "final-combine output as the fused SVDQ result." in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_tiling_bridge_recorded": (
            "../../dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8_tiling.h"
            in sources["kernel_tiling"]
            and "SVDQResidualW4A8BridgeTiling" in sources["kernel_tiling"]
            and "DispatchFFNCombineW4A8TilingData officialTiling" in sources["kernel_tiling"]
            and "SVDQResidualW4A8BridgeTiling residualW4A8Bridge" in sources["kernel_tiling"]
            and "ResidualW4A8OfficialWorkspaceBytes" in sources["host_tiling"]
            and "BuildResidualW4A8OfficialTiling" in sources["host_tiling"]
            and "BuildResidualW4A8OfficialTiling(tilingData)" in sources["host_tiling"]
            and "bridge.officialK = info.hiddenSize" in sources["host_tiling"]
            and "bridge.officialN = info.intermediateSize * 2" in sources["host_tiling"]
            and "bridge.officialListLen = info.expertPerRank" in sources["host_tiling"]
            and "bridge.hostExecutionFailClosed = true" in sources["host_tiling"]
            and "officialInfo.isWeightNz = true" in sources["host_tiling"]
            and "officialInfo.isTransposeB = false" in sources["host_tiling"]
            and "official.cocTiling.m0 = 128" in sources["host_tiling"]
            and "official.cocTiling.k0 = 256" in sources["host_tiling"]
            and "official.cocTiling.n0 = 256" in sources["host_tiling"]
            and "official.cocTiling.initRoutingQuantTilingKey = tilingData->dispatchRouting.initRoutingQuantTilingKey"
            in sources["host_tiling"]
            and "official.cocTiling.moeInitRoutingQuantV2TilingData =" in sources["host_tiling"]
        ),
        "kernel_residual_gmm_official_tiling_bridge_consumed": (
            "ResidualW4A8BridgeTiling() const" in sources["kernel_contract"]
            and "return tilingData_.residualW4A8Bridge" in sources["kernel_contract"]
            and "ResidualGmmOfficialTilingBridgeReady(uint32_t stageId) const" in sources["kernel_contract"]
            and "SVDQResidualW4A8BridgeTiling bridge = ResidualW4A8BridgeTiling()" in sources["kernel_contract"]
            and "DispatchFFNCombineW4A8Info officialInfo = "
            "bridge.officialTiling.dispatchFFNCombineW4A8Info" in sources["kernel_contract"]
            and "bridge.officialK == tilingData_.info.hiddenSize" in sources["kernel_contract"]
            and "bridge.officialN == tilingData_.info.intermediateSize * 2" in sources["kernel_contract"]
            and "bridge.hostExecutionFailClosed" in sources["kernel_contract"]
            and "officialInfo.isWeightNz" in sources["kernel_contract"]
            and "!officialInfo.isTransposeB" in sources["kernel_contract"]
            and "officialCoc.initRoutingQuantTilingKey == tilingData_.dispatchRouting.initRoutingQuantTilingKey"
            in sources["kernel_contract"]
            and "stageId == SVDQ_RESIDUAL_STAGE_W4A8_GMM1" in sources["kernel_contract"]
            and "launch.k == bridge.officialK && launch.n == bridge.officialN" in sources["kernel_contract"]
            and "stageId == SVDQ_RESIDUAL_STAGE_W4A8_GMM2" in sources["kernel_contract"]
            and "launch.k == bridge.officialN / 2 && launch.n == bridge.officialK" in sources["kernel_contract"]
            and "OfficialW4A8FullLifecycleLaunchReady(uint32_t stageId) const" in sources["kernel_contract"]
            and "!OfficialW4A8FullLifecycleLaunchReady(stageId)" in sources["kernel_contract"]
            and "Execution remains fail-closed until official W4A8 producer segments can be interleaved"
            in sources["kernel_contract"]
            and "without accepting the ordinary W4A8" in sources["kernel_contract"]
            and "final-combine output as the fused SVDQ result." in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_full_lifecycle_call_surface_recorded": (
            "GM_ADDR tiling;" in sources["kernel_contract"]
            and "GM_ADDR gmm1PostDequantFp32;" in sources["kernel_contract"]
            and "GM_ADDR gmm2PostDequantFp32;" in sources["kernel_contract"]
            and "GM_ADDR externalHiddenPacked;" in sources["kernel_contract"]
            and "GM_ADDR externalHiddenScale;" in sources["kernel_contract"]
            and "GM_ADDR gmm2AccumulatorInt32;" in sources["kernel_contract"]
            and "usesOfficialGmm1Fp32Tap" in sources["kernel_contract"]
            and "usesOfficialGmm2Fp32Tap" in sources["kernel_contract"]
            and "usesSvdqHiddenPackedBoundary" in sources["kernel_contract"]
            and "requiresNoOrdinaryW4A8FinalCombine" in sources["kernel_contract"]
            and "../../dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h" in sources[
                "kernel_contract"
            ]
            and "using SVDQOfficialW4A8Op =" in sources["kernel_contract"]
            and "DispatchFFNCombineW4A8Impl::DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true>"
            in sources["kernel_contract"]
            and "OfficialW4A8WrapperTypeBound() const" in sources["kernel_contract"]
            and "sizeof(SVDQOfficialW4A8Op) > 0" in sources["kernel_contract"]
            and "runtime_.tiling = tilingGM" in sources["kernel_contract"]
            and "struct SVDQOfficialW4A8FullLifecycleLaunch" in sources["kernel_contract"]
            and "EmbeddedOfficialW4A8TilingGM() const" in sources["kernel_contract"]
            and "reinterpret_cast<__gm__ DispatchFFNCombineW4A8SVDQTilingData*>(runtime_.tiling)"
            in sources["kernel_contract"]
            and "svdqTiling->residualW4A8Bridge.officialTiling" in sources["kernel_contract"]
            and "BuildOfficialW4A8FullLifecycleLaunch() const" in sources["kernel_contract"]
            and "runtime_.x, runtime_.residual.w1, runtime_.residual.w2, runtime_.expertId" in sources[
                "kernel_contract"
            ]
            and "runtime_.residual.scale1, runtime_.residual.scale2" in sources["kernel_contract"]
            and "runtime_.residual.bias1, runtime_.residual.bias2" in sources["kernel_contract"]
            and "runtime_.probs, runtime_.xActiveMask, WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT)"
            in sources["kernel_contract"]
            and "runtime_.expertTokenNums, WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE)" in sources[
                "kernel_contract"
            ]
            and "runtime_.tiling, EmbeddedOfficialW4A8TilingGM(), WorkspaceAddress(SVDQ_REGION_ACCUMULATOR_1)"
            in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_ACCUMULATOR_2), WorkspaceAddress(SVDQ_REGION_HIDDEN_Q)"
            in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_HIDDEN_SCALE)" in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR)" in sources["kernel_contract"]
            and "launch.gmm1PostDequantFp32 != nullptr && launch.gmm2PostDequantFp32 != nullptr"
            in sources["kernel_contract"]
            and "launch.gmm2AccumulatorInt32 == WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR)"
            in sources["kernel_contract"]
            and "launch.out == WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT)"
            in sources["kernel_contract"]
            and "launch.externalHiddenPacked != nullptr && launch.externalHiddenScale != nullptr"
            in sources["kernel_contract"]
            and "OfficialW4A8WrapperTypeBound()" in sources["kernel_contract"]
            and "launch.usesOfficialGmm1Fp32Tap && launch.usesOfficialGmm2Fp32Tap" in sources[
                "kernel_contract"
            ]
            and "launch.usesSvdqHiddenPackedBoundary && launch.requiresNoOrdinaryW4A8FinalCombine" in sources[
                "kernel_contract"
            ]
            and "launch.requiresOfficialWrapper && launch.requiresFullAicAivLifecycle" in sources[
                "kernel_contract"
            ]
        ),
        "kernel_residual_gmm_official_interleaved_producer_contract_recorded": (
            "struct SVDQOfficialW4A8InterleavedProducerContract" in sources["kernel_contract"]
            and "OfficialW4A8InterleavedProducerContract() const" in sources["kernel_contract"]
            and "OfficialW4A8InterleavedProducerReady() const" in sources["kernel_contract"]
            and "SVDQ_RESIDUAL_STAGE_W4A8_GMM1, SVDQ_RESIDUAL_STAGE_W4A8_GMM2"
            in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_1, SVDQ_REGION_ACCUMULATOR_2, SVDQ_REGION_HIDDEN_Q"
            in sources["kernel_contract"]
            and "SVDQ_REGION_HIDDEN_SCALE, SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN"
            in sources["kernel_contract"]
            and "requiresOfficialFetchAndPreprocessInt8ToInt4" in sources["kernel_contract"]
            and "requiresOfficialGmm1BeforeMixedSwiGLU" in sources["kernel_contract"]
            and "requiresSvdqHiddenBeforeOfficialGmm2" in sources["kernel_contract"]
            and "requiresOfficialC2VHandoffAndBlockEpilogue2" in sources["kernel_contract"]
            and "forbidsMonolithicProcessAsProductionResult" in sources["kernel_contract"]
            and "requiresScratchOrdinaryW4A8Output" in sources["kernel_contract"]
            and "!OfficialW4A8InterleavedProducerReady()" in sources["kernel_contract"]
            and "GMM1 must write the FP32 tap before mixed SwiGLU" in sources["kernel_contract"]
            and "GMM2 must consume the SVDQ hidden" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_split_producer_segment_contract_recorded": (
            "enum SVDQOfficialW4A8ProducerSegmentId" in sources["kernel_contract"]
            and "SVDQ_OFFICIAL_W4A8_SEGMENT_GMM1_FP32_TAP" in sources["kernel_contract"]
            and "SVDQ_OFFICIAL_W4A8_SEGMENT_GMM2_FP32_TAP" in sources["kernel_contract"]
            and "struct SVDQOfficialW4A8ProducerSegmentContract" in sources["kernel_contract"]
            and "OfficialW4A8ProducerSegmentContract(" in sources["kernel_contract"]
            and "uint32_t stageId) const" in sources["kernel_contract"]
            and "OfficialW4A8ProducerSegmentReady(uint32_t stageId) const" in sources["kernel_contract"]
            and "segment.stopsBeforeOrdinarySwiGLU && !segment.consumesSvdqHiddenBoundary"
            in sources["kernel_contract"]
            and "segment.stopsBeforeOrdinaryFinalCombine && segment.consumesSvdqHiddenBoundary"
            in sources["kernel_contract"]
            and "segment.preservesOfficialC2VHandoff && segment.executionFailClosed" in sources[
                "kernel_contract"
            ]
            and "!OfficialW4A8ProducerSegmentReady(stageId)" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_scratch_output_recorded": (
            "SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT = 16" in sources["kernel_tiling"]
            and "SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE = 17" in sources["kernel_tiling"]
            and "SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR = 18" in sources["kernel_tiling"]
            and "SVDQ_WORKSPACE_REGION_COUNT = 19" in sources["kernel_tiling"]
            and "SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT, offset" in sources["host_tiling"]
            and "activeSlots * hiddenSize * BF16_BYTES" in sources["host_tiling"]
            and "SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE, offset" in sources["host_tiling"]
            and "SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR, offset" in sources["host_tiling"]
            and "routedRows * hiddenSize * 2 * INT32_BYTES" in sources["host_tiling"]
            and "ResidualW4A8OfficialWorkspaceBytes(info)" in sources["host_tiling"]
            and "GM_ADDR officialW4A8ScratchOut;" in sources["kernel_contract"]
            and "GM_ADDR officialW4A8Workspace;" in sources["kernel_contract"]
            and "GM_ADDR officialW4A8Gmm2Accumulator;" in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT)" in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE)" in sources["kernel_contract"]
            and "WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR)" in sources["kernel_contract"]
            and "launch.out == WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT)"
            in sources["kernel_contract"]
            and "launch.workspace == WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE)"
            in sources["kernel_contract"]
            and "launch.gmm2AccumulatorInt32 == WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_GMM2_ACCUMULATOR)"
            in sources["kernel_contract"]
            and "workspace_.officialW4A8ScratchOut = WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_SCRATCH_OUT)"
            in sources["kernel_contract"]
            and "workspace_.officialW4A8Workspace = WorkspaceAddress(SVDQ_REGION_OFFICIAL_W4A8_WORKSPACE)"
            in sources["kernel_contract"]
            and "workspace_.officialW4A8Gmm2Accumulator ="
            in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_official_full_lifecycle_execution_enabled": (
            "DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true> op" in sources[
                "kernel_contract"
            ]
            and "op.Init(launch.x, launch.w1, launch.w2" in sources["kernel_contract"]
            and "op.Process()" in sources["kernel_contract"]
            and "launch.officialTiling != nullptr" in sources["kernel_contract"]
            and "return true;" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_scalar_execution_enabled": (
            "return RunResidualPackedW4A8ScalarGmmStage(launch);" in sources["kernel_contract"]
            and "RunResidualPackedW4A8ScalarGmmStage(const SVDQResidualGmmLaunch& launch) const"
            in sources["kernel_contract"]
            and "ResolveResidualGmmExpert(launch, row)" in sources["kernel_contract"]
            and "LoadResidualGmmExpertTokenCount(" in sources["kernel_contract"]
            and "LoadResidualGmmInputINT8(" in sources["kernel_contract"]
            and "LoadResidualGmmActivationScale(" in sources["kernel_contract"]
            and "LoadResidualGmmWeightINT4(" in sources["kernel_contract"]
            and "LoadResidualGmmWeightScale(" in sources["kernel_contract"]
            and "LoadResidualGmmBias(" in sources["kernel_contract"]
            and "StoreResidualGmmOutputBF16(" in sources["kernel_contract"]
            and "launch.n % 8 != 0" in sources["kernel_contract"]
            and "const uint32_t packedColumns = launch.n / 8" in sources["kernel_contract"]
            and "const uint32_t shift = (nColumn % 8) * 4" in sources["kernel_contract"]
            and "nibble >= 8 ? nibble - 16 : nibble" in sources["kernel_contract"]
            and "UInt32BitsToFloat(static_cast<uint32_t>(packedScale & 0xffffffffULL))"
            in sources["kernel_contract"]
            and "activation * activationScale * weightValue * weightScale" in sources["kernel_contract"]
            and "StoreResidualGmmOutputBF16(launch, row, column, static_cast<bfloat16_t>(accumulator))"
            in sources["kernel_contract"]
            and "return true;" in sources["kernel_contract"]
        ),
        "kernel_residual_gmm_scalar_helpers_absent": (
            "RunResidualPackedW4A8ScalarGmmStage" not in sources["kernel_contract"]
            and "LoadResidualGmmExpertTokenCount" not in sources["kernel_contract"]
            and "ResolveResidualGmmExpert" not in sources["kernel_contract"]
            and "LoadResidualGmmInputINT8" not in sources["kernel_contract"]
            and "LoadResidualGmmActivationScale" not in sources["kernel_contract"]
            and "LoadResidualGmmWeightINT4" not in sources["kernel_contract"]
            and "LoadResidualGmmWeightScale" not in sources["kernel_contract"]
            and "LoadResidualGmmBias" not in sources["kernel_contract"]
            and "StoreResidualGmmOutputBF16" not in sources["kernel_contract"]
            and "UInt32BitsToFloat" not in sources["kernel_contract"]
        ),
        "kernel_residual_quant_launch_descriptor_recorded": (
            "SVDQ_RESIDUAL_QUANT_COUNT = 2" in sources["kernel_tiling"]
            and "struct SVDQResidualQuantShape" in sources["kernel_tiling"]
            and "SVDQResidualQuantShape residualQuantShapes[SVDQ_RESIDUAL_QUANT_COUNT]" in sources["kernel_tiling"]
            and "SetResidualQuantShape" in sources["host_tiling"]
            and "BuildResidualQuantShapeTable" in sources["host_tiling"]
            and "BuildResidualQuantShapeTable(tilingData)" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN" in sources["host_tiling"]
            and "quant.usesRouting = usesRouting" in sources["host_tiling"]
            and "quant.residualOnly = true" in sources["host_tiling"]
            and "SVDQResidualQuantLaunch" in sources["kernel_contract"]
            and "ResidualQuantShape(uint32_t quantId)" in sources["kernel_contract"]
            and "ResidualQuantIdForStage(uint32_t stageId)" in sources["kernel_contract"]
            and "BuildResidualQuantLaunch(uint32_t stageId)" in sources["kernel_contract"]
            and "ResidualQuantLaunchReady(uint32_t stageId)" in sources["kernel_contract"]
            and "BuildResidualQuantLaunch(stageId)" in sources["kernel_contract"]
        ),
        "kernel_residual_routed_input_quant_execution_enabled": (
            "../../dispatch_ffn_combine_w4_a8/op_kernel/dispatch_ffn_combine_w4_a8.h"
            in sources["kernel_contract"]
            and "DispatchQuantRoutingTempWorkspace() const" in sources["kernel_contract"]
            and "SVDQResidualQuantLaunch launch = BuildResidualQuantLaunch(stageId)" in residual_quant_source
            and "if (launch.usesRouting)" in residual_quant_source
            and "moe_init_routing_quant_v2<bfloat16_t>" in residual_quant_source
            and "launch.output" in residual_quant_source
            and "launch.routeIndex" in residual_quant_source
            and "launch.expertTokenNums" in residual_quant_source
            and "launch.activationScale" in residual_quant_source
            and "launch.workspace" in residual_quant_source
            and "routingTiling.initRoutingQuantTilingKey" in residual_quant_source
            and "return true;" in residual_quant_source
        ),
        "kernel_residual_hidden_quant_scalar_execution_enabled": (
            "RunResidualScalarDynamicQuantStage(const SVDQResidualQuantLaunch& launch) const"
            in sources["kernel_contract"]
            and "return RunResidualScalarDynamicQuantStage(launch);" in residual_quant_source
            and "LoadResidualQuantInputBF16(" in sources["kernel_contract"]
            and "StoreResidualQuantOutputINT8(" in sources["kernel_contract"]
            and "StoreResidualQuantScaleFP32(" in sources["kernel_contract"]
            and "for (uint32_t row = coreIdx; row < launch.m; row += coreCount)" in sources["kernel_contract"]
            and "const float scale = maxAbs / 127.0F" in sources["kernel_contract"]
            and "StoreResidualQuantScaleFP32(launch, row, scale)" in sources["kernel_contract"]
            and "StoreResidualQuantOutputINT8(launch, row, column, static_cast<int8_t>(0))"
            in sources["kernel_contract"]
            and "const int32_t rounded = RoundQuantValue(value / scale)" in sources["kernel_contract"]
            and "ClampInt8QuantValue(rounded)" in sources["kernel_contract"]
            and "launch.scaleElements != launch.m" in sources["kernel_contract"]
        ),
        "kernel_residual_hidden_quant_aiv_execution_enabled": (
            "RunResidualHiddenQuantAIV(const SVDQResidualQuantLaunch& launch)" in sources["kernel_contract"]
            and "QuantizeResidualHiddenRowAIV" in sources["kernel_contract"]
            and "PackResidualHiddenOfficialI4AIV" in sources["kernel_contract"]
            and "if (g_coreType == AIC)" in sources["kernel_contract"]
            and "CopyInResidualQuantBf16(hiddenBf16, inputGm, row * launch.k + column,"
            in sources["kernel_contract"]
            and "ReduceMax(reduceTmp, absHidden, scaleLocal, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "const float scale = maxAbs / 127.0f" in sources["kernel_contract"]
            and "CopyOutResidualQuantScale(scaleGm, row, scaleLocal, 1)" in sources["kernel_contract"]
            and "SetDeqScale(static_cast<half>(1.0f))" in sources["kernel_contract"]
            and "Cast(hiddenI8, quantHalf, RoundMode::CAST_RINT, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "const uint32_t rowOffset = row * launch.k" in sources["kernel_contract"]
            and "CopyOutResidualQuantI8(outputGm, rowOffset + packedOffset,"
            in sources["kernel_contract"]
            and "CopyOutResidualQuantI8(outputGm, rowOffset + launch.k / 2 + packedOffset,"
            in sources["kernel_contract"]
            and "return RunResidualHiddenQuantAIV(launch);" in sources["kernel_contract"]
        ),
        "kernel_residual_hidden_quant_scalar_helpers_absent": (
            "RunResidualScalarDynamicQuantStage" not in sources["kernel_contract"]
            and "LoadResidualQuantInputBF16" not in sources["kernel_contract"]
            and "StoreResidualQuantOutputINT8" not in sources["kernel_contract"]
            and "StoreResidualQuantScaleFP32" not in sources["kernel_contract"]
            and "RoundQuantValue" not in sources["kernel_contract"]
            and "ClampInt8QuantValue" not in sources["kernel_contract"]
        ),
        "kernel_residual_execution_dispatch_enabled": (
            "RunW4A8ResidualStages()" in sources["kernel_contract"]
            and "RunResidualStage(stageId)" in sources["kernel_contract"]
            and "RunResidualDynamicQuantStage(uint32_t stageId)" in sources["kernel_contract"]
            and "RunResidualGmmStage(uint32_t stageId) const" in sources["kernel_contract"]
            and "RunMixedEpilogueStages()" in sources["kernel_contract"]
        ),
        "kernel_records_mixed_epilogue_contracts": (
            "SVDQMixedEpilogueContract" in sources["kernel_contract"]
            and "MixedEpilogueContract(uint32_t epilogueId)" in sources["kernel_contract"]
            and "MixedEpilogueReady(uint32_t epilogueId)" in sources["kernel_contract"]
            and "SVDQ_STAGE_MIXED_EPILOGUE_1" in sources["kernel_contract"]
            and "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE" in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_1" in sources["kernel_contract"]
            and "SVDQ_REGION_PROJECTION_1" in sources["kernel_contract"]
            and "SVDQ_REGION_HIDDEN" in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_2" in sources["kernel_contract"]
            and "SVDQ_REGION_PROJECTION_2" in sources["kernel_contract"]
            and "SVDQ_REGION_PEER_OUTPUT" in sources["kernel_contract"]
        ),
        "kernel_mixed_epilogue_launch_descriptor_recorded": (
            "SVDQ_MIXED_EPILOGUE_COUNT = 2" in sources["kernel_tiling"]
            and "struct SVDQMixedEpilogueShape" in sources["kernel_tiling"]
            and "SVDQMixedEpilogueShape mixedEpilogueShapes[SVDQ_MIXED_EPILOGUE_COUNT]" in sources["kernel_tiling"]
            and "SetMixedEpilogueShape" in sources["host_tiling"]
            and "BuildMixedEpilogueShapeTable" in sources["host_tiling"]
            and "BuildMixedEpilogueShapeTable(tilingData)" in sources["host_tiling"]
            and "epilogue.swigluLimit = tilingData->info.swigluLimit" in sources["host_tiling"]
            and "epilogue.appliesSwiGLU = appliesSwiGLU" in sources["host_tiling"]
            and "SVDQMixedEpilogueLaunch" in sources["kernel_contract"]
            and "MixedEpilogueShape(uint32_t epilogueId)" in sources["kernel_contract"]
            and "BuildMixedEpilogueLaunch(uint32_t epilogueId)" in sources["kernel_contract"]
            and "BuildMixedEpilogueLaunch(epilogueId)" in sources["kernel_contract"]
            and "shape.gateColumnOffset == 0" in sources["kernel_contract"]
            and "shape.upColumnOffset == tilingData_.info.intermediateSize" in sources["kernel_contract"]
            and "shape.gateColumnOffset == SVDQ_INVALID_ID" in sources["kernel_contract"]
            and "shape.upColumnOffset == SVDQ_INVALID_ID" in sources["kernel_contract"]
        ),
        "kernel_mixed_output_epilogue_scalar_execution_enabled": (
            "RunMixedOutputEpilogueStage(const SVDQMixedEpilogueLaunch& launch) const"
            in sources["kernel_contract"]
            and "if (launch.appliesSwiGLU)" in sources["kernel_contract"]
            and "return RunMixedOutputEpilogueStage(launch);" in sources["kernel_contract"]
            and "LoadMixedEpilogueResidualBF16(" in sources["kernel_contract"]
            and "LoadMixedEpilogueLowRankBF16(" in sources["kernel_contract"]
            and "StoreMixedEpilogueOutputBF16(" in sources["kernel_contract"]
            and "launch.residualColumns != launch.outputColumns" in sources["kernel_contract"]
            and "launch.lowRankColumns != launch.outputColumns" in sources["kernel_contract"]
            and "const uint64_t outputElements = static_cast<uint64_t>(launch.m) * launch.outputColumns"
            in sources["kernel_contract"]
            and "const float residual = static_cast<float>(LoadMixedEpilogueResidualBF16(launch, row, column))"
            in sources["kernel_contract"]
            and "const float lowRank = static_cast<float>(LoadMixedEpilogueLowRankBF16(launch, row, column))"
            in sources["kernel_contract"]
            and "StoreMixedEpilogueOutputBF16(launch, row, column, static_cast<bfloat16_t>(residual + lowRank))"
            in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_2, offset, routedRows * hiddenSize * FP32_BYTES" in sources[
                "host_tiling"
            ]
        ),
        "kernel_mixed_swiglu_epilogue_scalar_execution_enabled": (
            "RunMixedSwiGLUEpilogueStage(const SVDQMixedEpilogueLaunch& launch) const"
            in sources["kernel_contract"]
            and "return RunMixedSwiGLUEpilogueStage(launch);" in sources["kernel_contract"]
            and "launch.residualColumns != launch.outputColumns * 2" in sources["kernel_contract"]
            and "launch.lowRankColumns != launch.outputColumns * 2" in sources["kernel_contract"]
            and "launch.gateColumnOffset != 0" in sources["kernel_contract"]
            and "launch.upColumnOffset != launch.outputColumns" in sources["kernel_contract"]
            and "const uint32_t gateColumn = launch.gateColumnOffset + column" in sources["kernel_contract"]
            and "const uint32_t upColumn = launch.upColumnOffset + column" in sources["kernel_contract"]
            and "const float gate =" in sources["kernel_contract"]
            and "const float up =" in sources["kernel_contract"]
            and "SiluFloat(gate) * up" in sources["kernel_contract"]
            and "SiluFloat(float value) const" in sources["kernel_contract"]
            and "ExpApproxFloat(-value)" in sources["kernel_contract"]
            and "SVDQ_REGION_ACCUMULATOR_1, offset, routedRows * gateUpSize * FP32_BYTES" in sources[
                "host_tiling"
            ]
        ),
        "kernel_mixed_output_epilogue_aiv_execution_enabled": (
            "SVDQ_MIXED_EPILOGUE_VECTOR_TILE = 64" in sources["kernel_contract"]
            and "SVDQ_MIXED_EPILOGUE_UB_BYTES = 196352" in sources["kernel_contract"]
            and "RunMixedOutputEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)"
            in sources["kernel_contract"]
            and "if (g_coreType == AIC)" in sources["kernel_contract"]
            and "GlobalTensor<float> residualGm" in sources["kernel_contract"]
            and "GlobalTensor<bfloat16_t> lowRankGm" in sources["kernel_contract"]
            and "GlobalTensor<bfloat16_t> outputGm" in sources["kernel_contract"]
            and "CopyInMixedEpilogueFp32(residual, residualGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "CopyInMixedEpilogueBf16(lowRankBf16, lowRankGm, offset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "Add(residual, residual, lowRank, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "CopyOutMixedEpilogueBf16(outputGm, offset, outputBf16, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "return RunMixedOutputEpilogueAIV(launch);" in sources["kernel_contract"]
        ),
        "kernel_mixed_swiglu_epilogue_aiv_execution_enabled": (
            "RunMixedSwiGLUEpilogueAIV(const SVDQMixedEpilogueLaunch& launch)" in sources["kernel_contract"]
            and "launch.gateColumnOffset != 0" in sources["kernel_contract"]
            and "launch.upColumnOffset != launch.outputColumns" in sources["kernel_contract"]
            and "const uint32_t gateOffset = row * launch.residualColumns + launch.gateColumnOffset + column"
            in sources["kernel_contract"]
            and "const uint32_t upOffset = row * launch.residualColumns + launch.upColumnOffset + column"
            in sources["kernel_contract"]
            and "CopyInMixedEpilogueFp32(gate, residualGm, gateOffset,"
            in sources["kernel_contract"]
            and "CopyInMixedEpilogueBf16(lowRankGateBf16, lowRankGm, gateOffset,"
            in sources["kernel_contract"]
            and "CopyInMixedEpilogueFp32(up, residualGm, upOffset,"
            in sources["kernel_contract"]
            and "CopyInMixedEpilogueBf16(lowRankUpBf16, lowRankGm, upOffset, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)"
            in sources["kernel_contract"]
            and "Exp(tmp, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)" in sources["kernel_contract"]
            and "Div(hidden, gate, tmp, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)" in sources["kernel_contract"]
            and "Mul(hidden, hidden, up, SVDQ_MIXED_EPILOGUE_VECTOR_TILE)" in sources["kernel_contract"]
            and "CopyOutMixedEpilogueBf16(outputGm, outputOffset, hiddenBf16,"
            in sources["kernel_contract"]
            and "return RunMixedSwiGLUEpilogueAIV(launch);" in sources["kernel_contract"]
        ),
        "kernel_mixed_epilogue_scalar_helpers_absent": (
            "RunMixedOutputEpilogueStage" not in sources["kernel_contract"]
            and "RunMixedSwiGLUEpilogueStage" not in sources["kernel_contract"]
            and "LoadMixedEpilogueResidualBF16" not in sources["kernel_contract"]
            and "LoadMixedEpilogueLowRankBF16" not in sources["kernel_contract"]
            and "StoreMixedEpilogueOutputBF16" not in sources["kernel_contract"]
            and "SiluFloat" not in sources["kernel_contract"]
            and "ExpApproxFloat" not in sources["kernel_contract"]
        ),
        "kernel_records_final_combine_contract": (
            "SVDQFinalCombineContract" in sources["kernel_contract"]
            and "FinalCombineContract() const" in sources["kernel_contract"]
            and "FinalCombineReady() const" in sources["kernel_contract"]
            and "SVDQ_STAGE_UNPERMUTE_COMBINE" in sources["kernel_contract"]
            and "SVDQ_REGION_PEER_OUTPUT" in sources["kernel_contract"]
            and "SVDQ_REGION_EXPANDED_ROW_IDX" in sources["kernel_contract"]
            and "SVDQFinalCombineLaunch launch = BuildFinalCombineLaunch()" in sources["kernel_contract"]
            and "launch.output != nullptr" in sources["kernel_contract"]
            and "launch.expertId != nullptr" in sources["kernel_contract"]
            and "launch.probs != nullptr" in sources["kernel_contract"]
        ),
        "kernel_final_combine_launch_descriptor_recorded": (
            "struct SVDQFinalCombineShape" in sources["kernel_tiling"]
            and "SVDQFinalCombineShape finalCombineShape" in sources["kernel_tiling"]
            and "SVDQFinalCombineTiling finalCombine" in sources["kernel_tiling"]
            and "MoeTokenUnpermuteTilingData moeTokenUnpermuteTilingData" in sources["kernel_tiling"]
            and "BuildFinalCombineShape" in sources["host_tiling"]
            and "BuildFinalCombineShape(tilingData)" in sources["host_tiling"]
            and "BuildFinalCombineTiling" in sources["host_tiling"]
            and "BuildFinalCombineTiling(tilingData)" in sources["host_tiling"]
            and "MoeTokenUnpermuteTiling(info.m * info.topK, info.hiddenSize, info.topK" in sources["host_tiling"]
            and "finalCombine.stageId = SVDQ_STAGE_UNPERMUTE_COMBINE" in sources["host_tiling"]
            and "finalCombine.inputRegionId = SVDQ_REGION_PEER_OUTPUT" in sources["host_tiling"]
            and "finalCombine.routeRegionId = SVDQ_REGION_EXPANDED_ROW_IDX" in sources["host_tiling"]
            and "finalCombine.activeSlots = info.m * info.topK" in sources["host_tiling"]
            and "SVDQFinalCombineLaunch" in sources["kernel_contract"]
            and "FinalCombineShape() const" in sources["kernel_contract"]
            and "BuildFinalCombineLaunch() const" in sources["kernel_contract"]
            and "BuildFinalCombineLaunch()" in sources["kernel_contract"]
            and "shape.activeSlots == tilingData_.info.m * tilingData_.info.topK" in sources["kernel_contract"]
            and "shape.routedRows >= shape.activeSlots" in sources["kernel_contract"]
            and "launch.expertId != nullptr" in sources["kernel_contract"]
            and "launch.probs != nullptr" in sources["kernel_contract"]
        ),
        "kernel_final_combine_official_unpermute_execution_enabled": (
            '#include "../../dispatch_ffn_combine_w4_a8/op_kernel/unpermute/moe_token_unpermute.h"'
            in sources["kernel_contract"]
            and "KernelMoeTokenUnpermute<bfloat16_t, int32_t, float, true>" in sources["kernel_contract"]
            and "kernelMoeTokenUnpermuteOp.Init(launch.input, launch.routeIndex, launch.probs, launch.output"
            in sources["kernel_contract"]
            and "&tilingData_.finalCombine.moeTokenUnpermuteTilingData" in sources["kernel_contract"]
            and "kernelMoeTokenUnpermuteOp.Process()" in sources["kernel_contract"]
            and "return true;" in sources["kernel_contract"]
        ),
        "kernel_final_combine_scalar_execution_enabled": (
            "LoadFinalCombineRouteIndex(" in sources["kernel_contract"]
            and "LoadFinalCombineProb(" in sources["kernel_contract"]
            and "LoadFinalCombineInput(" in sources["kernel_contract"]
            and "StoreFinalCombineOutput(" in sources["kernel_contract"]
            and "AccumulateFinalCombineOutput(" in sources["kernel_contract"]
            and "const uint32_t slotBase = tokenIndex * launch.topK" in sources["kernel_contract"]
            and "for (uint32_t topKOffset = 0; topKOffset < launch.topK; ++topKOffset)" in sources[
                "kernel_contract"
            ]
            and "const int32_t routedRow = LoadFinalCombineRouteIndex(launch, slot)" in sources["kernel_contract"]
            and "LoadFinalCombineInput(launch, static_cast<uint32_t>(routedRow), hiddenOffset)) * probability"
            in sources["kernel_contract"]
            and "const uint64_t outputElements = static_cast<uint64_t>(launch.m) * launch.hiddenSize" in sources[
                "kernel_contract"
            ]
            and "for (uint64_t elementIndex = coreIdx; elementIndex < outputElements; elementIndex += coreCount)"
            in sources["kernel_contract"]
            and "StoreFinalCombineOutput(launch, tokenIndex, hiddenOffset, static_cast<bfloat16_t>(combined))"
            in sources["kernel_contract"]
            and "return true;" in sources["kernel_contract"]
        ),
        "kernel_final_combine_scalar_helpers_absent": (
            "LoadFinalCombineRouteIndex" not in sources["kernel_contract"]
            and "LoadFinalCombineProb" not in sources["kernel_contract"]
            and "LoadFinalCombineInput" not in sources["kernel_contract"]
            and "StoreFinalCombineOutput" not in sources["kernel_contract"]
            and "AccumulateFinalCombineOutput" not in sources["kernel_contract"]
        ),
        "kernel_mixed_final_execution_dispatch_enabled": (
            "RunMixedEpilogueStages()" in sources["kernel_contract"]
            and "MixedEpilogueReady(epilogueId)" in sources["kernel_contract"]
            and "RunFinalCombine() const" in sources["kernel_contract"]
            and "FinalCombineReady()" in sources["kernel_contract"]
        ),
        "residual_stage_contract_is_residual_only": (
            "stage.residualOnly = residualOnly" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_W4A8_GMM1" in sources["host_tiling"]
            and "SVDQ_RESIDUAL_STAGE_W4A8_GMM2" in sources["host_tiling"]
        ),
        "official_w4a8_debug_option_default_off": (
            "option(SVDQ_W4A8_DEBUG_READBACK" in sources["official_w4a8_cmake"]
            and "Compile official W4A8 epilogues with FP32 GMM readback"
            in sources["official_w4a8_cmake"]
            and "OFF)" in sources["official_w4a8_cmake"]
        ),
        "official_w4a8_debug_macro_scoped": (
            "if(SVDQ_W4A8_DEBUG_READBACK)" in sources["official_w4a8_cmake"]
            and "list(APPEND _DISPATCH_FFN_W4A8_DEBUG_OPTS -DW4A8_DEBUG)"
            in sources["official_w4a8_cmake"]
            and "${_DISPATCH_FFN_W4A8_DEBUG_OPTS}" in sources["official_w4a8_cmake"]
        ),
        "official_w4a8_mixed_aic_aiv_kernel": (
            "dispatch_ffn_combine_w4_a8" in sources["official_w4a8_kernel_entry"]
            and "KERNEL_TYPE_MIX_AIC_1_2" in sources["official_w4a8_kernel_entry"]
        ),
        "official_w4a8_aic_calls_gmm1_gmm2": (
            "operator()<AscendC::AIC>" in sources["official_w4a8_kernel"]
            and "GMM1(params);" in sources["official_w4a8_kernel"]
            and "GMM2(params);" in sources["official_w4a8_kernel"]
        ),
        "official_w4a8_aiv_calls_dispatch_and_combine": (
            "operator()<AscendC::AIV>" in sources["official_w4a8_kernel"]
            and "DispatchAndCombine(params);" in sources["official_w4a8_kernel"]
        ),
        "official_w4a8_workspace_has_ptr_cgmm1_cgmm2": (
            "ptrCGMM1" in sources["official_w4a8_kernel"]
            and "ptrCGMM2" in sources["official_w4a8_kernel"]
            and "#ifdef W4A8_DEBUG" in sources["official_w4a8_kernel"]
            and "workspaceOffset += params.maxOutputSize * params.problemShape.n() * sizeof(float);"
            in sources["official_w4a8_kernel"]
            and "workspaceOffset += params.maxOutputSize * n2 * sizeof(float);"
            in sources["official_w4a8_kernel"]
        ),
        "official_w4a8_debug_output_pointer_hook": (
            "GM_ADDR ptrDebugRoutedX;" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugRoutedScale;" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM1;" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM1Hidden;" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM2;" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugRoutedX_ = nullptr" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugRoutedScale_ = nullptr" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM1_ = nullptr" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM1Hidden_ = nullptr" in sources["official_w4a8_kernel"]
            and "GM_ADDR ptrDebugGMM2_ = nullptr" in sources["official_w4a8_kernel"]
            and "ptrDebugRoutedX(ptrDebugRoutedX_)" in sources["official_w4a8_kernel"]
            and "ptrDebugRoutedScale(ptrDebugRoutedScale_)" in sources["official_w4a8_kernel"]
            and "ptrDebugGMM1(ptrDebugGMM1_)" in sources["official_w4a8_kernel"]
            and "ptrDebugGMM1Hidden(ptrDebugGMM1Hidden_)" in sources["official_w4a8_kernel"]
            and "ptrDebugGMM2(ptrDebugGMM2_)" in sources["official_w4a8_kernel"]
            and "params.ptrDebugRoutedX != nullptr" in sources["official_w4a8_kernel"]
            and "params.ptrDebugRoutedScale != nullptr" in sources["official_w4a8_kernel"]
            and "if (params.ptrDebugGMM1 != nullptr)" in sources["official_w4a8_kernel"]
            and "ptrCGMM1 = params.ptrDebugGMM1;" in sources["official_w4a8_kernel"]
            and "if (params.ptrDebugGMM1Hidden != nullptr)" in sources["official_w4a8_kernel"]
            and "ptrCGMM1Hidden = params.ptrDebugGMM1Hidden;" in sources["official_w4a8_kernel"]
            and "if (params.ptrDebugGMM2 != nullptr)" in sources["official_w4a8_kernel"]
            and "ptrCGMM2 = params.ptrDebugGMM2;" in sources["official_w4a8_kernel"]
        ),
        "official_w4a8_debug_op_surface_wired": (
            "OP_NAME SVDQW4A8DebugReadback" in sources["official_w4a8_cmake"]
            and "-DW4A8_DEBUG" in sources["official_w4a8_cmake"]
            and "svdqw4_a8_debug_readback" in sources["official_w4a8_cmake"]
            and "class SVDQW4A8DebugReadback" in sources["official_w4a8_debug_def"]
            and 'this->Output("routedXInt8")' in sources["official_w4a8_debug_def"]
            and 'this->Output("routedXScale")' in sources["official_w4a8_debug_def"]
            and 'this->Output("gmm1PostDequant")' in sources["official_w4a8_debug_def"]
            and 'this->Output("gmm1HiddenPrequant")' in sources["official_w4a8_debug_def"]
            and 'this->Output("gmm2PostDequant")' in sources["official_w4a8_debug_def"]
            and "OP_ADD(SVDQW4A8DebugReadback)" in sources["official_w4a8_debug_def"]
            and "IMPL_OP_OPTILING(SVDQW4A8DebugReadback)" in sources["official_w4a8_host_tiling"]
        ),
        "official_w4a8_debug_op_aclnn_wrapper": (
            "aclnnSVDQW4A8DebugReadbackGetWorkspaceSize"
            in sources["official_w4a8_debug_api_header"]
            and "aclnnSVDQW4A8DebugReadback(" in sources["official_w4a8_debug_api_header"]
            and "aclnnInnerSVDQW4A8DebugReadbackGetWorkspaceSize"
            in sources["official_w4a8_debug_api_wrapper"]
            and "aclnnInnerSVDQW4A8DebugReadback(" in sources["official_w4a8_debug_api_wrapper"]
            and "routedXInt8" in sources["official_w4a8_debug_api_wrapper"]
            and "routedXScale" in sources["official_w4a8_debug_api_wrapper"]
            and "gmm1PostDequant" in sources["official_w4a8_debug_api_wrapper"]
            and "gmm1HiddenPrequant" in sources["official_w4a8_debug_api_wrapper"]
            and "gmm2PostDequant" in sources["official_w4a8_debug_api_wrapper"]
        ),
        "official_w4a8_debug_kernel_reuses_official_path": (
            "extern \"C\" __global__ __aicore__ void svdqw4_a8_debug_readback"
            in sources["official_w4a8_debug_kernel_entry"]
            and "KERNEL_TYPE_MIX_AIC_1_2" in sources["official_w4a8_debug_kernel_entry"]
            and "DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true> op"
            in sources["official_w4a8_debug_kernel_entry"]
            and "routedXInt8" in sources["official_w4a8_debug_kernel_entry"]
            and "routedXScale" in sources["official_w4a8_debug_kernel_entry"]
            and "gmm1PostDequant" in sources["official_w4a8_debug_kernel_entry"]
            and "gmm1HiddenPrequant" in sources["official_w4a8_debug_kernel_entry"]
            and "gmm2PostDequant" in sources["official_w4a8_debug_kernel_entry"]
        ),
        "official_w4a8_debug_torch_adapter_registered": (
            "aclnnSVDQW4A8DebugReadback" in sources["w4a8_debug_torch_adapter"]
            and "svdq_w4a8_debug_readback" in sources["w4a8_debug_torch_adapter"]
            and 'ops.def(\n        "svdq_w4a8_debug_readback' in sources["torch_binding"]
            and 'ops.impl("svdq_w4a8_debug_readback", torch::kPrivateUse1'
            in sources["torch_binding"]
            and "svdq_w4a8_debug_readback_meta" in sources["torch_binding_meta"]
            and 'ops.impl("svdq_w4a8_debug_readback", &vllm_ascend::meta::svdq_w4a8_debug_readback_meta)'
            in sources["torch_binding_meta"]
        ),
        "official_w4a8_debug_probe_launches_real_op": (
            "torch.ops._C_ascend.svdq_w4a8_debug_readback" in sources["w4a8_debug_probe"]
            and "aclnnSVDQW4A8DebugReadback" in sources["w4a8_debug_probe"]
            and "public_grouped_matmul_used" in sources["w4a8_debug_probe"]
            and "real_checkpoint_validation" in sources["w4a8_debug_probe"]
            and "routed_x_int8_active" in sources["w4a8_debug_probe"]
            and "routed_x_scale_active" in sources["w4a8_debug_probe"]
            and "gmm1_post_dequant_active" in sources["w4a8_debug_probe"]
            and "gmm1_hidden_prequant_active" in sources["w4a8_debug_probe"]
            and "gmm2_post_dequant_active" in sources["w4a8_debug_probe"]
            and "npu_grouped_matmul" not in sources["w4a8_debug_probe"]
        ),
        "official_w4a8_kernel_binds_block_mmad_and_epilogues": (
            "using BlockMmad = Gemm::Block::BlockMmad" in sources["official_w4a8_op"]
            and "EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant"
            in sources["official_w4a8_op"]
            and "EpilogueAtlasA2W4A8PostPerTokenDequantV2" in sources["official_w4a8_op"]
            and "DispatchFFNCombineW4A8Kernel<BlockMmad" in sources["official_w4a8_op"]
        ),
        "official_w4a8_gmm1_epilogue_debug_copies_fp32": (
            "#ifdef W4A8_DEBUG" in sources["official_w4a8_gmm1_epilogue"]
            and "using CopyUbToGmGMM1 = typename TileCopyDebug::CopyUbToGmD;"
            in sources["official_w4a8_gmm1_epilogue"]
            and "layout::RowMajor layoutGMM1{1, blockN};"
            in sources["official_w4a8_gmm1_epilogue"]
            and "copyUbToGmGMM1(gmTileGMM1, ubCFp32, layoutGMM1, layoutGMM1);"
            in sources["official_w4a8_gmm1_epilogue"]
        ),
        "official_w4a8_gmm2_epilogue_debug_copies_fp32": (
            "#ifdef W4A8_DEBUG" in sources["official_w4a8_gmm2_epilogue"]
            and "copyUbToGmGMM2(gmTileGMM2, ubFp32, layoutGM, layoutUB);"
            in sources["official_w4a8_gmm2_epilogue"]
        ),
        "lowrank_helper_enabled_by_contract": "IsImplemented() const\n    {\n        return HasCompleteContract();"
        in sources["lowrank_header"],
        "lowrank_helper_uses_separate_rank_workspace": (
            "GM_ADDR rank;" in sources["lowrank_header"]
            and "args_.rank != nullptr" in sources["lowrank_header"]
            and "GM_ADDR inputBase = stageIndex == 0 ? args_.input : args_.rank;" in sources["lowrank_header"]
            and "GM_ADDR outputBase = stageIndex == 0 ? args_.rank : args_.output;" in sources["lowrank_header"]
            and "WorkspaceAddress(invocation.rankRegionId)" in sources["kernel_contract"]
            and "invocation.rankRegionId = rankRegionId;" in sources["host_tiling"]
        ),
        "lowrank_helper_stage_orders_rank_consumers": (
            "ExecuteStage(stageIndex, coreIdx, scheduledCoreCount, resource)" in sources["lowrank_header"]
            and "blockScheduler.Update(" in sources["lowrank_header"]
            and "blockScheduler.GetBlockCoord(loopIdx)" in sources["lowrank_header"]
            and "SVDQLowRankOutputTilePlan outputTilePlan" in sources["lowrank_header"]
            and "SVDQOfficialBF16Resource resource;" in sources["lowrank_header"]
            and "SVDQOfficialBF16BlockMmad blockMmad(resource);" in sources["lowrank_header"]
            and "SVDQLowRankBF16RankBlockMmad blockMmad(resource);" not in sources["lowrank_header"]
            and "StageOutputColumnTile(stage)" in sources["lowrank_header"]
            and "AscendC::SyncAll()" in sources["lowrank_header"]
            and "stage-ordered so L2 stages cannot read rank workspace" in sources["lowrank_header"]
        ),
        "host_tiling_graph_success_enabled": (
            "BuildWorkspaceMap(tilingData);" in host_tiling_source
            and "BuildSyncFlagTable(tilingData);" in host_tiling_source
            and "BuildDispatchRoutingTiling(tilingData);" in host_tiling_source
            and "BuildBF16StageShapeTable(tilingData);" in host_tiling_source
            and "BuildResidualStageShapeTable(tilingData);" in host_tiling_source
            and "BuildResidualQuantShapeTable(tilingData);" in host_tiling_source
            and "BuildResidualGmmShapeTable(tilingData);" in host_tiling_source
            and "BuildMixedEpilogueShapeTable(tilingData);" in host_tiling_source
            and "BuildFinalCombineShape(tilingData);" in host_tiling_source
            and "BuildFinalCombineTiling(tilingData);" in host_tiling_source
            and "BuildLowRankInvocationTable(tilingData);" in host_tiling_source
            and "workSpaces[0] = SVDQ_SYSTEM_WORKSPACE + info.workspaceBytes +" in host_tiling_source
            and "return ge::GRAPH_SUCCESS;" in host_tiling_source
            and "production tiling is fail-closed" not in host_tiling_source
        ),
        "lowrank_mmad_debug_readback_macro": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK"
        in sources["lowrank_header"],
        "lowrank_mmad_debug_readback_uses_fp32_l0c_to_gm": (
            "l0c_to_gm<ArchType::ASCEND_V220, DataFormatT::ND, float, float>"
            in sources["lowrank_header"]
        ),
        "lowrank_mmad_debug_readback_targets_accumulator_gm": (
            "accumulatorGm, l0C, tile.mActual, tile.nActual, tile.nRound"
            in sources["lowrank_header"]
        ),
        "lowrank_debug_runner_exists": "class SVDQLowRankDebugReadback" in sources["lowrank_header"],
        "lowrank_debug_runner_macro_gated": (
            "IsEnabled() const" in sources["lowrank_header"]
            and "#ifdef SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in sources["lowrank_header"]
        ),
        "lowrank_debug_runner_bypasses_production_is_implemented_gate": (
            "lowRankOp.Process();" in sources["lowrank_header"]
            and "lowRankOp.HasCompleteContract()" in sources["lowrank_header"]
        ),
        "lowrank_debug_kernel_exists": "svdq_low_rank_debug_readback(" in sources["lowrank_debug_kernel"],
        "lowrank_debug_kernel_uses_debug_runner": (
            "SVDQLowRankDebugReadbackKernel op" in sources["lowrank_debug_kernel"]
            and "op.Process();" in sources["lowrank_debug_kernel"]
        ),
        "lowrank_debug_kernel_exposes_readback_buffers": (
            "GM_ADDR gateUpOutput" in sources["lowrank_debug_kernel"]
            and "GM_ADDR downOutput" in sources["lowrank_debug_kernel"]
            and "GM_ADDR gateUpAccumulator" in sources["lowrank_debug_kernel"]
            and "GM_ADDR downAccumulator" in sources["lowrank_debug_kernel"]
        ),
        "lowrank_debug_kernel_uses_separate_tiling_contract": (
            "struct SVDQLowRankDebugTilingData" in sources["lowrank_debug_tiling"]
            and "SVDQFusedDownUpTiling gateUpInvocation" in sources["lowrank_debug_tiling"]
            and "SVDQFusedDownUpTiling downInvocation" in sources["lowrank_debug_tiling"]
        ),
        "lowrank_debug_kernel_preserves_branch_separation": (
            "BuildGateUpArgs() const" in sources["lowrank_debug_header"]
            and "BuildDownArgs() const" in sources["lowrank_debug_header"]
            and "runtime_.gateSvdqL2" in sources["lowrank_debug_header"]
            and "runtime_.upSvdqL2" in sources["lowrank_debug_header"]
        ),
        "lowrank_debug_op_has_compile_options": (
            "OP_NAME SVDQLowRankDebugReadback" in sources["op_cmake"]
            and "${_DISPATCH_FFN_SVDQ_LOWRANK_DEBUG_OPTS}" in lowrank_debug_compile_options
        ),
        "lowrank_debug_op_inner_aclnn_linked": (
            "OPTYPE dispatch_ffn_combine_w4_a8_svdq svdq_low_rank_debug_readback"
            in sources["op_cmake"]
            and "ACLNNTYPE aclnn_inner aclnn_inner" in sources["op_cmake"]
            and "target_sources(op_host_aclnnInner PRIVATE" in sources["op_cmake"]
            and "svdq_low_rank_debug_readback_def.cpp" in sources["op_cmake"]
        ),
        "lowrank_debug_op_registered": (
            "class SVDQLowRankDebugReadback" in sources["debug_op_def"]
            and "OP_ADD(SVDQLowRankDebugReadback)" in sources["debug_op_def"]
            and "IMPL_OP_INFERSHAPE(SVDQLowRankDebugReadback)" in sources["op_proto"]
        ),
        "lowrank_debug_op_tiling_registered": (
            "SVDQLowRankDebugReadbackTilingFunc" in sources["host_tiling"]
            and "IMPL_OP_OPTILING(SVDQLowRankDebugReadback)" in sources["host_tiling"]
            and "return ge::GRAPH_SUCCESS;" in sources["host_tiling"]
        ),
        "lowrank_debug_op_public_aclnn_wrapper": (
            "aclnnSVDQLowRankDebugReadbackGetWorkspaceSize" in sources["debug_op_api_header"]
            and "aclnnInnerSVDQLowRankDebugReadbackGetWorkspaceSize" in sources["debug_op_api_wrapper"]
            and "aclnnSVDQLowRankDebugReadback(" in sources["debug_op_api_wrapper"]
        ),
        "lowrank_debug_op_exposes_same_readback_outputs": (
            'this->Output("gateUpOutput")' in sources["debug_op_def"]
            and 'this->Output("downOutput")' in sources["debug_op_def"]
            and 'this->Output("gateUpAccumulator")' in sources["debug_op_def"]
            and 'this->Output("downAccumulator")' in sources["debug_op_def"]
        ),
        "lowrank_debug_alias_source_root": (
            "add_op_to_compiled_list()" in sources["lowrank_debug_alias_cmake"]
            and "svdq_low_rank_debug_readback(" in sources["lowrank_debug_alias_kernel"]
            and "svdq_lowrank_debug_readback.h" in sources["lowrank_debug_alias_kernel"]
        ),
        "lowrank_debug_torch_adapter_registered": (
            "svdq_low_rank_debug_readback_torch_adpt.h" in sources["torch_binding"]
            and 'ops.def(\n        "svdq_low_rank_debug_readback' in sources["torch_binding"]
            and 'ops.impl("svdq_low_rank_debug_readback", torch::kPrivateUse1' in sources["torch_binding"]
            and "aclnnSVDQLowRankDebugReadback" in sources["lowrank_debug_torch_adapter"]
        ),
        "lowrank_debug_meta_registered": (
            "svdq_low_rank_debug_readback_meta" in sources["torch_binding_meta"]
            and 'ops.impl("svdq_low_rank_debug_readback", &vllm_ascend::meta::svdq_low_rank_debug_readback_meta)'
            in sources["torch_binding_meta"]
        ),
        "lowrank_debug_probe_launches_real_op": (
            "torch.ops._C_ascend.svdq_low_rank_debug_readback" in sources["lowrank_debug_probe"]
            and "_load_validation_layer(" in sources["lowrank_debug_probe"]
            and "build_svdq_bf16_stage_reference" in sources["lowrank_debug_probe"]
        ),
        "lowrank_debug_probe_preflights_runtime_soc_package": (
            "DEBUG_OP_NAME = \"SVDQLowRankDebugReadback\"" in sources["lowrank_debug_probe"]
            and "_custom_package_debug_op_support" in sources["lowrank_debug_probe"]
            and "_runtime_soc(args.device_id)" in sources["lowrank_debug_probe"]
            and "_package_supports_runtime_soc(" in sources["lowrank_debug_probe"]
            and "binary_info_config.json" in sources["lowrank_debug_probe"]
        ),
        "lowrank_debug_install_validator_checks_schema_symbols_and_soc": (
            "REQUIRED_OPAPI_SYMBOLS" in sources["lowrank_debug_install_validate"]
            and "REQUIRED_PRODUCTION_OPAPI_SYMBOLS" in sources["lowrank_debug_install_validate"]
            and "svdq_low_rank_debug_readback" in sources["lowrank_debug_install_validate"]
            and "dispatch_ffn_combine_w4a8_svdq" in sources["lowrank_debug_install_validate"]
            and "aclnnInnerSVDQLowRankDebugReadback" in sources["lowrank_debug_install_validate"]
            and "aclnnInnerDispatchFFNCombineW4A8SVDQ" in sources["lowrank_debug_install_validate"]
            and "_custom_package_debug_op_support()" in sources["lowrank_debug_install_validate"]
            and "require_runtime_soc_support" in sources["lowrank_debug_install_validate"]
        ),
        "svdq_ops_in_a2_a3_aclnn_package": _svdq_ops_selected_in_build_branch(a2_aclnn_branch)
        and _svdq_ops_selected_in_build_branch(a3_aclnn_branch),
    }


def validate_manifest_sources(manifest: dict[str, Any], repo_root: Path = REPO_ROOT) -> None:
    sources = _read_sources(repo_root)
    for factor in manifest["factor_abi"]:
        if factor["name"] not in sources["kernel_tiling"] or factor["name"] not in sources["host_tiling"]:
            raise ValueError(f"factor {factor['name']} is not present in host tiling and tiling header.")
        if f"case {factor['name']}:" not in sources["kernel_contract"]:
            raise ValueError(f"factor {factor['name']} is not handled by FactorAddress().")

    for region in manifest["workspace_regions"]:
        for source_name in ("host_tiling", "kernel_tiling"):
            if region["name"] not in sources[source_name]:
                raise ValueError(f"workspace region {region['name']} missing from {source_name}.")
        if region["dtype"] not in sources["kernel_tiling"]:
            raise ValueError(f"workspace dtype {region['dtype']} missing from tiling header.")
        if (
            region["producer_stage"] not in sources["host_tiling"]
            or region["consumer_stage"] not in sources["host_tiling"]
        ):
            raise ValueError(f"workspace region {region['name']} stages missing from host tiling.")

    for flag in manifest["sync_flags"]:
        for token in (flag["name"], flag["producer_stage"], flag["consumer_stage"], flag["workspace_region"]):
            if token not in sources["host_tiling"] or token not in sources["kernel_tiling"]:
                raise ValueError(f"sync flag token {token} missing from source tables.")

    for stage in manifest["bf16_stages"]:
        for token in (stage["name"], stage["factor"], stage["input_region"], stage["output_region"]):
            if token != "SVDQ_INVALID_ID" and token not in sources["host_tiling"]:
                raise ValueError(f"BF16 stage token {token} missing from host tiling.")
        if f"case {stage['name']}:" not in sources["kernel_contract"]:
            raise ValueError(f"BF16 stage {stage['name']} missing from kernel contract switch.")

    for stage in manifest["residual_stages"]:
        for source_name in ("host_tiling", "kernel_tiling"):
            for token in (stage["name"], stage["input_region"], stage["scale_region"], stage["output_region"]):
                if token != "SVDQ_INVALID_ID" and token not in sources[source_name]:
                    raise ValueError(f"residual stage token {token} missing from {source_name}.")
        if f"case {stage['name']}:" not in sources["kernel_contract"]:
            raise ValueError(f"residual stage {stage['name']} missing from kernel contract switch.")

    for launch in manifest["residual_quant_launches"]:
        for token in (
            launch["name"],
            launch["input_region"],
            launch["activation_scale_region"],
            launch["output_region"],
        ):
            if token not in sources["host_tiling"] or token not in sources["kernel_contract"]:
                raise ValueError(f"residual quant launch token {token} missing from source.")
        for token in ("SVDQResidualQuantShape", "BuildResidualQuantLaunch", "ResidualQuantLaunchReady"):
            if token not in sources["kernel_contract"]:
                raise ValueError(f"residual quant launch helper {token} missing from kernel contract.")

    for launch in manifest["residual_gmm_launches"]:
        for token in (
            launch["name"],
            launch["input_region"],
            launch["activation_scale_region"],
            launch["output_region"],
        ):
            if token not in sources["host_tiling"] or token not in sources["kernel_contract"]:
                raise ValueError(f"residual GMM launch token {token} missing from source.")
        for token in ("SVDQResidualGmmShape", "BuildResidualGmmLaunch", "ResidualGmmLaunchReady"):
            if token not in sources["kernel_contract"]:
                raise ValueError(f"residual GMM launch helper {token} missing from kernel contract.")

    for launch in manifest["mixed_epilogue_launches"]:
        for token in (
            launch["name"],
            launch["residual_region"],
            launch["lowrank_region"],
            launch["scale_region"],
            launch["output_region"],
        ):
            if token not in sources["host_tiling"] or token not in sources["kernel_contract"]:
                raise ValueError(f"mixed epilogue launch token {token} missing from source.")
        for token in ("SVDQMixedEpilogueShape", "SVDQMixedEpilogueLaunch", "BuildMixedEpilogueLaunch"):
            if token not in sources["kernel_contract"]:
                raise ValueError(f"mixed epilogue launch helper {token} missing from kernel contract.")

    final_launch = manifest["final_combine_launch"]
    for token in (
        final_launch["name"],
        final_launch["input_region"],
        final_launch["route_region"],
    ):
        if token not in sources["host_tiling"] or token not in sources["kernel_contract"]:
            raise ValueError(f"final combine launch token {token} missing from source.")
    for token in ("SVDQFinalCombineShape", "SVDQFinalCombineLaunch", "BuildFinalCombineLaunch"):
        if token not in sources["kernel_contract"]:
            raise ValueError(f"final combine launch helper {token} missing from kernel contract.")

    for invocation in manifest["lowrank_invocations"]:
        for token in (
            invocation["name"],
            invocation["input_region"],
            invocation["rank_region"],
            invocation["output_region"],
            invocation["down_factor"],
            invocation["up_factor"],
            invocation["second_up_factor"],
            invocation["accumulator_region"],
        ):
            if token != "SVDQ_INVALID_ID" and token not in sources["host_tiling"]:
                raise ValueError(f"low-rank invocation token {token} missing from host tiling.")

    expected_false_source_proofs = set()
    if manifest["production_fail_closed"]["host_tiling_returns_graph_failed"]:
        expected_false_source_proofs.update(
            {
                "host_tiling_graph_success_enabled",
                "kernel_residual_gmm_scalar_execution_enabled",
                "kernel_residual_hidden_quant_scalar_execution_enabled",
                "kernel_residual_gmm_official_full_lifecycle_execution_enabled",
                "kernel_mixed_output_epilogue_scalar_execution_enabled",
                "kernel_mixed_swiglu_epilogue_scalar_execution_enabled",
                "kernel_final_combine_scalar_execution_enabled",
            }
        )
    if any(
        not passed and name not in expected_false_source_proofs
        for name, passed in manifest["source_proof"].items()
    ):
        failed = [
            name
            for name, passed in manifest["source_proof"].items()
            if not passed and name not in expected_false_source_proofs
        ]
        raise ValueError(f"source proof failed: {failed}.")
    forbidden = ("gate_up" + "_svdq_l2", "gateUp" + "SvdqL2")
    for source_name, source in sources.items():
        for token in forbidden:
            if token in source:
                raise ValueError(f"forbidden token {token!r} found in {source_name}.")

    readback = manifest["debug_readback_contract"]
    for proof_name in readback["source_proof"]:
        if not manifest["source_proof"].get(proof_name):
            raise ValueError(f"debug readback source proof failed: {proof_name}.")

    debug_launch = manifest["debug_launch_contract"]
    for proof_name in debug_launch["source_proof"]:
        if not manifest["source_proof"].get(proof_name):
            raise ValueError(f"debug launch source proof failed: {proof_name}.")

    w4a8_readback = manifest["w4a8_debug_readback_contract"]
    for proof_name in w4a8_readback["source_proof"]:
        if not manifest["source_proof"].get(proof_name):
            raise ValueError(f"W4A8 debug readback source proof failed: {proof_name}.")


def _zero_error(error: dict[str, Any]) -> bool:
    return (
        bool(error.get("actual_finite"))
        and bool(error.get("expected_finite"))
        and bool(error.get("diff_finite"))
        and float(error.get("max_abs", float("inf"))) == 0.0
        and float(error.get("mean_abs", float("inf"))) == 0.0
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _stage2_2_official_gmm2_gate(evidence_dir: Path) -> dict[str, Any]:
    evidence_path = evidence_dir / STAGE2_2_GMM2_OFFICIAL_EVIDENCE
    gate: dict[str, Any] = {
        "name": "stage2_2_modified_hidden_official_w4a8_gmm2",
        "evidence_path": str(evidence_path),
        "historical_evidence_path": str(evidence_dir / STAGE2_2_GMM2_HISTORICAL_EVIDENCE),
        "required_probe": "svdq_w4a8_gmm2_from_mixed_hidden_probe",
        "required_summary_name": str(STAGE2_2_GMM2_OFFICIAL_EVIDENCE),
        "evidence_found": evidence_path.exists(),
        "passed": False,
        "appendix_reference": STAGE2_2_GMM2_OFFICIAL_PATH_APPENDIX,
        "source_of_truth": "official dispatch_ffn_combine_w4_a8 GMM2 AIC/AIV lifecycle",
        "stage2_3_and_later_blocked": True,
        "production_tiling_enable_allowed": False,
        "public_grouped_matmul_allowed": False,
        "binding_revision": (
            "stage2_appendix_gmm2_official_path_requires_explicit_official_lifecycle_state_table_"
            "and_gate_a_b_c_evidence"
        ),
        "required_official_path_correction_revision": STAGE2_2_GMM2_OFFICIAL_PATH_RESET_REVISION,
        "required_current_authoritative_requirements_revision": (
            STAGE2_2_GMM2_OFFICIAL_PATH_CURRENT_REQUIREMENTS_REVISION
        ),
        "effective_status_for_production": "pending_current_recheck",
        "reason": (
            "The Stage 2 appendix keeps modified-hidden official W4A8 GMM2 in FAIL / IN PROGRESS "
            "until evidence records the official-vs-debug state table, exact routed-row identity, "
            "Gate A input boundary, Gate B AIC raw/D2 output, and Gate C BlockEpilogue2/CombineV2 "
            "post-dequant readback under the current authoritative appendix state. Existing current-recheck "
            "summaries are retained only as historical evidence unless regenerated for the current "
            "official-lifecycle requirements revision."
        ),
        "required_gate_a": (
            "same routed-row identity with canonical hidden BF16, hidden INT8, packed INT4, "
            "hidden scale, expert token nums, prefix sums, row offsets, and W2 metadata"
        ),
        "required_gate_b": (
            "official GMM2 AIC raw/D2 output finite, nonzero, and reference-matched at the "
            "exact BlockEpilogue2 source boundary"
        ),
        "required_gate_c": (
            "official BlockEpilogue2/CombineV2 FP32 post-dequant output finite, nonzero, "
            "and strict-reference matched"
        ),
    }
    if not evidence_path.exists():
        gate["status"] = "fail_in_progress"
        gate["evidence_status"] = "missing_evidence"
        gate["reason"] = "Current Stage 2.2 official GMM2 recheck evidence has not been generated."
        gate["historical_passed_under_superseded_contract"] = False
        return gate

    try:
        summary = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["status"] = "fail_in_progress"
        gate["evidence_status"] = "invalid_evidence_json"
        gate["evidence_error"] = f"{type(exc).__name__}: {exc}"
        gate["historical_passed_under_superseded_contract"] = False
        return gate

    stage = summary.get("stage", {})
    checks = stage.get("checks", {})
    lifecycle = stage.get("official_lifecycle_debug_contract", {})
    routing = stage.get("routing_identity", {})
    shape = stage.get("shape", {})
    gmm2 = stage.get("gmm2", {})
    required_status_fields = {
        name: bool(checks.get(name))
        for name in (
            "official_gmm2_entry_reached",
            "official_gmm2_loop_stats_valid",
            "official_gmm2_active_tile_count_nonzero",
            "official_gmm2_aic_raw_output_finite",
            "official_gmm2_aic_raw_output_nonzero",
            "official_gmm2_aic_reference_passed",
            "official_gmm2_accumulator_int32_reference_passed",
            "official_gmm2_c2v_handoff_verified",
            "official_gmm2_post_dequant_finite",
            "official_gmm2_post_dequant_nonzero",
            "official_gmm2_post_dequant_reference_passed",
            "official_gmm2_numerical_gate_passed",
        )
    }
    gate_a_flags = {
        name: bool(checks.get(name))
        for name in (
            "gate_a_input_boundary_passed",
            "hidden_packed_exact",
            "hidden_packed_mismatch_count_zero",
            "hidden_post_override_readback_exact",
            "hidden_scale_post_override_readback_exact",
            "hidden_scale_finite",
            "hidden_scale_nonzero",
            "canonical_hidden_finite",
            "canonical_hidden_nonzero",
        )
    }
    routing_flags = {
        "expert_token_total_matches_active_rows": bool(
            routing.get("expert_token_total_matches_active_rows")
        ),
        "same_source_token_payload_across_topk_slots_proven": bool(
            routing.get("manifest_scope", {}).get("same_source_token_payload_across_topk_slots_proven")
        ),
        "reference_group_counts_match_expert_token_nums": bool(
            routing.get("reference_group_counts_match_expert_token_nums")
        ),
        "local_expert_id_equals_global_expert_id": bool(
            routing.get("tp_ep_mapping", {}).get("local_expert_id_equals_global_expert_id")
        ),
    }
    lifecycle_flags = {
        "uses_official_dispatch_and_combine_lifecycle": (
            lifecycle.get("mode") == "official DispatchAndCombine lifecycle with external hidden/scale overlay"
        ),
        "gmm2_only_path_disabled": lifecycle.get("gmm2_only_from_packed") is False,
        "custom_signal_substitute_absent": not bool(
            lifecycle.get("forbidden_paths", {}).get("custom_v2c_or_c2v_signal_substitute_used", False)
        ),
    }
    diagnostic_flags = {
        "loop_count_positive": _positive_int(checks.get("official_gmm2_loop_count")),
        "active_tile_count_positive": _positive_int(checks.get("official_gmm2_active_tile_count")),
        "int32_accumulator_reference_passed": bool(
            gmm2.get("int32_accumulator_readback_reference", {}).get("passed")
        ),
        "int32_accumulator_nonzero": bool(
            gmm2.get("int32_accumulator_readback_reference", {}).get("actual_nonzero")
        ),
        "actual_d2_post_dequant_reconstruction_passed": bool(
            gmm2.get("actual_d2_post_dequant_reconstruction", {}).get("passed_with_gate_tolerance")
        ),
    }
    shape_flags = {
        "top_k_is_1": int(shape.get("top_k", -1)) == 1,
        "active_rows_positive": int(shape.get("active_rows", 0)) > 0,
        "hidden_size_is_2048": int(shape.get("hidden_size", -1)) == 2048,
        "intermediate_size_is_512": int(shape.get("intermediate_size", -1)) == 512,
    }
    appendix_revision = stage.get("appendix_gmm2_official_path", {})
    observed_reset_revision = appendix_revision.get("official_path_correction_revision")
    post_reset_revision_passed = observed_reset_revision == STAGE2_2_GMM2_OFFICIAL_PATH_RESET_REVISION
    observed_current_requirements_revision = appendix_revision.get(
        "current_authoritative_requirements_revision"
    )
    current_requirements_revision_passed = (
        observed_current_requirements_revision
        == STAGE2_2_GMM2_OFFICIAL_PATH_CURRENT_REQUIREMENTS_REVISION
    )
    appendix_revision_flags = {
        "official_path_correction_revision_matches": post_reset_revision_passed,
        "current_authoritative_requirements_revision_matches": current_requirements_revision_passed,
        "official_vs_debug_state_table_complete": bool(
            appendix_revision.get("official_vs_debug_state_table_complete")
        ),
        "gate_a_routing_identity_complete": bool(
            appendix_revision.get("gate_a_routing_identity_complete")
        ),
        "gate_a_prefix_and_padded_row_evidence_complete": bool(
            appendix_revision.get("gate_a_prefix_and_padded_row_evidence_complete")
        ),
        "gate_b_exact_aic_raw_or_d2_boundary_identified": bool(
            appendix_revision.get("gate_b_exact_aic_raw_or_d2_boundary_identified")
        ),
        "gate_b_aic_raw_output_reference_passed": bool(
            appendix_revision.get("gate_b_aic_raw_output_reference_passed")
        ),
        "gate_c_validated_after_gate_b_nonzero": bool(
            appendix_revision.get("gate_c_validated_after_gate_b_nonzero")
        ),
        "active_failure_all_zero_post_dequant_resolved": bool(
            appendix_revision.get("active_failure_all_zero_post_dequant_resolved")
        ),
    }
    historical_passed_under_superseded_contract = bool(
        summary.get("passed")
        and stage.get("passed")
        and not bool(summary.get("skipped"))
        and bool(stage.get("real_checkpoint_validation"))
        and not bool(stage.get("public_grouped_matmul_used", False))
        and bool(stage.get("production_svdq_host_tiling_fail_closed", True))
        and all(required_status_fields.values())
        and all(gate_a_flags.values())
        and all(routing_flags.values())
        and all(lifecycle_flags.values())
        and all(diagnostic_flags.values())
        and all(shape_flags.values())
        and all(
            value
            for name, value in appendix_revision_flags.items()
            if name != "current_authoritative_requirements_revision_matches"
        )
    )
    passed = bool(
        historical_passed_under_superseded_contract
        and appendix_revision_flags["current_authoritative_requirements_revision_matches"]
    )
    missing_appendix_revision = evidence_path.exists() and not all(appendix_revision_flags.values())
    gate.update(
        {
            "status": "passed" if passed else "fail_in_progress",
            "evidence_status": (
                "current_recheck_passed"
                if passed
                else (
                    "current_recheck_missing_post_reset_official_path_correction_revision"
                    if not post_reset_revision_passed
                    else (
                        "current_recheck_missing_appendix_gmm2_official_path_revision_fields"
                        if missing_appendix_revision
                        else "current_recheck_failed"
                    )
                )
            ),
            "passed": passed,
            "historical_passed_under_superseded_contract": historical_passed_under_superseded_contract,
            "observed_official_path_correction_revision": observed_reset_revision,
            "observed_current_authoritative_requirements_revision": (
                observed_current_requirements_revision
            ),
            "summary_passed": bool(summary.get("passed")),
            "stage_passed": bool(stage.get("passed")),
            "shape": shape,
            "required_status_fields": required_status_fields,
            "required_gate_a_flags": gate_a_flags,
            "required_routing_flags": routing_flags,
            "required_lifecycle_flags": lifecycle_flags,
            "required_diagnostic_flags": diagnostic_flags,
            "required_shape_flags": shape_flags,
            "required_appendix_gmm2_official_path_revision_flags": appendix_revision_flags,
            "production_tiling_fail_closed_in_evidence": bool(
                stage.get("production_svdq_host_tiling_fail_closed", True)
            ),
        }
    )
    gate["stage2_3_and_later_blocked"] = not passed
    gate["effective_status_for_production"] = "stage2_2_gate_passed" if passed else "fail_in_progress"
    return gate


def _stage2_3_real_composition_gate(evidence_dir: Path, stage2_2_passed: bool) -> dict[str, Any]:
    evidence_path = evidence_dir / STAGE2_3_REAL_COMPOSITION_EVIDENCE
    gate: dict[str, Any] = {
        "name": "stage2_3_real_checkpoint_topk8_composition",
        "evidence_path": str(evidence_path),
        "required_probe": "svdq_w4a8_tap_mixed_epilogue_probe",
        "required_summary_name": str(STAGE2_3_REAL_COMPOSITION_EVIDENCE),
        "evidence_found": evidence_path.exists(),
        "passed": False,
        "production_scope": "isolated_single_device_real_checkpoint_gate",
        "production_tiling_enable_allowed": False,
        "public_grouped_matmul_used": False,
    }
    if not evidence_path.exists():
        gate["status"] = "missing_evidence"
        gate["reason"] = "Stage 2.3 top-k 8 real-device summary has not been generated in the evidence directory."
        return gate

    try:
        summary = json.loads(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        gate["status"] = "invalid_evidence_json"
        gate["reason"] = f"{type(exc).__name__}: {exc}"
        return gate

    stage = summary.get("stage", {})
    shape = stage.get("shape", {})
    stage_passed = stage.get("stage_passed", {})
    same_route = stage.get("stage2_3_same_routing_manifest", {})
    same_route_checks = same_route.get("checks", {})
    gmm2 = stage.get("official_gmm2_from_svdq_hidden", {})
    gmm2_checks = gmm2.get("checks", {})
    gmm2_error = gmm2.get("unfused_reference", {}).get("error", {})
    stage_errors = stage.get("stage_errors", {})
    down_error = stage_errors.get("down_mixed", {})
    out_error = stage_errors.get("out_bf16", {})
    final_combine_error = stage_errors.get("final_combine_output", {})
    final_combine = stage.get("real_final_combine", {})
    required_stage_flags = {
        name: bool(stage_passed.get(name))
        for name in (
            "first_mixed_epilogue",
            "official_gmm2_from_svdq_hidden",
            "gate_mixed",
            "up_mixed",
            "hidden_bf16",
            "hidden_scale",
            "hidden_q",
            "down_mixed",
            "out_bf16",
            "final_combine_output",
            "same_routing_identity",
        )
    }
    required_same_route_flags = {
        name: bool(same_route_checks.get(name))
        for name in (
            "expert_token_total_matches_active_rows",
            "same_canonical_hidden_feeds_svdq_down_and_w4a8_hidden_quant",
            "official_gmm2_output_feeds_final_mixed_residual_down",
            "final_mixed_output_is_final_combine_input",
            "final_combine_consumes_mixed_down_peer_output",
            "final_combine_output_validated",
            "same_source_token_payload_across_topk_slots_proven",
        )
    }
    required_gmm2_flags = {
        name: bool(gmm2_checks.get(name))
        for name in (
            "official_gmm2_entry_reached",
            "gate_a_input_boundary_passed",
            "official_gmm2_post_dequant_finite",
            "official_gmm2_post_dequant_nonzero",
            "official_gmm2_post_dequant_reference_passed",
            "official_gmm2_numerical_gate_passed",
        )
    }
    shape_passed = {
        "num_tokens_is_4": int(shape.get("num_tokens", -1)) == 4,
        "top_k_is_8": int(shape.get("top_k", -1)) == 8,
        "active_rows_is_32": int(shape.get("active_rows", -1)) == 32,
        "hidden_size_is_2048": int(shape.get("hidden_size", -1)) == 2048,
        "intermediate_size_is_512": int(shape.get("intermediate_size", -1)) == 512,
    }
    exact_numerics = {
        "official_gmm2_error_zero": _zero_error(gmm2_error),
        "final_down_mixed_error_zero": _zero_error(down_error),
        "final_out_bf16_error_zero": _zero_error(out_error),
        "final_combine_output_error_zero": _zero_error(final_combine_error),
    }
    final_combine_flags = {
        "real_final_combine_present": bool(final_combine),
        "real_final_combine_passed": bool(final_combine.get("passed")),
        "real_final_combine_uses_token_unpermute": (
            final_combine.get("official_surface") == "torch_npu.npu_moe_token_unpermute"
        ),
        "real_final_combine_uses_reference": (
            final_combine.get("reference") == "build_svdq_final_combine_reference"
        ),
        "real_final_combine_uses_official_index_semantics": (
            final_combine.get("index_semantics") == "official_token_major_output_slots_to_permuted_input_rows"
        ),
    }
    historical_passed = bool(
        summary.get("passed")
        and stage.get("passed")
        and all(required_stage_flags.values())
        and all(required_same_route_flags.values())
        and all(required_gmm2_flags.values())
        and all(shape_passed.values())
        and all(exact_numerics.values())
        and all(final_combine_flags.values())
        and not bool(stage.get("public_grouped_matmul_used", False))
        and bool(stage.get("production_svdq_host_tiling_fail_closed", True))
    )
    passed = bool(stage2_2_passed and historical_passed)
    gate.update(
        {
            "status": "passed" if passed else (
                "blocked_by_stage2_2_official_gmm2_gate" if historical_passed else "failed"
            ),
            "passed": passed,
            "historical_passed_under_superseded_contract": historical_passed,
            "blocking_gate": None if stage2_2_passed else "stage2_2_modified_hidden_official_w4a8_gmm2",
            "summary_passed": bool(summary.get("passed")),
            "stage_passed": bool(stage.get("passed")),
            "shape": shape,
            "required_stage_flags": required_stage_flags,
            "required_same_routing_flags": required_same_route_flags,
            "required_official_gmm2_flags": required_gmm2_flags,
            "required_shape_flags": shape_passed,
            "exact_numerical_flags": exact_numerics,
            "required_final_combine_flags": final_combine_flags,
            "rank_metadata": stage.get("rank_metadata"),
            "production_tiling_fail_closed_in_evidence": bool(
                stage.get("production_svdq_host_tiling_fail_closed", True)
            ),
        }
    )
    return gate


def build_manifest(repo_root: Path = REPO_ROOT, evidence_dir: Path = DEFAULT_EVIDENCE_DIR) -> dict[str, Any]:
    sources = _read_sources(repo_root)
    source_proof = _source_proof(sources)
    host_tiling_source = _production_host_tiling_source(sources)
    stage2_2_gate = _stage2_2_official_gmm2_gate(evidence_dir)
    stage2_3_gate = _stage2_3_real_composition_gate(evidence_dir, bool(stage2_2_gate["passed"]))
    manifest = {
        "schema_version": 1,
        "operator": "DispatchFFNCombineW4A8SVDQ",
        "source_files": {
            "op_cmake": str(OP_CMAKE),
            "op_def": str(OP_DEF),
            "debug_op_def": str(DEBUG_OP_DEF),
            "op_proto": str(OP_PROTO),
            "debug_op_api_header": str(DEBUG_OP_API_HEADER),
            "debug_op_api_wrapper": str(DEBUG_OP_API_WRAPPER),
            "host_tiling": str(HOST_TILING),
            "kernel_contract": str(KERNEL_CONTRACT),
            "kernel_tiling": str(KERNEL_TILING),
            "lowrank_header": str(LOWRANK_HEADER),
            "lowrank_debug_tiling": str(LOWRANK_DEBUG_TILING),
            "lowrank_debug_header": str(LOWRANK_DEBUG_HEADER),
            "lowrank_debug_kernel": str(LOWRANK_DEBUG_KERNEL),
            "lowrank_debug_alias_cmake": str(LOWRANK_DEBUG_ALIAS_CMAKE),
            "lowrank_debug_alias_kernel": str(LOWRANK_DEBUG_ALIAS_KERNEL),
            "lowrank_debug_torch_adapter": str(LOWRANK_DEBUG_TORCH_ADAPTER),
            "torch_binding": str(TORCH_BINDING),
            "torch_binding_meta": str(TORCH_BINDING_META),
            "lowrank_debug_probe": str(LOWRANK_DEBUG_PROBE),
            "lowrank_debug_install_validate": str(LOWRANK_DEBUG_INSTALL_VALIDATE),
            "build_aclnn": str(BUILD_ACLNN),
            "official_w4a8_cmake": str(OFFICIAL_W4A8_CMAKE),
            "official_w4a8_debug_def": str(OFFICIAL_W4A8_DEBUG_DEF),
            "official_w4a8_debug_api_header": str(OFFICIAL_W4A8_DEBUG_API_HEADER),
            "official_w4a8_debug_api_wrapper": str(OFFICIAL_W4A8_DEBUG_API_WRAPPER),
            "official_w4a8_host_tiling": str(OFFICIAL_W4A8_HOST_TILING),
            "official_w4a8_kernel": str(OFFICIAL_W4A8_KERNEL),
            "official_w4a8_kernel_entry": str(OFFICIAL_W4A8_KERNEL_ENTRY),
            "official_w4a8_debug_kernel_entry": str(OFFICIAL_W4A8_DEBUG_KERNEL_ENTRY),
            "w4a8_debug_torch_adapter": str(W4A8_DEBUG_TORCH_ADAPTER),
            "w4a8_debug_probe": str(W4A8_DEBUG_PROBE),
            "official_w4a8_op": str(OFFICIAL_W4A8_OP),
            "official_w4a8_gmm1_epilogue": str(OFFICIAL_W4A8_GMM1_EPILOGUE),
            "official_w4a8_gmm2_epilogue": str(OFFICIAL_W4A8_GMM2_EPILOGUE),
        },
        "factor_abi": FACTOR_ABI,
        "workspace_regions": WORKSPACE_REGIONS,
        "sync_flags": _sync_flag_records(),
        "bf16_stages": _bf16_stage_records(),
        "residual_stages": _residual_stage_records(),
        "residual_quant_launches": RESIDUAL_QUANT_LAUNCHES,
        "residual_gmm_launches": RESIDUAL_GMM_LAUNCHES,
        "mixed_epilogue_launches": MIXED_EPILOGUE_LAUNCHES,
        "final_combine_launch": FINAL_COMBINE_LAUNCH,
        "lowrank_invocations": LOWRANK_INVOCATIONS,
        "lowrank_tile_shape": {"m": 16, "n": 64, "k": 64},
        "debug_readback_contract": {
            "compile_macro": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
            "cmake_option": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
            "default_enabled": True,
            "production_abi_changed": False,
            "readback_region": "lowRankAccumulator region selected by invocation.accumulatorRegionId",
            "readback_dtype": "FP32",
            "readback_source": "L0C accumulator after each MMAD K tile",
            "final_tile_semantics": "final full-K FP32 accumulator is mirrored before BF16 output conversion",
            "partial_tile_semantics": "non-final K-tile partial sums are mirrored for host-readable debug validation",
            "stage2_5_required_boundary_checks": [
                "gate_output_bf16_vs_accumulator_cast",
                "up_output_bf16_vs_accumulator_cast",
                "down_output_bf16_vs_accumulator_cast",
            ],
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
        },
        "debug_launch_contract": {
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
        },
        "w4a8_debug_readback_contract": {
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
        },
        "rank_split_contract": {
            "gate_rank_offset": 0,
            "up_rank_offset": "gateRank",
            "gate_up_l1_rank_columns": "gateRank + upRank",
            "split_source": "explicit gateRank/upRank offsets",
        },
        "production_fail_closed": {
            "host_tiling_returns_graph_failed": (
                "production tiling is fail-closed" in host_tiling_source
                or "return ge::GRAPH_FAILED;" in host_tiling_source
            ),
            "host_tiling_metadata_builds_before_fail_closed": source_proof[
                "host_tiling_fail_closed_after_metadata_construction"
            ],
            "host_tiling_success_enabled": source_proof["host_tiling_graph_success_enabled"],
            "sync_handoff_source_enabled": source_proof["kernel_synchronizes_stage_boundaries"],
            "lowrank_is_implemented_uses_complete_contract": (
                "IsImplemented() const\n    {\n        return HasCompleteContract();" in sources["lowrank_header"]
            ),
            "reason": (
                "Synchronization handoff validation, production device runtime validation, "
                "and target-model E2E validation are incomplete."
            ),
            "dispatch_routing_execution_enabled": source_proof["kernel_dispatch_routing_execution_enabled"],
            "residual_routed_input_quant_execution_enabled": source_proof[
                "kernel_residual_routed_input_quant_execution_enabled"
            ],
            "residual_hidden_quant_execution_enabled": source_proof[
                "kernel_residual_hidden_quant_aiv_execution_enabled"
            ],
            "residual_hidden_quant_scalar_helpers_absent": source_proof[
                "kernel_residual_hidden_quant_scalar_helpers_absent"
            ],
            "residual_quant_launch_descriptor_recorded": source_proof[
                "kernel_residual_quant_launch_descriptor_recorded"
            ],
            "residual_gmm_launch_descriptor_recorded": source_proof[
                "kernel_residual_gmm_launch_descriptor_recorded"
            ],
            "residual_gmm_official_bridge_contract_recorded": source_proof[
                "kernel_residual_gmm_official_bridge_contract_recorded"
            ],
            "residual_gmm_official_tiling_bridge_recorded": source_proof[
                "kernel_residual_gmm_official_tiling_bridge_recorded"
            ],
            "residual_gmm_official_tiling_bridge_consumed": source_proof[
                "kernel_residual_gmm_official_tiling_bridge_consumed"
            ],
            "residual_gmm_official_full_lifecycle_call_surface_recorded": source_proof[
                "kernel_residual_gmm_official_full_lifecycle_call_surface_recorded"
            ],
            "residual_gmm_official_interleaved_producer_contract_recorded": source_proof[
                "kernel_residual_gmm_official_interleaved_producer_contract_recorded"
            ],
            "residual_gmm_official_split_producer_segment_contract_recorded": source_proof[
                "kernel_residual_gmm_official_split_producer_segment_contract_recorded"
            ],
            "residual_gmm_official_scratch_output_recorded": source_proof[
                "kernel_residual_gmm_official_scratch_output_recorded"
            ],
            "residual_gmm_embedded_official_tiling_pointer_recorded": source_proof[
                "kernel_residual_gmm_official_full_lifecycle_call_surface_recorded"
            ],
            "residual_gmm_official_wrapper_type_bound": source_proof[
                "kernel_residual_gmm_official_full_lifecycle_call_surface_recorded"
            ],
            "residual_gmm_official_full_lifecycle_execution_enabled": source_proof[
                "kernel_residual_gmm_official_full_lifecycle_execution_enabled"
            ],
            "residual_gmm_execution_enabled": source_proof[
                "kernel_residual_gmm_official_full_lifecycle_execution_enabled"
            ],
            "residual_gmm_scalar_helpers_absent": source_proof[
                "kernel_residual_gmm_scalar_helpers_absent"
            ],
            "mixed_epilogue_launch_descriptor_recorded": source_proof[
                "kernel_mixed_epilogue_launch_descriptor_recorded"
            ],
            "mixed_output_epilogue_execution_enabled": source_proof[
                "kernel_mixed_output_epilogue_aiv_execution_enabled"
            ],
            "mixed_swiglu_epilogue_execution_enabled": source_proof[
                "kernel_mixed_swiglu_epilogue_aiv_execution_enabled"
            ],
            "mixed_epilogue_scalar_helpers_absent": source_proof[
                "kernel_mixed_epilogue_scalar_helpers_absent"
            ],
            "final_combine_launch_descriptor_recorded": source_proof[
                "kernel_final_combine_launch_descriptor_recorded"
            ],
            "final_combine_execution_enabled": source_proof[
                "kernel_final_combine_official_unpermute_execution_enabled"
            ],
            "final_combine_scalar_helpers_absent": source_proof[
                "kernel_final_combine_scalar_helpers_absent"
            ],
            "w4a8_residual_execution_fail_closed": (
                "production tiling is fail-closed" in host_tiling_source
                or not source_proof["kernel_residual_routed_input_quant_execution_enabled"]
                or not source_proof["kernel_residual_hidden_quant_aiv_execution_enabled"]
                or not source_proof["kernel_residual_gmm_scalar_execution_enabled"]
            ),
            "mixed_epilogue_execution_fail_closed": (
                "production tiling is fail-closed" in host_tiling_source
                or not source_proof["kernel_mixed_output_epilogue_aiv_execution_enabled"]
                or not source_proof["kernel_mixed_swiglu_epilogue_aiv_execution_enabled"]
            ),
            "final_combine_execution_fail_closed": "production tiling is fail-closed" in host_tiling_source,
            "w4a8_residual_contract_recorded": True,
            "mixed_epilogue_contract_recorded": True,
            "final_combine_contract_recorded": True,
        },
        "production_admission": {
            "stage2_2_official_gmm2_gate": stage2_2_gate,
            "stage2_2_official_gmm2_gate_passed": bool(stage2_2_gate["passed"]),
            "stage2_3_real_checkpoint_composition_gate": stage2_3_gate,
            "stage2_3_isolated_gate_passed": bool(stage2_2_gate["passed"] and stage2_3_gate["passed"]),
            "host_tiling_must_remain_fail_closed": True,
            "production_enable_allowed": False,
            "reason": (
                "Production tiling remains blocked until Stage 2.2 official GMM2 Gate A/B/C, Stage 2.3 "
                "same-routing composition, fused production execution, and four-NPU target-model validation "
                "all pass. Passing isolated gates does not enable the production host tiling."
            ),
            "remaining_execution_requirements": {
                "dispatch_routing_execution_enabled": source_proof["kernel_dispatch_routing_execution_enabled"],
                "residual_hidden_quant_execution_enabled": source_proof[
                    "kernel_residual_hidden_quant_aiv_execution_enabled"
                ],
                "residual_w4a8_gmm_execution_enabled": source_proof[
                    "kernel_residual_gmm_official_full_lifecycle_execution_enabled"
                ],
                "mixed_swiglu_epilogue_execution_enabled": source_proof[
                    "kernel_mixed_swiglu_epilogue_aiv_execution_enabled"
                ],
                "mixed_output_epilogue_execution_enabled": source_proof[
                    "kernel_mixed_output_epilogue_aiv_execution_enabled"
                ],
                "final_combine_execution_enabled": source_proof[
                    "kernel_final_combine_official_unpermute_execution_enabled"
                ],
                "four_npu_target_model_e2e_validated": False,
            },
        },
        "source_proof": source_proof,
        "counts": {
            "factor_abi": len(FACTOR_ABI),
            "workspace_regions": len(WORKSPACE_REGIONS),
            "sync_flags": len(SYNC_FLAGS),
            "bf16_stages": len(BF16_STAGES),
            "residual_stages": len(RESIDUAL_STAGES),
            "residual_quant_launches": len(RESIDUAL_QUANT_LAUNCHES),
            "residual_gmm_launches": len(RESIDUAL_GMM_LAUNCHES),
            "mixed_epilogue_launches": len(MIXED_EPILOGUE_LAUNCHES),
            "final_combine_launches": 1,
            "lowrank_invocations": len(LOWRANK_INVOCATIONS),
        },
    }
    validate_manifest_sources(manifest, repo_root)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manifest(args.repo_root, evidence_dir=args.evidence_dir)
    output = args.output or args.evidence_dir / "phase_s_kernel_contract_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
