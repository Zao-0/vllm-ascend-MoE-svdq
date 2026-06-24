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

OP_ROOT = Path("csrc/mc2/dispatch_ffn_combine_w4_a8_svdq")
OP_CMAKE = OP_ROOT / "op_host/CMakeLists.txt"
OP_DEF = OP_ROOT / "op_host/dispatch_ffn_combine_w4_a8_svdq_def.cpp"
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
        "dtype": "SVDQ_DTYPE_INT32",
        "size_expr": "maxOutputSize * intermediateSize * 2 * INT32_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_1",
        "consumer_stage": "SVDQ_STAGE_MIXED_EPILOGUE_1",
        "lifetime_id": 6,
        "purpose": "future W4A8 gate/up residual accumulator",
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
        "dtype": "SVDQ_DTYPE_INT32",
        "size_expr": "maxOutputSize * hiddenSize * INT32_BYTES",
        "producer_stage": "SVDQ_STAGE_W4A8_GEMM_2",
        "consumer_stage": "SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE",
        "lifetime_id": 11,
        "purpose": "future W4A8 down residual accumulator",
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
        "SVDQ_REGION_PROJECTION_1",
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
        "SVDQ_REGION_PROJECTION_1",
        "SVDQ_REGION_PROJECTION_1",
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
        "SVDQ_REGION_PROJECTION_1",
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
        "SVDQ_REGION_PROJECTION_1",
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
        "SVDQ_REGION_PROJECTION_2",
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
        "SVDQ_REGION_PROJECTION_2",
        "SVDQ_REGION_PROJECTION_2",
        "routedRows",
        "info.downRank",
        "info.hiddenSize",
        "0",
        "0",
    ),
]

LOWRANK_INVOCATIONS = [
    {
        "id": 0,
        "name": "SVDQ_LOWRANK_INVOCATION_GATE_UP",
        "input_region": "SVDQ_REGION_ROUTED_X",
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
        "torch_binding": (repo_root / TORCH_BINDING).read_text(encoding="utf-8"),
        "torch_binding_meta": (repo_root / TORCH_BINDING_META).read_text(encoding="utf-8"),
        "lowrank_debug_probe": (repo_root / LOWRANK_DEBUG_PROBE).read_text(encoding="utf-8"),
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


def _source_proof(sources: dict[str, str]) -> dict[str, bool]:
    return {
        "host_tiling_builds_workspace_map": "BuildWorkspaceMap(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_sync_flags": "BuildSyncFlagTable(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_bf16_stage_shapes": "BuildBF16StageShapeTable(tilingData)" in sources["host_tiling"],
        "host_tiling_builds_lowrank_invocations": "BuildLowRankInvocationTable(tilingData)" in sources["host_tiling"],
        "op_cmake_has_local_debug_readback_option": (
            "option(SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK" in sources["op_cmake"]
        ),
        "op_cmake_debug_readback_defaults_off": (
            "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK)" in sources["op_cmake"]
            and "OFF)" in sources["op_cmake"]
        ),
        "op_cmake_scopes_debug_readback_to_svdq_op": (
            "${_DISPATCH_FFN_SVDQ_DEBUG_OPTS}" in sources["op_cmake"]
            and "OP_NAME DispatchFFNCombineW4A8SVDQ" in sources["op_cmake"]
        ),
        "kernel_resolves_workspace_addresses": "WorkspaceAddress(uint32_t regionId)" in sources["kernel_contract"],
        "kernel_exposes_bf16_stage_contracts": "BF16StageContract(uint32_t stageId)" in sources["kernel_contract"],
        "lowrank_helper_fail_closed": "IsImplemented() const\n    {\n        return false;"
        in sources["lowrank_header"],
        "host_tiling_fail_closed": "AscendC kernel is not implemented yet" in sources["host_tiling"]
        and "return ge::GRAPH_FAILED;" in sources["host_tiling"],
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
            and "${_DISPATCH_FFN_SVDQ_DEBUG_OPTS}" in sources["op_cmake"]
        ),
        "lowrank_debug_op_registered": (
            "class SVDQLowRankDebugReadback" in sources["op_def"]
            and "OP_ADD(SVDQLowRankDebugReadback)" in sources["op_def"]
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
            'this->Output("gateUpOutput")' in sources["op_def"]
            and 'this->Output("downOutput")' in sources["op_def"]
            and 'this->Output("gateUpAccumulator")' in sources["op_def"]
            and 'this->Output("downAccumulator")' in sources["op_def"]
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

    for invocation in manifest["lowrank_invocations"]:
        for token in (
            invocation["name"],
            invocation["input_region"],
            invocation["output_region"],
            invocation["down_factor"],
            invocation["up_factor"],
            invocation["second_up_factor"],
            invocation["accumulator_region"],
        ):
            if token != "SVDQ_INVALID_ID" and token not in sources["host_tiling"]:
                raise ValueError(f"low-rank invocation token {token} missing from host tiling.")

    if any(not passed for passed in manifest["source_proof"].values()):
        failed = [name for name, passed in manifest["source_proof"].items() if not passed]
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


def build_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    sources = _read_sources(repo_root)
    manifest = {
        "schema_version": 1,
        "operator": "DispatchFFNCombineW4A8SVDQ",
        "source_files": {
            "op_cmake": str(OP_CMAKE),
            "op_def": str(OP_DEF),
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
        },
        "factor_abi": FACTOR_ABI,
        "workspace_regions": WORKSPACE_REGIONS,
        "sync_flags": _sync_flag_records(),
        "bf16_stages": _bf16_stage_records(),
        "lowrank_invocations": LOWRANK_INVOCATIONS,
        "lowrank_tile_shape": {"m": 16, "n": 64, "k": 64},
        "debug_readback_contract": {
            "compile_macro": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
            "cmake_option": "SVDQ_LOWRANK_DEBUG_ACCUMULATOR_READBACK",
            "default_enabled": False,
            "production_abi_changed": False,
            "readback_region": "lowRankAccumulator region selected by invocation.accumulatorRegionId",
            "readback_dtype": "FP32",
            "readback_source": "L0C accumulator after each MMAD K tile",
            "final_tile_semantics": "final full-K FP32 accumulator is mirrored before BF16 output conversion",
            "partial_tile_semantics": "non-final K-tile partial sums are mirrored for host-readable debug validation",
            "source_proof": [
                "op_cmake_has_local_debug_readback_option",
                "op_cmake_debug_readback_defaults_off",
                "op_cmake_scopes_debug_readback_to_svdq_op",
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
                "lowrank_debug_op_registered",
                "lowrank_debug_op_tiling_registered",
                "lowrank_debug_op_public_aclnn_wrapper",
                "lowrank_debug_op_exposes_same_readback_outputs",
                "lowrank_debug_alias_source_root",
                "lowrank_debug_torch_adapter_registered",
                "lowrank_debug_meta_registered",
                "lowrank_debug_probe_launches_real_op",
            ],
        },
        "rank_split_contract": {
            "gate_rank_offset": 0,
            "up_rank_offset": "gateRank",
            "gate_up_l1_rank_columns": "gateRank + upRank",
            "split_source": "explicit gateRank/upRank offsets",
        },
        "production_fail_closed": {
            "host_tiling_returns_graph_failed": "AscendC kernel is not implemented yet" in sources["host_tiling"],
            "lowrank_is_implemented_returns_false": "IsImplemented() const\n    {\n        return false;"
            in sources["lowrank_header"],
            "reason": (
                "W4A8 residual stages, mixed epilogues, final combine, "
                "and custom-kernel numerical readback are incomplete."
            ),
            "w4a8_residual_unblocked": False,
        },
        "source_proof": _source_proof(sources),
        "counts": {
            "factor_abi": len(FACTOR_ABI),
            "workspace_regions": len(WORKSPACE_REGIONS),
            "sync_flags": len(SYNC_FLAGS),
            "bf16_stages": len(BF16_STAGES),
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
    manifest = build_manifest(args.repo_root)
    output = args.output or args.evidence_dir / "phase_s_kernel_contract_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
