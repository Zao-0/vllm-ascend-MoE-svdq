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
HOST_TILING = OP_ROOT / "op_host/dispatch_ffn_combine_w4_a8_svdq_tiling.cpp"
KERNEL_CONTRACT = OP_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8_svdq.h"
KERNEL_TILING = OP_ROOT / "op_kernel/dispatch_ffn_combine_w4_a8_svdq_tiling.h"
LOWRANK_HEADER = OP_ROOT / "op_kernel/lowrank/svdq_fused_down_up.hpp"

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
        "host_tiling": (repo_root / HOST_TILING).read_text(encoding="utf-8"),
        "kernel_contract": (repo_root / KERNEL_CONTRACT).read_text(encoding="utf-8"),
        "kernel_tiling": (repo_root / KERNEL_TILING).read_text(encoding="utf-8"),
        "lowrank_header": (repo_root / LOWRANK_HEADER).read_text(encoding="utf-8"),
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
        "kernel_resolves_workspace_addresses": "WorkspaceAddress(uint32_t regionId)" in sources["kernel_contract"],
        "kernel_exposes_bf16_stage_contracts": "BF16StageContract(uint32_t stageId)" in sources["kernel_contract"],
        "lowrank_helper_fail_closed": "IsImplemented() const\n    {\n        return false;"
        in sources["lowrank_header"],
        "host_tiling_fail_closed": "AscendC kernel is not implemented yet" in sources["host_tiling"]
        and "return ge::GRAPH_FAILED;" in sources["host_tiling"],
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


def build_manifest(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    sources = _read_sources(repo_root)
    manifest = {
        "schema_version": 1,
        "operator": "DispatchFFNCombineW4A8SVDQ",
        "source_files": {
            "host_tiling": str(HOST_TILING),
            "kernel_contract": str(KERNEL_CONTRACT),
            "kernel_tiling": str(KERNEL_TILING),
            "lowrank_header": str(LOWRANK_HEADER),
        },
        "factor_abi": FACTOR_ABI,
        "workspace_regions": WORKSPACE_REGIONS,
        "sync_flags": _sync_flag_records(),
        "bf16_stages": _bf16_stage_records(),
        "lowrank_invocations": LOWRANK_INVOCATIONS,
        "lowrank_tile_shape": {"m": 16, "n": 64, "k": 64},
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
