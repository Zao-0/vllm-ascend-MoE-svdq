#!/usr/bin/env python3
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
"""Validate installed SVDQ low-rank debug op package surfaces."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from svdq_loader_pre_kernel_validate import DEFAULT_EVIDENCE_DIR  # noqa: E402
from svdq_lowrank_debug_readback_probe import (  # noqa: E402
    DEBUG_OP_NAME,
    _custom_package_debug_op_support,
    _package_supports_runtime_soc,
    _runtime_soc,
)

from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op  # noqa: E402

DEFAULT_SUMMARY_NAME = "phase_z_lowrank_debug_install_validate_summary.json"
CUSTOM_OP_VENDOR_DIR = REPO_ROOT / "vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
CUSTOM_OPAPI_LIB = CUSTOM_OP_VENDOR_DIR / "op_api/lib/libcust_opapi.so"
REQUIRED_OPAPI_SYMBOLS = (
    "aclnnSVDQLowRankDebugReadbackGetWorkspaceSize",
    "aclnnSVDQLowRankDebugReadback",
    "aclnnInnerSVDQLowRankDebugReadbackGetWorkspaceSize",
    "aclnnInnerSVDQLowRankDebugReadback",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--require-runtime-soc-support",
        action="store_true",
        help="Fail unless the installed package advertises SVDQLowRankDebugReadback for the current runtime SOC.",
    )
    return parser.parse_args()


def _torch_schema_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "custom_op_enabled": False,
        "has_namespace": hasattr(torch.ops, "_C_ascend"),
        "has_debug_op": False,
    }
    try:
        status["custom_op_enabled"] = bool(enable_custom_op())
    except Exception as exc:
        status["enable_custom_op_error"] = f"{type(exc).__name__}: {exc}"

    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is not None:
        status["has_debug_op"] = getattr(namespace, "svdq_low_rank_debug_readback", None) is not None
    return status


def _opapi_symbol_status(lib_path: Path = CUSTOM_OPAPI_LIB) -> dict[str, Any]:
    status: dict[str, Any] = {
        "lib_path": str(lib_path),
        "exists": lib_path.exists(),
        "loaded": False,
        "symbols": {name: False for name in REQUIRED_OPAPI_SYMBOLS},
    }
    if not lib_path.exists():
        return status
    old_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    lib_dir = str(lib_path.parent)
    if lib_dir not in old_ld_library_path.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld_library_path}" if old_ld_library_path else lib_dir
    try:
        handle = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)
    except Exception as exc:
        status["load_error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["loaded"] = True
    status["symbols"] = {name: hasattr(handle, name) for name in REQUIRED_OPAPI_SYMBOLS}
    return status


def build_summary(*, device_id: int, require_runtime_soc_support: bool) -> dict[str, Any]:
    bootstrap_custom_op_env(include_vendor_lib=True)
    runtime_soc = _runtime_soc(device_id)
    package_support = _custom_package_debug_op_support()
    torch_schema = _torch_schema_status()
    opapi_symbols = _opapi_symbol_status()
    required_static_checks = [
        bool(torch_schema["has_debug_op"]),
        bool(opapi_symbols["exists"]),
        bool(opapi_symbols["loaded"]),
        all(opapi_symbols["symbols"].values()),
        bool(package_support["debug_op_supported_socs"]),
    ]
    runtime_supported = _package_supports_runtime_soc(
        package_support=package_support,
        runtime_soc=runtime_soc.get("normalized_soc"),
    )
    passed = all(required_static_checks) and (runtime_supported or not require_runtime_soc_support)
    return {
        "passed": passed,
        "require_runtime_soc_support": require_runtime_soc_support,
        "debug_op_name": DEBUG_OP_NAME,
        "runtime_soc": runtime_soc,
        "runtime_soc_supported": runtime_supported,
        "torch_schema": torch_schema,
        "opapi_symbols": opapi_symbols,
        "custom_package_debug_op_support": package_support,
    }


def main() -> None:
    args = _parse_args()
    os.makedirs(args.evidence_dir, exist_ok=True)
    summary = build_summary(
        device_id=args.device_id,
        require_runtime_soc_support=args.require_runtime_soc_support,
    )
    summary_path = Path(args.evidence_dir) / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
