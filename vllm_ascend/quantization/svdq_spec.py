#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""Checkpoint-backed specification helpers for ModelSlim W4A8-SVDQ MoE."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import torch
from safetensors import safe_open

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")

SVDQ_FACTOR_TO_RAW_NAME = {
    "svdq_w1_l1": "gate_svd_l1_raw",
    "svdq_w1_l2": "gate_svd_l2_raw",
    "svdq_w3_l1": "up_svd_l1_raw",
    "svdq_w3_l2": "up_svd_l2_raw",
    "svdq_w2_l1": "down_svd_l1_raw",
    "svdq_w2_l2": "down_svd_l2_raw",
}

SVDQ_FACTOR_SPECS = {
    "svdq_w1_l1": ("gate_proj", "svd_lowrank_l1", None, 0),
    "svdq_w1_l2": ("gate_proj", "svd_lowrank_l2", 0, 1),
    "svdq_w3_l1": ("up_proj", "svd_lowrank_l1", None, 0),
    "svdq_w3_l2": ("up_proj", "svd_lowrank_l2", 0, 1),
    "svdq_w2_l1": ("down_proj", "svd_lowrank_l1", 1, 0),
    "svdq_w2_l2": ("down_proj", "svd_lowrank_l2", None, 1),
}

SVDQ_PROJECTION_KEYS = ("gate_proj", "up_proj", "down_proj")
SVDQ_RESIDUAL_KINDS = ("weight", "weight_scale", "weight_offset", "scale_bias")
SVDQ_FACTOR_KINDS = ("svd_lowrank_l1", "svd_lowrank_l2")


@dataclass(frozen=True)
class SVDQMoELayerSpec:
    prefix: str
    checkpoint_prefix: str
    layer_index: int
    num_experts: int
    hidden_size: int
    intermediate_size: int
    gate_rank: int
    up_rank: int
    down_rank: int
    lowrank_dtype: torch.dtype
    residual_quant_type: str
    group_size: int
    modelslim_version: str

    @property
    def rank_by_loader_name(self) -> dict[str, int]:
        return {
            "svdq_w1_l1": self.gate_rank,
            "svdq_w1_l2": self.gate_rank,
            "svdq_w3_l1": self.up_rank,
            "svdq_w3_l2": self.up_rank,
            "svdq_w2_l1": self.down_rank,
            "svdq_w2_l2": self.down_rank,
        }


def _extract_layer_index(prefix: str) -> int | None:
    match = _LAYER_RE.search(prefix)
    if match is None:
        return None
    return int(match.group(1))


def _torch_dtype(dtype_name: str) -> torch.dtype:
    dtype_name = dtype_name.removeprefix("torch.")
    if dtype_name in ("bfloat16", "BF16"):
        return torch.bfloat16
    if dtype_name in ("float16", "F16"):
        return torch.float16
    if dtype_name in ("float32", "F32"):
        return torch.float32
    raise ValueError(f"Unsupported SVDQ low-rank dtype {dtype_name!r}.")


def _tensor_metadata_by_key(
    model_path: str,
    keys: list[str],
    weight_map: dict[str, str],
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    by_shard: dict[str, list[str]] = {}
    for key in keys:
        by_shard.setdefault(weight_map[key], []).append(key)

    metadata = {}
    for shard, shard_keys in by_shard.items():
        with safe_open(os.path.join(model_path, shard), framework="pt", device="cpu") as f:
            for key in shard_keys:
                tensor_slice = f.get_slice(key)
                metadata[key] = (tuple(tensor_slice.get_shape()), _torch_dtype(str(tensor_slice.get_dtype())))
    return metadata


@lru_cache(maxsize=4)
def _weight_map(model_path: str) -> dict[str, str]:
    index_path = os.path.join(model_path, "quant_model_weights.safetensors.index.json")
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def _find_checkpoint_prefix(weight_map: dict[str, str], layer_index: int) -> str | None:
    suffix = f"layers.{layer_index}.mlp.experts.0.gate_proj.svd_lowrank_l1"
    for key in weight_map:
        if key.endswith(suffix):
            return key[: -len(".0.gate_proj.svd_lowrank_l1")]
    return None


def _has_complete_quant_description(quant_description: dict[str, Any], prefix: str, num_experts: int) -> bool:
    for expert in range(num_experts):
        for projection in SVDQ_PROJECTION_KEYS:
            for kind in SVDQ_RESIDUAL_KINDS:
                if quant_description.get(f"{prefix}.{expert}.{projection}.{kind}") != "W4A8_DYNAMIC":
                    return False
            for kind in SVDQ_FACTOR_KINDS:
                if quant_description.get(f"{prefix}.{expert}.{projection}.{kind}") != "FLOAT":
                    return False
    return True


def detect_svdq_moe_layer(
    quant_description: dict[str, Any],
    prefix: str,
    num_experts: int,
) -> bool:
    """Return True only for a complete W4A8 residual + FLOAT factor MoE layer."""
    layer_index = _extract_layer_index(prefix)
    if layer_index is None:
        return False
    if _has_complete_quant_description(quant_description, prefix, num_experts):
        return True

    # Prefixes in ModelSlim metadata may be pre- or post-vLLM mapper. Use the
    # layer index as the stable anchor, but still require every expert key.
    candidates = set()
    suffix = f"layers.{layer_index}.mlp.experts.0.gate_proj.svd_lowrank_l1"
    for key, value in quant_description.items():
        if value == "FLOAT" and key.endswith(suffix):
            candidates.add(key[: -len(".0.gate_proj.svd_lowrank_l1")])
    return any(_has_complete_quant_description(quant_description, candidate, num_experts) for candidate in candidates)


def build_svdq_moe_layer_spec(
    quant_description: dict[str, Any],
    prefix: str,
    model_path: str,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> SVDQMoELayerSpec:
    layer_index = _extract_layer_index(prefix)
    if layer_index is None:
        raise ValueError(f"Cannot build SVDQ spec for non-layer MoE prefix {prefix!r}.")
    if not os.path.isdir(model_path):
        raise ValueError(f"SVDQ checkpoint inspection requires a local model path, got {model_path!r}.")

    weight_map = _weight_map(model_path)
    checkpoint_prefix = _find_checkpoint_prefix(weight_map, layer_index)
    if checkpoint_prefix is None:
        raise ValueError(f"No SVDQ checkpoint factors found for layer {layer_index}.")

    ranks: dict[str, int] = {}
    dtypes: set[torch.dtype] = set()
    expected_shapes = {
        ("gate_proj", "svd_lowrank_l1"): (None, hidden_size),
        ("gate_proj", "svd_lowrank_l2"): (intermediate_size, None),
        ("up_proj", "svd_lowrank_l1"): (None, hidden_size),
        ("up_proj", "svd_lowrank_l2"): (intermediate_size, None),
        ("down_proj", "svd_lowrank_l1"): (None, intermediate_size),
        ("down_proj", "svd_lowrank_l2"): (hidden_size, None),
    }

    factor_keys = []
    for expert in range(num_experts):
        for projection in SVDQ_PROJECTION_KEYS:
            for kind in SVDQ_FACTOR_KINDS:
                key = f"{checkpoint_prefix}.{expert}.{projection}.{kind}"
                if key not in weight_map:
                    raise ValueError(f"Missing SVDQ factor {key}.")
                factor_keys.append(key)

    metadata_by_key = _tensor_metadata_by_key(model_path, factor_keys, weight_map)
    for expert in range(num_experts):
        for projection in SVDQ_PROJECTION_KEYS:
            for kind in SVDQ_FACTOR_KINDS:
                key = f"{checkpoint_prefix}.{expert}.{projection}.{kind}"
                shape, dtype = metadata_by_key[key]
                dtypes.add(dtype)
                expected = expected_shapes[(projection, kind)]
                if len(shape) != 2:
                    raise ValueError(f"SVDQ factor {key} must be rank-2, got shape {shape}.")
                for dim, expected_dim in enumerate(expected):
                    if expected_dim is not None and shape[dim] != expected_dim:
                        raise ValueError(f"SVDQ factor {key} expected dim {dim}={expected_dim}, got {shape}.")
                rank = shape[0] if kind == "svd_lowrank_l1" else shape[1]
                rank_key = projection.removesuffix("_proj")
                previous = ranks.setdefault(rank_key, rank)
                if previous != rank:
                    raise ValueError(f"SVDQ rank mismatch for {rank_key}: {previous} vs {rank} in {key}.")

    if dtypes != {torch.bfloat16}:
        raise ValueError(f"Only BF16 SVDQ factors are supported, got {sorted(str(dtype) for dtype in dtypes)}.")

    return SVDQMoELayerSpec(
        prefix=prefix,
        checkpoint_prefix=checkpoint_prefix,
        layer_index=layer_index,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_rank=ranks["gate"],
        up_rank=ranks["up"],
        down_rank=ranks["down"],
        lowrank_dtype=torch.bfloat16,
        residual_quant_type="W4A8_DYNAMIC",
        group_size=int(quant_description.get("group_size", 0)),
        modelslim_version=str(quant_description.get("version", "0")),
    )
