#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
#

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE

import vllm_ascend.patch.worker.patch_svdq_moe_loading  # noqa: F401
from vllm_ascend.quantization.methods.svdq_weight_loader import make_svdq_factor_weight_loader
from vllm_ascend.quantization.svdq_spec import build_svdq_moe_layer_spec, detect_svdq_moe_layer

MODEL_PATH = "/root/workspace/lza/LLM/Qwen3.5-35B-A3B-W4A8-svdq-r64-mtp-canonical"


def _load_quant_description():
    path = os.path.join(MODEL_PATH, "quant_model_description.json")
    if not os.path.exists(path):
        pytest.skip(f"SVDQ real checkpoint metadata not available at {MODEL_PATH}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_real_qwen35_svdq_spec_detects_complete_layers():
    quant_description = _load_quant_description()

    for layer_idx in (0, 39):
        prefix = f"model.language_model.layers.{layer_idx}.mlp.experts"
        assert detect_svdq_moe_layer(quant_description, prefix, num_experts=256)

        spec = build_svdq_moe_layer_spec(
            quant_description=quant_description,
            prefix=prefix,
            model_path=MODEL_PATH,
            num_experts=256,
            hidden_size=2048,
            intermediate_size=512,
        )

        assert spec.gate_rank == 64
        assert spec.up_rank == 64
        assert spec.down_rank == 64
        assert spec.lowrank_dtype is torch.bfloat16
        assert spec.group_size == 0
        assert spec.modelslim_version == "1.0.0"


def test_real_qwen35_svdq_spec_rejects_missing_factor():
    quant_description = _load_quant_description()
    incomplete = copy.deepcopy(quant_description)
    del incomplete["model.language_model.layers.0.mlp.experts.0.gate_proj.svd_lowrank_l1"]

    assert not detect_svdq_moe_layer(
        incomplete,
        "model.language_model.layers.0.mlp.experts",
        num_experts=256,
    )


def test_svdq_factor_loader_preserves_ep_tp_rules():
    layer = SimpleNamespace(
        _expert_map=torch.tensor([1, -1, 0], dtype=torch.int32),
        moe_parallel_config=SimpleNamespace(tp_rank=1),
    )

    param = torch.nn.Parameter(torch.empty(2, 3, 2, dtype=torch.bfloat16), requires_grad=False)
    param.svdq_shard_id = "svdq_w1_l2"
    param.svdq_rank_dim = 1
    param.svdq_rank = 2
    param.svdq_dtype = torch.bfloat16
    param.svdq_loaded_experts = set()

    loader = make_svdq_factor_weight_loader(layer)
    loaded = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(torch.bfloat16)

    assert loader(param, loaded, "expert0.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 0, True)
    assert torch.equal(param.data[1], loaded[3:6])
    assert param.svdq_loaded_experts == {1}

    assert not loader(param, loaded, "expert1.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 1, True)
    assert param.svdq_loaded_experts == {1}

    assert loader(param, loaded + 100, "expert2.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 2, True)
    assert torch.equal(param.data[0], (loaded + 100)[3:6])
    assert param.svdq_loaded_experts == {0, 1}

    with pytest.raises(ValueError, match="Duplicate SVDQ factor load"):
        loader(param, loaded, "expert2.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 2, True)

    bad_dtype = loaded.to(torch.float16)
    with pytest.raises(ValueError, match="must use dtype"):
        loader(param, bad_dtype, "expert0.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 0, True)

    bad_rank = torch.empty(6, 3, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="rank mismatch"):
        loader(param, bad_rank, "expert0.gate_proj.svd_lowrank_l2", "svdq_w1_l2", 0, True)


def test_svdq_patch_adds_factor_expert_mappings():
    module = torch.nn.Module()
    module.experts = torch.nn.Module()
    module.experts.register_parameter("svdq_w1_l1", torch.nn.Parameter(torch.empty(1), requires_grad=False))

    mappings = FusedMoE.make_expert_params_mapping(
        module,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=2,
    )

    svdq_mappings = [mapping for mapping in mappings if mapping[3].startswith("svdq_")]
    assert len(svdq_mappings) == 12
    assert (
        "experts.svdq_w1_l1",
        "experts.0.gate_proj.svd_lowrank_l1",
        0,
        "svdq_w1_l1",
    ) in svdq_mappings
    assert (
        "experts.svdq_w2_l2",
        "experts.1.down_proj.svd_lowrank_l2",
        1,
        "svdq_w2_l2",
    ) in svdq_mappings
