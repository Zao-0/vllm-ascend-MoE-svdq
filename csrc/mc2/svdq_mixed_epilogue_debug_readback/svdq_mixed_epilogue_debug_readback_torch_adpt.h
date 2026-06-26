/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OF CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_TORCH_ADPT_H
#define SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_TORCH_ADPT_H

#include <torch/extension.h>

#include "../dispatch_ffn_combine_w4_a8_svdq/op_host/op_api/aclnn_svdq_mixed_epilogue_debug_readback.h"

#include <tuple>

namespace vllm_ascend {

namespace {

inline void check_mixed_debug_rank2(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.dim() == 2, name, " must be rank-2, got ", tensor.sizes(), ".");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
svdq_mixed_epilogue_debug_readback(
    const at::Tensor& residual_gate_up,
    const at::Tensor& gate_up_low_rank,
    const at::Tensor& residual_down,
    const at::Tensor& down_low_rank,
    double swiglu_limit)
{
    TORCH_CHECK(residual_gate_up.scalar_type() == at::kFloat, "residual_gate_up must be FP32.");
    TORCH_CHECK(gate_up_low_rank.scalar_type() == at::kBFloat16, "gate_up_low_rank must be BF16.");
    TORCH_CHECK(residual_down.scalar_type() == at::kFloat, "residual_down must be FP32.");
    TORCH_CHECK(down_low_rank.scalar_type() == at::kBFloat16, "down_low_rank must be BF16.");
    check_mixed_debug_rank2(residual_gate_up, "residual_gate_up");
    check_mixed_debug_rank2(gate_up_low_rank, "gate_up_low_rank");
    check_mixed_debug_rank2(residual_down, "residual_down");
    check_mixed_debug_rank2(down_low_rank, "down_low_rank");
    TORCH_CHECK(swiglu_limit >= 0.0, "swiglu_limit must be non-negative.");
    TORCH_CHECK(residual_gate_up.size(1) % 2 == 0, "residual_gate_up width must be even.");
    TORCH_CHECK(gate_up_low_rank.sizes() == residual_gate_up.sizes(),
                "gate_up_low_rank shape must match residual_gate_up.");
    TORCH_CHECK(down_low_rank.sizes() == residual_down.sizes(), "down_low_rank shape must match residual_down.");
    TORCH_CHECK(residual_down.size(0) == residual_gate_up.size(0),
                "down branch row count must match gate/up row count.");

    const auto rows = residual_gate_up.size(0);
    const auto intermediate_size = residual_gate_up.size(1) / 2;
    const auto hidden_size = residual_down.size(1);
    TORCH_CHECK(intermediate_size % 64 == 0 && hidden_size % 64 == 0,
                "intermediate_size and hidden_size must be multiples of 64 for the debug AIV tile.");

    auto gate_up_total = at::empty_like(residual_gate_up);
    auto hidden_bf16 = at::empty({rows, intermediate_size}, gate_up_low_rank.options());
    auto hidden_int8 = at::empty({rows, intermediate_size}, gate_up_low_rank.options().dtype(at::kChar));
    auto hidden_int4_packed = at::empty({rows, intermediate_size}, gate_up_low_rank.options().dtype(at::kChar));
    auto hidden_scale = at::empty({rows}, residual_gate_up.options());
    auto down_total = at::empty_like(residual_down);
    auto out_bf16 = at::empty({rows, hidden_size}, down_low_rank.options());

    EXEC_NPU_CMD(
        aclnnSVDQMixedEpilogueDebugReadback,
        residual_gate_up,
        gate_up_low_rank,
        residual_down,
        down_low_rank,
        swiglu_limit,
        gate_up_total,
        hidden_bf16,
        hidden_int8,
        hidden_int4_packed,
        hidden_scale,
        down_total,
        out_bf16);
    return {gate_up_total, hidden_bf16, hidden_int8, hidden_int4_packed, hidden_scale, down_total, out_bf16};
}

}  // namespace vllm_ascend

#endif  // SVDQ_MIXED_EPILOGUE_DEBUG_READBACK_TORCH_ADPT_H
