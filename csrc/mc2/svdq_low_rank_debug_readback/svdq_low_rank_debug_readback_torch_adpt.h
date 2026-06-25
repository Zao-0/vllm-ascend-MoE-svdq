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
#ifndef SVDQ_LOW_RANK_DEBUG_READBACK_TORCH_ADPT_H
#define SVDQ_LOW_RANK_DEBUG_READBACK_TORCH_ADPT_H

#include <torch/extension.h>

#include "../dispatch_ffn_combine_w4_a8_svdq/op_host/op_api/aclnn_svdq_lowrank_debug_readback.h"

#include <tuple>

namespace vllm_ascend {

namespace {

inline void check_debug_rank(const at::Tensor& tensor, int64_t rank, const char* name)
{
    TORCH_CHECK(tensor.dim() == rank, name, " must be rank-", rank, ", got ", tensor.sizes(), ".");
}

inline void check_debug_bf16_rank3(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16, got ", tensor.scalar_type(), ".");
    check_debug_rank(tensor, 3, name);
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> svdq_low_rank_debug_readback(
    const at::Tensor& routed_x,
    const at::Tensor& hidden,
    const at::Tensor& gate_up_svdq_l1,
    const at::Tensor& gate_svdq_l2,
    const at::Tensor& up_svdq_l2,
    const at::Tensor& down_svdq_l1,
    const at::Tensor& down_svdq_l2,
    const at::Tensor& expert_token_nums,
    int64_t gate_rank,
    int64_t up_rank,
    int64_t down_rank,
    int64_t gate_rank_offset,
    int64_t up_rank_offset)
{
    TORCH_CHECK(routed_x.scalar_type() == at::kBFloat16, "routed_x must be BF16.");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be BF16.");
    TORCH_CHECK(expert_token_nums.scalar_type() == at::kInt, "expert_token_nums must be INT32.");
    check_debug_rank(routed_x, 2, "routed_x");
    check_debug_rank(hidden, 2, "hidden");
    check_debug_rank(expert_token_nums, 1, "expert_token_nums");
    check_debug_bf16_rank3(gate_up_svdq_l1, "gate_up_svdq_l1");
    check_debug_bf16_rank3(gate_svdq_l2, "gate_svdq_l2");
    check_debug_bf16_rank3(up_svdq_l2, "up_svdq_l2");
    check_debug_bf16_rank3(down_svdq_l1, "down_svdq_l1");
    check_debug_bf16_rank3(down_svdq_l2, "down_svdq_l2");

    TORCH_CHECK(gate_rank > 0 && up_rank > 0 && down_rank > 0, "SVDQ ranks must be positive.");
    TORCH_CHECK(gate_rank_offset == 0, "gate_rank_offset must be 0.");
    TORCH_CHECK(up_rank_offset == gate_rank, "up_rank_offset must equal gate_rank.");
    TORCH_CHECK(routed_x.size(0) == hidden.size(0), "routed_x and hidden must have the same token count.");

    const auto num_experts = gate_up_svdq_l1.size(0);
    const auto routed_rows = routed_x.size(0);
    const auto hidden_size = routed_x.size(1);
    const auto intermediate_size = hidden.size(1);
    const auto padded_down_rank = ((down_rank + 255) / 256) * 256;
    TORCH_CHECK(gate_up_svdq_l1.size(1) == gate_rank + up_rank,
                "gate_up_svdq_l1 rank dim must equal gate_rank + up_rank.");
    TORCH_CHECK(gate_up_svdq_l1.size(2) == hidden_size, "gate_up_svdq_l1 hidden dim must match routed_x.");
    TORCH_CHECK(gate_svdq_l2.size(0) == num_experts && gate_svdq_l2.size(1) == intermediate_size &&
                    gate_svdq_l2.size(2) == gate_rank,
                "gate_svdq_l2 shape is inconsistent with hidden and gate_rank.");
    TORCH_CHECK(up_svdq_l2.size(0) == num_experts && up_svdq_l2.size(1) == intermediate_size &&
                    up_svdq_l2.size(2) == up_rank,
                "up_svdq_l2 shape is inconsistent with hidden and up_rank.");
    TORCH_CHECK(down_svdq_l1.size(0) == num_experts && down_svdq_l1.size(1) == padded_down_rank &&
                    down_svdq_l1.size(2) == intermediate_size,
                "down_svdq_l1 shape is inconsistent with hidden and padded down_rank.");
    TORCH_CHECK(down_svdq_l2.size(0) == num_experts && down_svdq_l2.size(1) == hidden_size &&
                    down_svdq_l2.size(2) == padded_down_rank,
                "down_svdq_l2 shape is inconsistent with routed_x and padded down_rank.");
    TORCH_CHECK(expert_token_nums.size(0) == num_experts, "expert_token_nums expert count must match factors.");

    auto gate_up_output = at::empty({routed_rows, intermediate_size * 2}, routed_x.options());
    auto down_output = at::empty({routed_rows, hidden_size}, routed_x.options());
    auto gate_up_accumulator = at::empty({routed_rows, intermediate_size * 2}, routed_x.options().dtype(at::kFloat));
    auto down_accumulator = at::empty({routed_rows, hidden_size}, routed_x.options().dtype(at::kFloat));

    EXEC_NPU_CMD(
        aclnnSVDQLowRankDebugReadback,
        routed_x,
        hidden,
        gate_up_svdq_l1,
        gate_svdq_l2,
        up_svdq_l2,
        down_svdq_l1,
        down_svdq_l2,
        expert_token_nums,
        gate_rank,
        up_rank,
        down_rank,
        gate_rank_offset,
        up_rank_offset,
        gate_up_output,
        down_output,
        gate_up_accumulator,
        down_accumulator);
    return {gate_up_output, down_output, gate_up_accumulator, down_accumulator};
}

}  // namespace vllm_ascend

#endif  // SVDQ_LOW_RANK_DEBUG_READBACK_TORCH_ADPT_H
