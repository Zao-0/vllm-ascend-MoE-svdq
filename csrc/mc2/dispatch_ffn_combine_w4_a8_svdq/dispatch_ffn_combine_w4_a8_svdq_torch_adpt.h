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
#ifndef DISPATCH_FFN_COMBINE_W4_A8_SVDQ_TORCH_ADPT_H
#define DISPATCH_FFN_COMBINE_W4_A8_SVDQ_TORCH_ADPT_H

#include <torch/extension.h>

#include "op_host/op_api/aclnn_dispatch_ffn_combine_w4_a8_svdq.h"

#include <string>
#include <tuple>

namespace vllm_ascend {

namespace {

inline const at::Tensor& first_svdq_tensor(const at::TensorList& tensors, const char* name)
{
    TORCH_CHECK(!tensors.empty(), name, " must not be an empty tensor list.");
    return tensors[0];
}

inline void check_rank(const at::Tensor& tensor, int64_t rank, const char* name)
{
    TORCH_CHECK(tensor.dim() == rank, name, " must be rank-", rank, ", got ", tensor.sizes(), ".");
}

inline void check_bf16_factor(const at::Tensor& tensor, const char* name)
{
    TORCH_CHECK(tensor.scalar_type() == at::kBFloat16, name, " must be BF16, got ", tensor.scalar_type(), ".");
    check_rank(tensor, 3, name);
}

}  // namespace

std::tuple<at::Tensor&, at::Tensor&> dispatch_ffn_combine_w4a8_svdq(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::TensorList& bias1,
    const at::TensorList& bias2,
    const at::Tensor& probs,
    const at::Tensor& gate_up_svdq_l1,
    const at::Tensor& gate_svdq_l2,
    const at::Tensor& up_svdq_l2,
    const at::Tensor& down_svdq_l1,
    const at::Tensor& down_svdq_l2,
    int64_t gate_rank,
    int64_t up_rank,
    int64_t down_rank,
    int64_t gate_rank_offset,
    int64_t up_rank_offset,
    c10::string_view group,
    int64_t max_output_size,
    at::Tensor& out,
    at::Tensor& expert_token_nums,
    const c10::optional<at::Tensor>& x_active_mask,
    double swiglu_limit)
{
    check_rank(x, 2, "x");
    check_rank(expert_idx, 2, "expert_idx");
    check_rank(probs, 2, "probs");
    check_rank(out, 2, "out");
    TORCH_CHECK(x.sizes() == out.sizes(), "out shape must match x shape.");
    TORCH_CHECK(expert_idx.sizes() == probs.sizes(), "expert_idx shape must match probs shape.");
    TORCH_CHECK(expert_idx.size(0) == x.size(0), "expert_idx token dimension must match x.");
    if (x_active_mask.has_value()) {
        TORCH_CHECK(x_active_mask.value().size(0) == x.size(0), "x_active_mask token dimension must match x.");
    }
    TORCH_CHECK(gate_rank > 0 && up_rank > 0 && down_rank > 0, "SVDQ ranks must be positive.");
    TORCH_CHECK(gate_rank_offset == 0, "gate_rank_offset must be 0.");
    TORCH_CHECK(up_rank_offset == gate_rank, "up_rank_offset must equal gate_rank.");
    TORCH_CHECK(max_output_size > 0, "max_output_size must be positive.");
    TORCH_CHECK(swiglu_limit >= 0, "swiglu_limit must be non-negative.");
    std::string group_string(group);
    TORCH_CHECK(!group_string.empty(), "group must be a non-empty string.");

    check_bf16_factor(gate_up_svdq_l1, "gate_up_svdq_l1");
    check_bf16_factor(gate_svdq_l2, "gate_svdq_l2");
    check_bf16_factor(up_svdq_l2, "up_svdq_l2");
    check_bf16_factor(down_svdq_l1, "down_svdq_l1");
    check_bf16_factor(down_svdq_l2, "down_svdq_l2");

    const auto num_experts = gate_up_svdq_l1.size(0);
    const auto total_rank = gate_up_svdq_l1.size(1);
    const auto hidden_size = gate_up_svdq_l1.size(2);
    TORCH_CHECK(hidden_size == x.size(1), "gate_up_svdq_l1 hidden dim must match x hidden dim.");
    TORCH_CHECK(total_rank == gate_rank + up_rank, "gate_up_svdq_l1 rank dim must equal gate_rank + up_rank.");
    TORCH_CHECK(gate_svdq_l2.size(0) == num_experts && gate_svdq_l2.size(2) == gate_rank,
                "gate_svdq_l2 shape is inconsistent with gate_rank.");
    const auto intermediate_size = gate_svdq_l2.size(1);
    TORCH_CHECK(up_svdq_l2.size(0) == num_experts && up_svdq_l2.size(1) == intermediate_size &&
                    up_svdq_l2.size(2) == up_rank,
                "up_svdq_l2 shape is inconsistent with up_rank.");
    TORCH_CHECK(down_svdq_l1.size(0) == num_experts && down_svdq_l1.size(1) == down_rank &&
                    down_svdq_l1.size(2) == intermediate_size,
                "down_svdq_l1 shape is inconsistent with down_rank/intermediate_size.");
    TORCH_CHECK(down_svdq_l2.size(0) == num_experts && down_svdq_l2.size(1) == hidden_size &&
                    down_svdq_l2.size(2) == down_rank,
                "down_svdq_l2 shape is inconsistent with down_rank/hidden_size.");
    TORCH_CHECK(gate_svdq_l2.data_ptr() != up_svdq_l2.data_ptr(),
                "gate_svdq_l2 and up_svdq_l2 must not share storage.");

    const at::Tensor& residual_w1 = first_svdq_tensor(weight1, "weight1");
    const at::Tensor& residual_w2 = first_svdq_tensor(weight2, "weight2");
    first_svdq_tensor(scale1, "scale1");
    first_svdq_tensor(scale2, "scale2");
    first_svdq_tensor(bias1, "bias1");
    first_svdq_tensor(bias2, "bias2");
    check_rank(residual_w1, 3, "weight1");
    check_rank(residual_w2, 3, "weight2");
    TORCH_CHECK(residual_w1.size(0) == num_experts && residual_w2.size(0) == num_experts,
                "residual W4A8 expert count must match SVDQ factor expert count.");
    check_rank(expert_token_nums, 1, "expert_token_nums");

    char* group_ep_ptr = group_string.data();
    EXEC_NPU_CMD(
        aclnnDispatchFFNCombineW4A8SVDQ,
        x,
        weight1,
        weight2,
        expert_idx,
        scale1,
        scale2,
        bias1,
        bias2,
        probs,
        gate_up_svdq_l1,
        gate_svdq_l2,
        up_svdq_l2,
        down_svdq_l1,
        down_svdq_l2,
        gate_rank,
        up_rank,
        down_rank,
        gate_rank_offset,
        up_rank_offset,
        x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
        group_ep_ptr,
        max_output_size,
        swiglu_limit,
        out,
        expert_token_nums);
    return {out, expert_token_nums};
}

}  // namespace vllm_ascend

#endif  // DISPATCH_FFN_COMBINE_W4_A8_SVDQ_TORCH_ADPT_H
