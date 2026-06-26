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
#ifndef SVDQ_W4A8_GMM2_DEBUG_READBACK_TORCH_ADPT_H
#define SVDQ_W4A8_GMM2_DEBUG_READBACK_TORCH_ADPT_H

#include <torch/extension.h>

#include "../dispatch_ffn_combine_w4_a8/op_host/op_api/aclnn_svdq_w4a8_gmm2_debug_readback.h"

#include <string>

namespace vllm_ascend {

namespace {

inline const at::Tensor& first_svdq_w4a8_gmm2_debug_tensor(const at::TensorList& tensors, const char* name)
{
    TORCH_CHECK(!tensors.empty(), name, " must not be an empty tensor list.");
    return tensors[0];
}

inline void check_svdq_w4a8_gmm2_debug_rank(const at::Tensor& tensor, int64_t rank, const char* name)
{
    TORCH_CHECK(tensor.dim() == rank, name, " must be rank-", rank, ", got ", tensor.sizes(), ".");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> svdq_w4a8_gmm2_debug_readback(
    const at::Tensor& x,
    const at::TensorList& weight1,
    const at::TensorList& weight2,
    const at::Tensor& expert_idx,
    const at::TensorList& scale1,
    const at::TensorList& scale2,
    const at::TensorList& bias1,
    const at::TensorList& bias2,
    const at::Tensor& probs,
    const at::Tensor& hidden_x_int4_packed,
    const at::Tensor& hidden_x_scale,
    const at::Tensor& external_expert_token_nums,
    c10::string_view group,
    int64_t max_output_size,
    const c10::optional<at::Tensor>& x_active_mask,
    double swiglu_limit)
{
    check_svdq_w4a8_gmm2_debug_rank(x, 2, "x");
    check_svdq_w4a8_gmm2_debug_rank(expert_idx, 2, "expert_idx");
    check_svdq_w4a8_gmm2_debug_rank(probs, 2, "probs");
    check_svdq_w4a8_gmm2_debug_rank(hidden_x_int4_packed, 2, "hidden_x_int4_packed");
    check_svdq_w4a8_gmm2_debug_rank(hidden_x_scale, 1, "hidden_x_scale");
    check_svdq_w4a8_gmm2_debug_rank(external_expert_token_nums, 2, "external_expert_token_nums");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf,
                "x must be BF16 or FP16, got ", x.scalar_type(), ".");
    TORCH_CHECK(expert_idx.scalar_type() == at::kInt, "expert_idx must be INT32.");
    TORCH_CHECK(probs.scalar_type() == at::kFloat, "probs must be FP32.");
    TORCH_CHECK(hidden_x_int4_packed.scalar_type() == at::kChar, "hidden_x_int4_packed must be INT8.");
    TORCH_CHECK(hidden_x_scale.scalar_type() == at::kFloat, "hidden_x_scale must be FP32.");
    TORCH_CHECK(external_expert_token_nums.scalar_type() == at::kInt,
                "external_expert_token_nums must be INT32.");
    TORCH_CHECK(expert_idx.sizes() == probs.sizes(), "expert_idx shape must match probs shape.");
    TORCH_CHECK(expert_idx.size(0) == x.size(0), "expert_idx token dimension must match x.");
    TORCH_CHECK(max_output_size > 0, "max_output_size must be positive.");
    TORCH_CHECK(swiglu_limit >= 0, "swiglu_limit must be non-negative.");
    std::string group_string(group);
    TORCH_CHECK(!group_string.empty(), "group must be a non-empty string.");
    if (x_active_mask.has_value()) {
        check_svdq_w4a8_gmm2_debug_rank(x_active_mask.value(), 1, "x_active_mask");
        TORCH_CHECK(x_active_mask.value().size(0) == x.size(0), "x_active_mask token dimension must match x.");
        TORCH_CHECK(x_active_mask.value().scalar_type() == at::kBool, "x_active_mask must be bool.");
    }

    const at::Tensor& residual_w1 = first_svdq_w4a8_gmm2_debug_tensor(weight1, "weight1");
    const at::Tensor& residual_w2 = first_svdq_w4a8_gmm2_debug_tensor(weight2, "weight2");
    const at::Tensor& residual_scale1 = first_svdq_w4a8_gmm2_debug_tensor(scale1, "scale1");
    const at::Tensor& residual_scale2 = first_svdq_w4a8_gmm2_debug_tensor(scale2, "scale2");
    const at::Tensor& residual_bias1 = first_svdq_w4a8_gmm2_debug_tensor(bias1, "bias1");
    const at::Tensor& residual_bias2 = first_svdq_w4a8_gmm2_debug_tensor(bias2, "bias2");
    check_svdq_w4a8_gmm2_debug_rank(residual_w1, 3, "weight1");
    check_svdq_w4a8_gmm2_debug_rank(residual_w2, 3, "weight2");
    TORCH_CHECK(residual_w1.scalar_type() == at::kInt, "weight1 must be INT32 packed W4.");
    TORCH_CHECK(residual_w2.scalar_type() == at::kInt, "weight2 must be INT32 packed W4.");
    TORCH_CHECK(residual_scale1.scalar_type() == at::kLong, "scale1 must be INT64 packed FP32 bits.");
    TORCH_CHECK(residual_scale2.scalar_type() == at::kLong, "scale2 must be INT64 packed FP32 bits.");
    TORCH_CHECK(residual_bias1.scalar_type() == at::kFloat, "bias1 must be FP32.");
    TORCH_CHECK(residual_bias2.scalar_type() == at::kFloat, "bias2 must be FP32.");
    TORCH_CHECK(residual_w1.size(0) == residual_w2.size(0), "weight1/weight2 expert counts must match.");
    TORCH_CHECK(residual_w1.size(1) == x.size(1), "weight1 K dimension must match x hidden dim.");
    TORCH_CHECK(residual_w2.size(2) * 8 == x.size(1), "weight2 packed N dimension must match x hidden dim.");

    const auto num_experts = residual_w1.size(0);
    const auto hidden_size = x.size(1);
    const auto gmm1_columns = residual_w1.size(2) * 8;
    TORCH_CHECK(gmm1_columns % 2 == 0, "GMM1 output columns must contain gate and up halves.");
    const auto intermediate_size = gmm1_columns / 2;
    TORCH_CHECK(residual_w2.size(1) == intermediate_size, "weight2 K dimension must match GMM1 half width.");
    TORCH_CHECK(hidden_x_int4_packed.size(0) == max_output_size,
                "hidden_x_int4_packed row count must equal max_output_size.");
    TORCH_CHECK(hidden_x_int4_packed.size(1) == intermediate_size,
                "hidden_x_int4_packed width must match GMM2 K.");
    TORCH_CHECK(hidden_x_scale.size(0) == max_output_size, "hidden_x_scale row count must equal max_output_size.");
    TORCH_CHECK(external_expert_token_nums.size(0) == 1, "external_expert_token_nums must be rank-2 [1, experts].");
    TORCH_CHECK(external_expert_token_nums.size(1) == num_experts,
                "external_expert_token_nums expert count must match weights.");

    auto out = at::empty({x.size(0), hidden_size}, x.options());
    auto expert_token_nums = at::empty({1, num_experts}, expert_idx.options());
    auto gmm2_post_dequant = at::zeros({max_output_size, hidden_size}, x.options().dtype(at::kFloat));
    auto hidden_x_readback = at::zeros({max_output_size, intermediate_size}, hidden_x_int4_packed.options());
    auto hidden_scale_readback = at::zeros({max_output_size}, hidden_x_scale.options());
    auto gmm2_accumulator_int32 = at::zeros({max_output_size * 2, hidden_size}, expert_idx.options());

    char* group_ep_ptr = group_string.data();
    EXEC_NPU_CMD(
        aclnnSVDQW4A8GMM2DebugReadback,
        x,
        weight1,
        weight2,
        expert_idx,
        scale1,
        scale2,
        bias1,
        bias2,
        probs,
        x_active_mask.has_value() ? x_active_mask.value() : at::Tensor(),
        hidden_x_int4_packed,
        hidden_x_scale,
        external_expert_token_nums,
        group_ep_ptr,
        max_output_size,
        swiglu_limit,
        out,
        expert_token_nums,
        gmm2_post_dequant,
        hidden_x_readback,
        hidden_scale_readback,
        gmm2_accumulator_int32);
    return {gmm2_post_dequant, hidden_x_readback, hidden_scale_readback, gmm2_accumulator_int32};
}

}  // namespace vllm_ascend

#endif  // SVDQ_W4A8_GMM2_DEBUG_READBACK_TORCH_ADPT_H
