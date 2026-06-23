/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_dispatch_ffn_combine_w4_a8_svdq.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerDispatchFFNCombineW4A8SVDQGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* gateUpSvdqL1, const aclTensor* gateSvdqL2, const aclTensor* upSvdqL2,
    const aclTensor* downSvdqL1, const aclTensor* downSvdqL2, int64_t gateRank, int64_t upRank,
    int64_t downRank, int64_t gateRankOffset, int64_t upRankOffset, const aclTensor* xActiveMask,
    const char* group, int64_t maxOutputSize, bool transB, bool weightNz, double swigluLimit,
    const aclTensor* out, const aclTensor* expertTokenNums, uint64_t* workspaceSize,
    aclOpExecutor** executor);
extern aclnnStatus aclnnInnerDispatchFFNCombineW4A8SVDQ(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnDispatchFFNCombineW4A8SVDQGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* gateUpSvdqL1, const aclTensor* gateSvdqL2, const aclTensor* upSvdqL2,
    const aclTensor* downSvdqL1, const aclTensor* downSvdqL2, int64_t gateRank, int64_t upRank,
    int64_t downRank, int64_t gateRankOffset, int64_t upRankOffset, const aclTensor* xActiveMask,
    const char* group, int64_t maxOutputSize, double swigluLimit, const aclTensor* out,
    const aclTensor* expertTokenNums, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool transB = false;
    bool weightNz = true;
    return aclnnInnerDispatchFFNCombineW4A8SVDQGetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, bias1, bias2, probs, gateUpSvdqL1, gateSvdqL2,
        upSvdqL2, downSvdqL1, downSvdqL2, gateRank, upRank, downRank, gateRankOffset, upRankOffset,
        xActiveMask, group, maxOutputSize, transB, weightNz, swigluLimit, out, expertTokenNums,
        workspaceSize, executor);
}

aclnnStatus aclnnDispatchFFNCombineW4A8SVDQ(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerDispatchFFNCombineW4A8SVDQ(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
