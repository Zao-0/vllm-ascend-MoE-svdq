/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details.
 */

#include "aclnn_svdq_w4a8_debug_readback.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerSVDQW4A8DebugReadbackGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* xActiveMask, const char* group, int64_t maxOutputSize, bool transB,
    bool weightNz, double swigluLimit, const aclTensor* out, const aclTensor* expertTokenNums,
    const aclTensor* routedXInt8, const aclTensor* routedXScale, const aclTensor* gmm1PostDequant,
    const aclTensor* gmm1HiddenPrequant, const aclTensor* gmm2PostDequant, uint64_t* workspaceSize,
    aclOpExecutor** executor);

extern aclnnStatus aclnnInnerSVDQW4A8DebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnSVDQW4A8DebugReadbackGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* xActiveMask, const char* group, int64_t maxOutputSize, double swigluLimit,
    const aclTensor* out, const aclTensor* expertTokenNums, const aclTensor* routedXInt8,
    const aclTensor* routedXScale, const aclTensor* gmm1PostDequant, const aclTensor* gmm1HiddenPrequant,
    const aclTensor* gmm2PostDequant,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    bool transB = false;
    bool weightNz = true;
    return aclnnInnerSVDQW4A8DebugReadbackGetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, bias1, bias2, probs, xActiveMask,
        group, maxOutputSize, transB, weightNz, swigluLimit, out, expertTokenNums,
        routedXInt8, routedXScale, gmm1PostDequant, gmm1HiddenPrequant, gmm2PostDequant, workspaceSize, executor);
}

aclnnStatus aclnnSVDQW4A8DebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerSVDQW4A8DebugReadback(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
