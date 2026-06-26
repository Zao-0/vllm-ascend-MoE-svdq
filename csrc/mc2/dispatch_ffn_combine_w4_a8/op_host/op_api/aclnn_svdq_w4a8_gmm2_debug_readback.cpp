/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details.
 */

#include "aclnn_svdq_w4a8_gmm2_debug_readback.h"

#ifdef __cplusplus
extern "C" {
#endif

enum NnopbaseHcclServerType {
    NNOPBASE_HCCL_SERVER_TYPE_AICPU = 0,
    NNOPBASE_HCCL_SERVER_TYPE_MTE,
    NNOPBASE_HCCL_SERVER_TYPE_END
};

extern aclnnStatus aclnnInnerSVDQW4A8GMM2DebugReadbackGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* xActiveMask, const aclTensor* hiddenXInt4Packed, const aclTensor* hiddenXScale,
    const aclTensor* externalExpertTokenNums, const char* group, int64_t maxOutputSize, bool transB,
    bool weightNz, double swigluLimit, const aclTensor* out, const aclTensor* expertTokenNums,
    const aclTensor* gmm2PostDequant, const aclTensor* hiddenXReadback, const aclTensor* hiddenScaleReadback,
    uint64_t* workspaceSize, aclOpExecutor** executor);

extern aclnnStatus aclnnInnerSVDQW4A8GMM2DebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
extern "C" void __attribute__((weak)) NnopbaseSetHcclServerType(void* executor, NnopbaseHcclServerType sType);

aclnnStatus aclnnSVDQW4A8GMM2DebugReadbackGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* xActiveMask, const aclTensor* hiddenXInt4Packed, const aclTensor* hiddenXScale,
    const aclTensor* externalExpertTokenNums, const char* group, int64_t maxOutputSize, double swigluLimit,
    const aclTensor* out, const aclTensor* expertTokenNums, const aclTensor* gmm2PostDequant,
    const aclTensor* hiddenXReadback, const aclTensor* hiddenScaleReadback, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    bool transB = false;
    bool weightNz = true;
    return aclnnInnerSVDQW4A8GMM2DebugReadbackGetWorkspaceSize(
        x, weight1, weight2, expertId, scale1, scale2, bias1, bias2, probs, xActiveMask,
        hiddenXInt4Packed, hiddenXScale, externalExpertTokenNums, group, maxOutputSize, transB, weightNz,
        swigluLimit, out, expertTokenNums, gmm2PostDequant, hiddenXReadback, hiddenScaleReadback, workspaceSize,
        executor);
}

aclnnStatus aclnnSVDQW4A8GMM2DebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    if (NnopbaseSetHcclServerType) {
        NnopbaseSetHcclServerType(executor, NNOPBASE_HCCL_SERVER_TYPE_MTE);
    }
    return aclnnInnerSVDQW4A8GMM2DebugReadback(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
