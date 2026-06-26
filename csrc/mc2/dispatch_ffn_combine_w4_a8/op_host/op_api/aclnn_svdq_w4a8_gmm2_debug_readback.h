/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details.
 */

#ifndef OP_API_INC_SVDQ_W4A8_GMM2_DEBUG_READBACK_
#define OP_API_INC_SVDQ_W4A8_GMM2_DEBUG_READBACK_

#include "aclnn/aclnn_base.h"
#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQW4A8GMM2DebugReadbackGetWorkspaceSize(
    const aclTensor* x, const aclTensorList* weight1, const aclTensorList* weight2,
    const aclTensor* expertId, const aclTensorList* scale1, const aclTensorList* scale2,
    const aclTensorList* bias1, const aclTensorList* bias2, const aclTensor* probs,
    const aclTensor* xActiveMask, const aclTensor* hiddenXInt4Packed, const aclTensor* hiddenXScale,
    const aclTensor* externalExpertTokenNums, const char* group, int64_t maxOutputSize, double swigluLimit,
    const aclTensor* out, const aclTensor* expertTokenNums, const aclTensor* gmm2PostDequant,
    const aclTensor* hiddenXReadback, const aclTensor* hiddenScaleReadback, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnSVDQW4A8GMM2DebugReadback(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_SVDQ_W4A8_GMM2_DEBUG_READBACK_
