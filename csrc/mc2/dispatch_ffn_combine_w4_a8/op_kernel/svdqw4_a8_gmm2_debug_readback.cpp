/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "dispatch_ffn_combine_w4_a8_tiling.h"
#include "dispatch_ffn_combine_w4_a8.h"

using namespace AscendC;
using namespace DispatchFFNCombineW4A8Impl;

extern "C" __global__ __aicore__ void svdqw4_a8_gmm2_debug_readback(
    GM_ADDR x, GM_ADDR w1, GM_ADDR w2, GM_ADDR expertId, GM_ADDR scale1, GM_ADDR scale2,
    GM_ADDR bias1, GM_ADDR bias2, GM_ADDR probs, GM_ADDR xActiveMask, GM_ADDR hiddenXInt4Packed,
    GM_ADDR hiddenXScale, GM_ADDR externalExpertTokenNums, GM_ADDR c, GM_ADDR expertTokenNums,
    GM_ADDR gmm2PostDequant, GM_ADDR hiddenXReadback, GM_ADDR hiddenScaleReadback, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DispatchFFNCombineW4A8TilingData);
    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(DispatchFFNCombineW4A8TilingData, tilingData, tilingGM);
        DispatchFFNCombineW4A8<DTYPE_A, DTYPE_W1, DTYPE_OUT, false, true> op;
        op.InitGMM2OnlyFromPacked(x, w1, w2, expertId, scale1, scale2, bias1, bias2, probs, xActiveMask, c,
            expertTokenNums, workspaceGM, tilingGM, hiddenXInt4Packed, hiddenXScale, externalExpertTokenNums,
            gmm2PostDequant, hiddenXReadback, hiddenScaleReadback);
        op.Process();
    }
}
