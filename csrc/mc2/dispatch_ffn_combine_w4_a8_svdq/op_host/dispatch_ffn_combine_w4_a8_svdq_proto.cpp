/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;
namespace ops {

static ge::graphStatus InferShapeDispatchFFNCombineW4A8SVDQ(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDispatchFFNCombineW4A8SVDQ(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DispatchFFNCombineW4A8SVDQ)
    .InferShape(InferShapeDispatchFFNCombineW4A8SVDQ)
    .InferDataType(InferDataTypeDispatchFFNCombineW4A8SVDQ);

static ge::graphStatus InferShapeSVDQLowRankDebugReadback(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSVDQLowRankDebugReadback(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SVDQLowRankDebugReadback)
    .InferShape(InferShapeSVDQLowRankDebugReadback)
    .InferDataType(InferDataTypeSVDQLowRankDebugReadback);
}  // namespace ops
