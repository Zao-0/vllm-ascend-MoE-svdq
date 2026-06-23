/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
class DispatchFFNCombineW4A8SVDQ : public OpDef {
 public:
  explicit DispatchFFNCombineW4A8SVDQ(const char *name) : OpDef(name) {
    this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_BF16})
        .Format({ge::FORMAT_ND})
        .UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("w1").ParamType(DYNAMIC).DataType({ge::DT_INT32}).Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ}).IgnoreContiguous();
    this->Input("w2").ParamType(DYNAMIC).DataType({ge::DT_INT32}).Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ}).IgnoreContiguous();
    this->Input("expertIdx").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("scale1").ParamType(DYNAMIC).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("scale2").ParamType(DYNAMIC).DataType({ge::DT_INT64}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("bias1").ParamType(DYNAMIC).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("bias2").ParamType(DYNAMIC).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("probs").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gateUpSvdqL1").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gateSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("upSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("downSvdqL1").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("downSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("xActiveMaskOptional").ParamType(OPTIONAL).DataType({ge::DT_BOOL}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("expert_token_nums").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

    this->Attr("gateRank").AttrType(REQUIRED).Int();
    this->Attr("upRank").AttrType(REQUIRED).Int();
    this->Attr("downRank").AttrType(REQUIRED).Int();
    this->Attr("gateRankOffset").AttrType(REQUIRED).Int();
    this->Attr("upRankOffset").AttrType(REQUIRED).Int();
    this->Attr("group").AttrType(REQUIRED).String();
    this->Attr("maxOutputSize").AttrType(REQUIRED).Int();
    this->Attr("transB").AttrType(OPTIONAL).Bool(false);
    this->Attr("weightNz").AttrType(OPTIONAL).Bool(true);
    this->Attr("swigluLimit").AttrType(OPTIONAL).Float(0.0f);

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910_93", aicore_config);
    this->MC2().HcclGroup("group");
  }
};

OP_ADD(DispatchFFNCombineW4A8SVDQ);

class SVDQLowRankDebugReadback : public OpDef {
 public:
  explicit SVDQLowRankDebugReadback(const char *name) : OpDef(name) {
    this->Input("routedX").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("hidden").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gateUpSvdqL1").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("gateSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("upSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("downSvdqL1").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("downSvdqL2").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Input("expertTokenNums").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

    this->Output("gateUpOutput").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("downOutput").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("gateUpAccumulator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
    this->Output("downAccumulator").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});

    this->Attr("gateRank").AttrType(REQUIRED).Int();
    this->Attr("upRank").AttrType(REQUIRED).Int();
    this->Attr("downRank").AttrType(REQUIRED).Int();
    this->Attr("gateRankOffset").AttrType(REQUIRED).Int();
    this->Attr("upRankOffset").AttrType(REQUIRED).Int();

    OpAICoreConfig aicore_config;
    aicore_config.DynamicCompileStaticFlag(true)
        .DynamicFormatFlag(true)
        .DynamicRankSupportFlag(true)
        .DynamicShapeSupportFlag(true)
        .NeedCheckSupportFlag(false)
        .PrecisionReduceFlag(true)
        .ExtendCfgInfo("aclnnSupport.value", "support_aclnn")
        .ExtendCfgInfo("jitCompile.flag", "static_false")
        .ExtendCfgInfo("multiKernelSupportDynamicGraph.value", "multi_kernel");
    this->AICore().AddConfig("ascend910_93", aicore_config);
  }
};

OP_ADD(SVDQLowRankDebugReadback);
}  // namespace ops
