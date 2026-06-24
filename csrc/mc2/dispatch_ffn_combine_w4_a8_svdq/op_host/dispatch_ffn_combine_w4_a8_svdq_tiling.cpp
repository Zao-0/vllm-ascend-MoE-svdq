/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>

#include "dispatch_ffn_combine_w4_a8_svdq_tiling.h"
#include "lowrank/svdq_lowrank_debug_readback_tiling.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling_base/error_log.h"

using namespace ge;
using DispatchFFNCombineW4A8SVDQImpl::SVDQLowRankDebugTilingData;

namespace {
constexpr const char* K_INNER_DEBUG = "DispatchFFNCombineW4A8SVDQ Tiling";

constexpr uint32_t ATTR_GATE_RANK_INDEX = 0;
constexpr uint32_t ATTR_UP_RANK_INDEX = 1;
constexpr uint32_t ATTR_DOWN_RANK_INDEX = 2;
constexpr uint32_t ATTR_GATE_RANK_OFFSET_INDEX = 3;
constexpr uint32_t ATTR_UP_RANK_OFFSET_INDEX = 4;
constexpr uint32_t ATTR_GROUP_INDEX = 5;
constexpr uint32_t ATTR_MAX_OUTPUT_SIZE_INDEX = 6;
constexpr uint32_t ATTR_SWIGLU_LIMIT_INDEX = 9;

constexpr uint32_t X_INDEX = 0;
constexpr uint32_t WEIGHT1_INDEX = 1;
constexpr uint32_t WEIGHT2_INDEX = 2;
constexpr uint32_t EXPERT_ID_INDEX = 3;
constexpr uint32_t SCALE1_INDEX = 4;
constexpr uint32_t SCALE2_INDEX = 5;
constexpr uint32_t BIAS1_INDEX = 6;
constexpr uint32_t BIAS2_INDEX = 7;
constexpr uint32_t PROBS_INDEX = 8;
constexpr uint32_t GATE_UP_SVDQ_L1_INDEX = 9;
constexpr uint32_t GATE_SVDQ_L2_INDEX = 10;
constexpr uint32_t UP_SVDQ_L2_INDEX = 11;
constexpr uint32_t DOWN_SVDQ_L1_INDEX = 12;
constexpr uint32_t DOWN_SVDQ_L2_INDEX = 13;
constexpr uint32_t X_ACTIVE_MASK_INDEX = 14;
constexpr uint32_t OUT_INDEX = 0;
constexpr uint32_t EXPERT_TOKEN_NUMS_INDEX = 1;

constexpr uint32_t DEBUG_ROUTED_X_INDEX = 0;
constexpr uint32_t DEBUG_HIDDEN_INDEX = 1;
constexpr uint32_t DEBUG_GATE_UP_SVDQ_L1_INDEX = 2;
constexpr uint32_t DEBUG_GATE_SVDQ_L2_INDEX = 3;
constexpr uint32_t DEBUG_UP_SVDQ_L2_INDEX = 4;
constexpr uint32_t DEBUG_DOWN_SVDQ_L1_INDEX = 5;
constexpr uint32_t DEBUG_DOWN_SVDQ_L2_INDEX = 6;
constexpr uint32_t DEBUG_EXPERT_TOKEN_NUMS_INDEX = 7;
constexpr uint32_t DEBUG_GATE_UP_OUTPUT_INDEX = 0;
constexpr uint32_t DEBUG_DOWN_OUTPUT_INDEX = 1;
constexpr uint32_t DEBUG_GATE_UP_ACCUMULATOR_INDEX = 2;
constexpr uint32_t DEBUG_DOWN_ACCUMULATOR_INDEX = 3;

constexpr uint64_t SVDQ_WORKSPACE_ALIGNMENT = 512;
constexpr uint64_t SVDQ_SYSTEM_WORKSPACE = 16UL * 1024UL * 1024UL;
constexpr uint64_t INT8_BYTES = 1;
constexpr uint64_t INT32_BYTES = 4;
constexpr uint64_t BF16_BYTES = 2;
constexpr uint64_t FP32_BYTES = 4;
constexpr uint32_t SVDQ_LOWRANK_ROW_TILE = 16;
constexpr uint32_t SVDQ_LOWRANK_OUTPUT_COLUMN_TILE = 64;
constexpr uint32_t SVDQ_LOWRANK_K_TILE = 64;
}  // namespace

namespace optiling {

static uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

static void SetWorkspaceRegion(
    DispatchFFNCombineW4A8SVDQTilingData* tilingData, uint32_t regionId, uint64_t& offset,
    uint64_t size, uint32_t dtype, uint32_t producerStage, uint32_t consumerStage, uint32_t lifetimeId)
{
    auto& region = tilingData->workspaceRegions[regionId];
    region.offset = AlignUp(offset, SVDQ_WORKSPACE_ALIGNMENT);
    region.size = AlignUp(size, SVDQ_WORKSPACE_ALIGNMENT);
    region.dtype = dtype;
    region.producerStage = producerStage;
    region.consumerStage = consumerStage;
    region.lifetimeId = lifetimeId;
    offset = region.offset + region.size;
}

static void BuildWorkspaceMap(DispatchFFNCombineW4A8SVDQTilingData* tilingData)
{
    auto& info = tilingData->info;
    const uint64_t activeSlots = static_cast<uint64_t>(info.m) * static_cast<uint64_t>(info.topK);
    const uint64_t routedRows = static_cast<uint64_t>(info.maxOutputSize);
    const uint64_t hiddenSize = static_cast<uint64_t>(info.hiddenSize);
    const uint64_t intermediateSize = static_cast<uint64_t>(info.intermediateSize);
    const uint64_t gateUpSize = intermediateSize * 2;
    uint64_t offset = 0;

    SetWorkspaceRegion(tilingData, SVDQ_REGION_EXPANDED_ROW_IDX, offset, activeSlots * INT32_BYTES,
        SVDQ_DTYPE_INT32, SVDQ_STAGE_BF16_DISPATCH, SVDQ_STAGE_UNPERMUTE_COMBINE, 1);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_ROUTED_X, offset, routedRows * hiddenSize * BF16_BYTES,
        SVDQ_DTYPE_BF16, SVDQ_STAGE_BF16_DISPATCH, SVDQ_STAGE_QUANT_1, 2);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_X_Q, offset, routedRows * hiddenSize * INT8_BYTES,
        SVDQ_DTYPE_INT8, SVDQ_STAGE_QUANT_1, SVDQ_STAGE_W4A8_GEMM_1, 3);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_X_SCALE, offset, routedRows * FP32_BYTES,
        SVDQ_DTYPE_FP32, SVDQ_STAGE_QUANT_1, SVDQ_STAGE_MIXED_EPILOGUE_1, 4);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_PROJECTION_1, offset, routedRows * gateUpSize * BF16_BYTES,
        SVDQ_DTYPE_BF16, SVDQ_STAGE_LOWRANK_1, SVDQ_STAGE_MIXED_EPILOGUE_1, 5);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_ACCUMULATOR_1, offset, routedRows * gateUpSize * INT32_BYTES,
        SVDQ_DTYPE_INT32, SVDQ_STAGE_W4A8_GEMM_1, SVDQ_STAGE_MIXED_EPILOGUE_1, 6);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_HIDDEN, offset, routedRows * intermediateSize * BF16_BYTES,
        SVDQ_DTYPE_BF16, SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_STAGE_QUANT_2, 7);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_HIDDEN_Q, offset, routedRows * intermediateSize * INT8_BYTES,
        SVDQ_DTYPE_INT8, SVDQ_STAGE_QUANT_2, SVDQ_STAGE_W4A8_GEMM_2, 8);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_HIDDEN_SCALE, offset, routedRows * FP32_BYTES,
        SVDQ_DTYPE_FP32, SVDQ_STAGE_QUANT_2, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, 9);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_PROJECTION_2, offset, routedRows * hiddenSize * BF16_BYTES,
        SVDQ_DTYPE_BF16, SVDQ_STAGE_LOWRANK_2, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, 10);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_ACCUMULATOR_2, offset, routedRows * hiddenSize * INT32_BYTES,
        SVDQ_DTYPE_INT32, SVDQ_STAGE_W4A8_GEMM_2, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, 11);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_LOWRANK_ACCUMULATOR_1, offset, routedRows * gateUpSize * FP32_BYTES,
        SVDQ_DTYPE_FP32, SVDQ_STAGE_LOWRANK_1, SVDQ_STAGE_LOWRANK_1, 12);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_LOWRANK_ACCUMULATOR_2, offset, routedRows * hiddenSize * FP32_BYTES,
        SVDQ_DTYPE_FP32, SVDQ_STAGE_LOWRANK_2, SVDQ_STAGE_LOWRANK_2, 13);
    SetWorkspaceRegion(tilingData, SVDQ_REGION_PEER_OUTPUT, offset, routedRows * hiddenSize * BF16_BYTES,
        SVDQ_DTYPE_BF16, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_UNPERMUTE_COMBINE, 14);

    info.workspaceBytes = offset;
}

static void SetSyncFlag(
    DispatchFFNCombineW4A8SVDQTilingData* tilingData, uint32_t flagId, uint32_t producerStage,
    uint32_t consumerStage, uint32_t workspaceRegionId, uint32_t signalWaitIndex)
{
    auto& flag = tilingData->syncFlags[flagId];
    flag.flagId = flagId;
    flag.producerStage = producerStage;
    flag.consumerStage = consumerStage;
    flag.workspaceRegionId = workspaceRegionId;
    flag.producerSignalIndex = signalWaitIndex;
    flag.consumerWaitIndex = signalWaitIndex;
}

static void BuildSyncFlagTable(DispatchFFNCombineW4A8SVDQTilingData* tilingData)
{
    SetSyncFlag(tilingData, SVDQ_SYNC_DISPATCH_TO_QUANT_1, SVDQ_STAGE_BF16_DISPATCH,
        SVDQ_STAGE_QUANT_1, SVDQ_REGION_ROUTED_X, 0);
    SetSyncFlag(tilingData, SVDQ_SYNC_DISPATCH_TO_LOWRANK_1, SVDQ_STAGE_BF16_DISPATCH,
        SVDQ_STAGE_LOWRANK_1, SVDQ_REGION_ROUTED_X, 1);
    SetSyncFlag(tilingData, SVDQ_SYNC_QUANT_1_TO_W4A8_GEMM_1, SVDQ_STAGE_QUANT_1,
        SVDQ_STAGE_W4A8_GEMM_1, SVDQ_REGION_X_Q, 2);
    SetSyncFlag(tilingData, SVDQ_SYNC_QUANT_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_QUANT_1,
        SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_X_SCALE, 3);
    SetSyncFlag(tilingData, SVDQ_SYNC_LOWRANK_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_LOWRANK_1,
        SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_PROJECTION_1, 4);
    SetSyncFlag(tilingData, SVDQ_SYNC_W4A8_GEMM_1_TO_MIXED_EPILOGUE_1, SVDQ_STAGE_W4A8_GEMM_1,
        SVDQ_STAGE_MIXED_EPILOGUE_1, SVDQ_REGION_ACCUMULATOR_1, 5);
    SetSyncFlag(tilingData, SVDQ_SYNC_MIXED_EPILOGUE_1_TO_QUANT_2, SVDQ_STAGE_MIXED_EPILOGUE_1,
        SVDQ_STAGE_QUANT_2, SVDQ_REGION_HIDDEN, 6);
    SetSyncFlag(tilingData, SVDQ_SYNC_MIXED_EPILOGUE_1_TO_LOWRANK_2, SVDQ_STAGE_MIXED_EPILOGUE_1,
        SVDQ_STAGE_LOWRANK_2, SVDQ_REGION_HIDDEN, 7);
    SetSyncFlag(tilingData, SVDQ_SYNC_QUANT_2_TO_W4A8_GEMM_2, SVDQ_STAGE_QUANT_2,
        SVDQ_STAGE_W4A8_GEMM_2, SVDQ_REGION_HIDDEN_Q, 8);
    SetSyncFlag(tilingData, SVDQ_SYNC_QUANT_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_QUANT_2,
        SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_HIDDEN_SCALE, 9);
    SetSyncFlag(tilingData, SVDQ_SYNC_LOWRANK_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_LOWRANK_2,
        SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_PROJECTION_2, 10);
    SetSyncFlag(tilingData, SVDQ_SYNC_W4A8_GEMM_2_TO_MIXED_OUTPUT_EPILOGUE, SVDQ_STAGE_W4A8_GEMM_2,
        SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE, SVDQ_REGION_ACCUMULATOR_2, 11);
    SetSyncFlag(tilingData, SVDQ_SYNC_MIXED_OUTPUT_EPILOGUE_TO_UNPERMUTE, SVDQ_STAGE_MIXED_OUTPUT_EPILOGUE,
        SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_PEER_OUTPUT, 12);
    SetSyncFlag(tilingData, SVDQ_SYNC_DISPATCH_METADATA_TO_UNPERMUTE, SVDQ_STAGE_BF16_DISPATCH,
        SVDQ_STAGE_UNPERMUTE_COMBINE, SVDQ_REGION_EXPANDED_ROW_IDX, 13);

    tilingData->info.syncFlagCount = SVDQ_SYNC_FLAG_COUNT;
}

static void SetBF16StageShape(
    DispatchFFNCombineW4A8SVDQTilingData* tilingData, uint32_t stageId, uint32_t factorId,
    uint32_t inputRegionId, uint32_t outputRegionId, uint32_t m, uint32_t k, uint32_t n,
    uint32_t inputColumnOffset, uint32_t outputColumnOffset, uint32_t factorColumnOffset)
{
    auto& stage = tilingData->bf16StageShapes[stageId];
    stage.stageId = stageId;
    stage.factorId = factorId;
    stage.inputRegionId = inputRegionId;
    stage.outputRegionId = outputRegionId;
    stage.m = m;
    stage.k = k;
    stage.n = n;
    stage.inputColumnOffset = inputColumnOffset;
    stage.outputColumnOffset = outputColumnOffset;
    stage.factorColumnOffset = factorColumnOffset;
}

static void BuildBF16StageShapeTable(DispatchFFNCombineW4A8SVDQTilingData* tilingData)
{
    auto& info = tilingData->info;
    const uint32_t routedRows = info.maxOutputSize;
    const uint32_t gateUpRank = info.gateRank + info.upRank;
    const uint32_t gateOutputOffset = 0;
    const uint32_t upOutputOffset = info.intermediateSize;

    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_ROUTING, SVDQ_INVALID_ID,
        SVDQ_INVALID_ID, SVDQ_REGION_ROUTED_X, routedRows, info.hiddenSize, info.hiddenSize, 0, 0, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_GATE_UP_L1_GEMM, SVDQ_FACTOR_GATE_UP_L1,
        SVDQ_REGION_ROUTED_X, SVDQ_REGION_PROJECTION_1, routedRows, info.hiddenSize, gateUpRank, 0, 0, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_GATE_UP_RANK_SPLIT, SVDQ_INVALID_ID,
        SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1, routedRows, gateUpRank, gateUpRank, 0, 0, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_GATE_L2_GEMM, SVDQ_FACTOR_GATE_L2,
        SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1, routedRows, info.gateRank, info.intermediateSize,
        info.gateRankOffset, gateOutputOffset, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_UP_L2_GEMM, SVDQ_FACTOR_UP_L2,
        SVDQ_REGION_PROJECTION_1, SVDQ_REGION_PROJECTION_1, routedRows, info.upRank, info.intermediateSize,
        info.upRankOffset, upOutputOffset, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_DOWN_L1_GEMM, SVDQ_FACTOR_DOWN_L1,
        SVDQ_REGION_HIDDEN, SVDQ_REGION_PROJECTION_2, routedRows, info.intermediateSize, info.downRank, 0, 0, 0);
    SetBF16StageShape(tilingData, SVDQ_BF16_STAGE_DOWN_L2_GEMM, SVDQ_FACTOR_DOWN_L2,
        SVDQ_REGION_PROJECTION_2, SVDQ_REGION_PROJECTION_2, routedRows, info.downRank, info.hiddenSize, 0, 0, 0);
}

static void SetResidualStageShape(
    DispatchFFNCombineW4A8SVDQTilingData* tilingData, uint32_t stageId, uint32_t inputRegionId,
    uint32_t scaleRegionId, uint32_t outputRegionId, uint32_t m, uint32_t k, uint32_t n,
    uint32_t residualWeightSlot, uint32_t residualScaleSlot, bool residualOnly)
{
    auto& stage = tilingData->residualStageShapes[stageId];
    stage.stageId = stageId;
    stage.inputRegionId = inputRegionId;
    stage.scaleRegionId = scaleRegionId;
    stage.outputRegionId = outputRegionId;
    stage.m = m;
    stage.k = k;
    stage.n = n;
    stage.residualWeightSlot = residualWeightSlot;
    stage.residualScaleSlot = residualScaleSlot;
    stage.residualOnly = residualOnly;
}

static void BuildResidualStageShapeTable(DispatchFFNCombineW4A8SVDQTilingData* tilingData)
{
    auto& info = tilingData->info;
    const uint32_t routedRows = info.maxOutputSize;
    constexpr uint32_t weight1Slot = 1;
    constexpr uint32_t weight2Slot = 2;
    constexpr uint32_t scale1Slot = 4;
    constexpr uint32_t scale2Slot = 5;

    SetResidualStageShape(tilingData, SVDQ_RESIDUAL_STAGE_QUANT_ROUTED_INPUT,
        SVDQ_REGION_ROUTED_X, SVDQ_REGION_X_SCALE, SVDQ_REGION_X_Q,
        routedRows, info.hiddenSize, info.hiddenSize, SVDQ_INVALID_ID, SVDQ_INVALID_ID, true);
    SetResidualStageShape(tilingData, SVDQ_RESIDUAL_STAGE_W4A8_GMM1,
        SVDQ_REGION_X_Q, SVDQ_REGION_X_SCALE, SVDQ_REGION_ACCUMULATOR_1,
        routedRows, info.hiddenSize, info.intermediateSize * 2, weight1Slot, scale1Slot, true);
    SetResidualStageShape(tilingData, SVDQ_RESIDUAL_STAGE_QUANT_HIDDEN,
        SVDQ_REGION_HIDDEN, SVDQ_REGION_HIDDEN_SCALE, SVDQ_REGION_HIDDEN_Q,
        routedRows, info.intermediateSize, info.intermediateSize, SVDQ_INVALID_ID, SVDQ_INVALID_ID, true);
    SetResidualStageShape(tilingData, SVDQ_RESIDUAL_STAGE_W4A8_GMM2,
        SVDQ_REGION_HIDDEN_Q, SVDQ_REGION_HIDDEN_SCALE, SVDQ_REGION_ACCUMULATOR_2,
        routedRows, info.intermediateSize, info.hiddenSize, weight2Slot, scale2Slot, true);
}

static void SetLowRankInvocation(
    DispatchFFNCombineW4A8SVDQTilingData* tilingData, uint32_t invocationId, uint32_t inputRegionId,
    uint32_t outputRegionId, uint32_t downFactorId, uint32_t upFactorId, uint32_t secondUpFactorId,
    uint32_t m, uint32_t inputColumns, uint32_t rankColumns, uint32_t secondRankColumns,
    uint32_t outputColumns, uint32_t inputColumnOffset, uint32_t outputColumnOffset,
    uint32_t secondInputColumnOffset, uint32_t secondOutputColumnOffset, uint32_t coreCount,
    uint32_t accumulatorRegionId)
{
    auto& invocation = tilingData->lowRankInvocations[invocationId];
    invocation.invocationId = invocationId;
    invocation.inputRegionId = inputRegionId;
    invocation.outputRegionId = outputRegionId;
    invocation.downFactorId = downFactorId;
    invocation.upFactorId = upFactorId;
    invocation.secondUpFactorId = secondUpFactorId;
    invocation.m = m;
    invocation.inputColumns = inputColumns;
    invocation.rankColumns = rankColumns;
    invocation.secondRankColumns = secondRankColumns;
    invocation.outputColumns = outputColumns;
    invocation.inputColumnOffset = inputColumnOffset;
    invocation.outputColumnOffset = outputColumnOffset;
    invocation.secondInputColumnOffset = secondInputColumnOffset;
    invocation.secondOutputColumnOffset = secondOutputColumnOffset;
    invocation.rowTile = SVDQ_LOWRANK_ROW_TILE;
    invocation.outputColumnTile = SVDQ_LOWRANK_OUTPUT_COLUMN_TILE;
    invocation.kTile = SVDQ_LOWRANK_K_TILE;
    invocation.coreCount = coreCount;
    invocation.accumulatorRegionId = accumulatorRegionId;
}

static void BuildLowRankInvocationTable(DispatchFFNCombineW4A8SVDQTilingData* tilingData)
{
    auto& info = tilingData->info;
    const uint32_t routedRows = info.maxOutputSize;

    SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_GATE_UP,
        SVDQ_REGION_ROUTED_X, SVDQ_REGION_PROJECTION_1, SVDQ_FACTOR_GATE_UP_L1, SVDQ_FACTOR_GATE_L2,
        SVDQ_FACTOR_UP_L2, routedRows, info.hiddenSize, info.gateRank, info.upRank,
        info.intermediateSize * 2, info.gateRankOffset, 0, info.upRankOffset, info.intermediateSize,
        info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_1);
    SetLowRankInvocation(tilingData, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_DOWN,
        SVDQ_REGION_HIDDEN, SVDQ_REGION_PROJECTION_2, SVDQ_FACTOR_DOWN_L1, SVDQ_FACTOR_DOWN_L2,
        SVDQ_INVALID_ID, routedRows, info.intermediateSize, info.downRank, 0, info.hiddenSize, 0, 0, 0, 0,
        info.lowRankCoreCount, SVDQ_REGION_LOWRANK_ACCUMULATOR_2);
}

static ge::graphStatus DispatchFFNCombineW4A8SVDQGetPlatformInfoAndSetTiling(
    gert::TilingContext* context, DispatchFFNCombineW4A8SVDQInfo& info)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    info.lowRankCoreCount = blockDim;
    context->SetBlockDim(blockDim);
    context->SetTilingKey(1000000);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineW4A8SVDQCheckAttrAndSetTiling(
    gert::TilingContext* context, DispatchFFNCombineW4A8SVDQInfo& info)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "attrs is null."), return ge::GRAPH_FAILED);

    auto gateRank = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_GATE_RANK_INDEX));
    auto upRank = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_UP_RANK_INDEX));
    auto downRank = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_DOWN_RANK_INDEX));
    auto gateRankOffset = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_GATE_RANK_OFFSET_INDEX));
    auto upRankOffset = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_UP_RANK_OFFSET_INDEX));
    auto group = attrs->GetAttrPointer<char>(static_cast<int>(ATTR_GROUP_INDEX));
    auto maxOutputSize = attrs->GetAttrPointer<int>(static_cast<int>(ATTR_MAX_OUTPUT_SIZE_INDEX));
    auto swigluLimit = attrs->GetAttrPointer<float>(static_cast<int>(ATTR_SWIGLU_LIMIT_INDEX));

    OP_TILING_CHECK(gateRank == nullptr || upRank == nullptr || downRank == nullptr,
        OP_LOGE(K_INNER_DEBUG, "rank attrs are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gateRankOffset == nullptr || upRankOffset == nullptr,
        OP_LOGE(K_INNER_DEBUG, "rank offset attrs are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(group == nullptr || strlen(group) == 0,
        OP_LOGE(K_INNER_DEBUG, "group is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(maxOutputSize == nullptr || *maxOutputSize <= 0,
        OP_LOGE(K_INNER_DEBUG, "maxOutputSize is invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*gateRank <= 0 || *upRank <= 0 || *downRank <= 0,
        OP_LOGE(K_INNER_DEBUG, "SVDQ ranks must be positive."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*gateRankOffset != 0,
        OP_LOGE(K_INNER_DEBUG, "gateRankOffset must be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*upRankOffset != *gateRank,
        OP_LOGE(K_INNER_DEBUG, "upRankOffset must equal gateRank."), return ge::GRAPH_FAILED);

    info.gateRank = static_cast<uint32_t>(*gateRank);
    info.upRank = static_cast<uint32_t>(*upRank);
    info.downRank = static_cast<uint32_t>(*downRank);
    info.gateRankOffset = static_cast<uint32_t>(*gateRankOffset);
    info.upRankOffset = static_cast<uint32_t>(*upRankOffset);
    info.maxOutputSize = static_cast<uint32_t>(*maxOutputSize);
    info.swigluLimit = swigluLimit == nullptr ? 0.0f : *swigluLimit;
    info.worldSize = 1;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRank3Shape(
    gert::TilingContext* context, uint32_t index, const char* name, int64_t expectedDim0)
{
    const gert::StorageShape* shape = context->GetInputShape(index);
    OP_TILING_CHECK(shape == nullptr, OP_LOGE(K_INNER_DEBUG, "%s shape is null.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != 3,
        OP_LOGE(K_INNER_DEBUG, "%s must be rank-3.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expectedDim0 >= 0 && shape->GetStorageShape().GetDim(0) != expectedDim0,
        OP_LOGE(K_INNER_DEBUG, "%s expert dim mismatch.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckRequiredInputDType(
    gert::TilingContext* context, uint32_t index, const char* name, ge::DataType expected)
{
    auto desc = context->GetInputDesc(index);
    OP_TILING_CHECK(desc == nullptr, OP_LOGE(K_INNER_DEBUG, "%s desc is null.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(desc->GetDataType() != expected,
        OP_LOGE(K_INNER_DEBUG, "%s dtype mismatch.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDynamicInputDType(
    gert::TilingContext* context, uint32_t index, const char* name, ge::DataType expected)
{
    auto desc = context->GetDynamicInputDesc(index, 0);
    OP_TILING_CHECK(desc == nullptr, OP_LOGE(K_INNER_DEBUG, "%s dynamic desc is null.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(desc->GetDataType() != expected,
        OP_LOGE(K_INNER_DEBUG, "%s dtype mismatch.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOutputDType(
    gert::TilingContext* context, uint32_t index, const char* name, ge::DataType expected)
{
    auto desc = context->GetOutputDesc(index);
    OP_TILING_CHECK(desc == nullptr, OP_LOGE(K_INNER_DEBUG, "%s output desc is null.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(desc->GetDataType() != expected,
        OP_LOGE(K_INNER_DEBUG, "%s output dtype mismatch.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineW4A8SVDQCheckDType(gert::TilingContext* context)
{
    OP_TILING_CHECK(CheckRequiredInputDType(context, X_INDEX, "x", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "x dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, WEIGHT1_INDEX, "w1", ge::DT_INT32) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "w1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, WEIGHT2_INDEX, "w2", ge::DT_INT32) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "w2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, EXPERT_ID_INDEX, "expertIdx", ge::DT_INT32) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "expertIdx dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, SCALE1_INDEX, "scale1", ge::DT_INT64) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "scale1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, SCALE2_INDEX, "scale2", ge::DT_INT64) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "scale2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, BIAS1_INDEX, "bias1", ge::DT_FLOAT) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "bias1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDynamicInputDType(context, BIAS2_INDEX, "bias2", ge::DT_FLOAT) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "bias2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, PROBS_INDEX, "probs", ge::DT_FLOAT) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "probs dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, GATE_UP_SVDQ_L1_INDEX, "gateUpSvdqL1", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "gateUpSvdqL1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, GATE_SVDQ_L2_INDEX, "gateSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "gateSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, UP_SVDQ_L2_INDEX, "upSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "upSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DOWN_SVDQ_L1_INDEX, "downSvdqL1", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "downSvdqL1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DOWN_SVDQ_L2_INDEX, "downSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "downSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, OUT_INDEX, "out", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "out dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, EXPERT_TOKEN_NUMS_INDEX, "expertTokenNums", ge::DT_INT32) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "expertTokenNums dtype check failed."), return ge::GRAPH_FAILED);

    auto xActiveMaskDesc = context->GetOptionalInputDesc(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskDesc != nullptr) {
        OP_TILING_CHECK(xActiveMaskDesc->GetDataType() != ge::DT_BOOL,
            OP_LOGE(K_INNER_DEBUG, "xActiveMask dtype mismatch."), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineW4A8SVDQCheckShapeAndSetTiling(
    gert::TilingContext* context, DispatchFFNCombineW4A8SVDQInfo& info)
{
    const gert::StorageShape* xShape = context->GetInputShape(X_INDEX);
    OP_TILING_CHECK(xShape == nullptr, OP_LOGE(K_INNER_DEBUG, "x shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(xShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(K_INNER_DEBUG, "x must be rank-2."), return ge::GRAPH_FAILED);
    info.m = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(0));
    info.hiddenSize = static_cast<uint32_t>(xShape->GetStorageShape().GetDim(1));

    auto expertIdxTensor = context->GetInputTensor(EXPERT_ID_INDEX);
    OP_TILING_CHECK(expertIdxTensor == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertIdx tensor is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertIdxTensor->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(K_INNER_DEBUG, "expertIdx must be rank-2."), return ge::GRAPH_FAILED);
    info.topK = static_cast<uint32_t>(expertIdxTensor->GetStorageShape().GetDim(1));

    auto w1Tensor = context->GetDynamicInputTensor(WEIGHT1_INDEX, 0);
    auto w2Tensor = context->GetDynamicInputTensor(WEIGHT2_INDEX, 0);
    OP_TILING_CHECK(w1Tensor == nullptr || w2Tensor == nullptr,
        OP_LOGE(K_INNER_DEBUG, "residual weight lists must be non-empty."), return ge::GRAPH_FAILED);
    info.expertPerRank = static_cast<uint32_t>(w1Tensor->GetStorageShape().GetDim(0));

    OP_TILING_CHECK(CheckRank3Shape(context, GATE_UP_SVDQ_L1_INDEX, "gateUpSvdqL1", info.expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "gateUpSvdqL1 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, GATE_SVDQ_L2_INDEX, "gateSvdqL2", info.expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "gateSvdqL2 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, UP_SVDQ_L2_INDEX, "upSvdqL2", info.expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "upSvdqL2 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DOWN_SVDQ_L1_INDEX, "downSvdqL1", info.expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "downSvdqL1 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DOWN_SVDQ_L2_INDEX, "downSvdqL2", info.expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "downSvdqL2 shape check failed."), return ge::GRAPH_FAILED);

    const gert::StorageShape* gateUpL1Shape = context->GetInputShape(GATE_UP_SVDQ_L1_INDEX);
    const gert::StorageShape* gateL2Shape = context->GetInputShape(GATE_SVDQ_L2_INDEX);
    const gert::StorageShape* upL2Shape = context->GetInputShape(UP_SVDQ_L2_INDEX);
    const gert::StorageShape* downL1Shape = context->GetInputShape(DOWN_SVDQ_L1_INDEX);
    const gert::StorageShape* downL2Shape = context->GetInputShape(DOWN_SVDQ_L2_INDEX);
    OP_TILING_CHECK(gateUpL1Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.gateRank + info.upRank),
        OP_LOGE(K_INNER_DEBUG, "gateUpSvdqL1 rank dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gateUpL1Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.hiddenSize),
        OP_LOGE(K_INNER_DEBUG, "gateUpSvdqL1 hidden dim mismatch."), return ge::GRAPH_FAILED);
    info.intermediateSize = static_cast<uint32_t>(gateL2Shape->GetStorageShape().GetDim(1));
    OP_TILING_CHECK(gateL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.gateRank),
        OP_LOGE(K_INNER_DEBUG, "gateSvdqL2 rank dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(upL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.intermediateSize),
        OP_LOGE(K_INNER_DEBUG, "upSvdqL2 intermediate dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(upL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.upRank),
        OP_LOGE(K_INNER_DEBUG, "upSvdqL2 rank dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL1Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.downRank),
        OP_LOGE(K_INNER_DEBUG, "downSvdqL1 rank dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL1Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.intermediateSize),
        OP_LOGE(K_INNER_DEBUG, "downSvdqL1 intermediate dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.hiddenSize),
        OP_LOGE(K_INNER_DEBUG, "downSvdqL2 hidden dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(info.downRank),
        OP_LOGE(K_INNER_DEBUG, "downSvdqL2 rank dim mismatch."), return ge::GRAPH_FAILED);

    const gert::StorageShape* outShape = context->GetOutputShape(OUT_INDEX);
    OP_TILING_CHECK(outShape == nullptr, OP_LOGE(K_INNER_DEBUG, "out shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(K_INNER_DEBUG, "out must be rank-2."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.m),
        OP_LOGE(K_INNER_DEBUG, "out token dim mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(outShape->GetStorageShape().GetDim(1) != static_cast<int64_t>(info.hiddenSize),
        OP_LOGE(K_INNER_DEBUG, "out hidden dim mismatch."), return ge::GRAPH_FAILED);

    const gert::StorageShape* expertTokenNumsShape = context->GetOutputShape(EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsShape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "expertTokenNums shape is null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsShape->GetStorageShape().GetDimNum() != 1,
        OP_LOGE(K_INNER_DEBUG, "expertTokenNums must be rank-1."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expertTokenNumsShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.expertPerRank),
        OP_LOGE(K_INNER_DEBUG, "expertTokenNums expert dim mismatch."), return ge::GRAPH_FAILED);

    const gert::StorageShape* xActiveMaskShape = context->GetOptionalInputShape(X_ACTIVE_MASK_INDEX);
    if (xActiveMaskShape != nullptr) {
        OP_TILING_CHECK(xActiveMaskShape->GetStorageShape().GetDimNum() != 1,
            OP_LOGE(K_INNER_DEBUG, "xActiveMask must be rank-1."), return ge::GRAPH_FAILED);
        OP_TILING_CHECK(xActiveMaskShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(info.m),
            OP_LOGE(K_INNER_DEBUG, "xActiveMask token dim mismatch."), return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus DispatchFFNCombineW4A8SVDQTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    DispatchFFNCombineW4A8SVDQTilingData* tilingData =
        context->GetTilingData<DispatchFFNCombineW4A8SVDQTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "tilingData is nullptr."), return ge::GRAPH_FAILED);

    auto& info = tilingData->info;
    OP_TILING_CHECK(DispatchFFNCombineW4A8SVDQCheckAttrAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckAttrAndSetTiling failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineW4A8SVDQCheckDType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckDType failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineW4A8SVDQCheckShapeAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "CheckShapeAndSetTiling failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(DispatchFFNCombineW4A8SVDQGetPlatformInfoAndSetTiling(context, info) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "GetPlatformInfoAndSetTiling failed."), return ge::GRAPH_FAILED);

    BuildWorkspaceMap(tilingData);
    BuildSyncFlagTable(tilingData);
    BuildBF16StageShapeTable(tilingData);
    BuildResidualStageShapeTable(tilingData);
    BuildLowRankInvocationTable(tilingData);
    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SVDQ_SYSTEM_WORKSPACE + info.workspaceBytes;

    OP_LOGE(nodeName, "DispatchFFNCombineW4A8SVDQ AscendC kernel is not implemented yet.");
    return ge::GRAPH_FAILED;
}

struct DispatchFFNCombineW4A8SVDQCompileInfo {};

ge::graphStatus TilingParseForDispatchFFNCombineW4A8SVDQ(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(DispatchFFNCombineW4A8SVDQ)
    .Tiling(DispatchFFNCombineW4A8SVDQTilingFunc)
    .TilingParse<DispatchFFNCombineW4A8SVDQCompileInfo>(TilingParseForDispatchFFNCombineW4A8SVDQ);

static ge::graphStatus SVDQLowRankDebugReadbackCheckAttr(
    gert::TilingContext* context, SVDQLowRankDebugTilingData* tilingData)
{
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(K_INNER_DEBUG, "debug attrs is null."), return ge::GRAPH_FAILED);

    auto gateRank = attrs->GetAttrPointer<int>(0);
    auto upRank = attrs->GetAttrPointer<int>(1);
    auto downRank = attrs->GetAttrPointer<int>(2);
    auto gateRankOffset = attrs->GetAttrPointer<int>(3);
    auto upRankOffset = attrs->GetAttrPointer<int>(4);

    OP_TILING_CHECK(gateRank == nullptr || upRank == nullptr || downRank == nullptr,
        OP_LOGE(K_INNER_DEBUG, "debug rank attrs are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gateRankOffset == nullptr || upRankOffset == nullptr,
        OP_LOGE(K_INNER_DEBUG, "debug rank offset attrs are invalid."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*gateRank <= 0 || *upRank <= 0 || *downRank <= 0,
        OP_LOGE(K_INNER_DEBUG, "debug ranks must be positive."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*gateRankOffset != 0,
        OP_LOGE(K_INNER_DEBUG, "debug gateRankOffset must be 0."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(*upRankOffset != *gateRank,
        OP_LOGE(K_INNER_DEBUG, "debug upRankOffset must equal gateRank."), return ge::GRAPH_FAILED);

    tilingData->gateUpInvocation.rankColumns = static_cast<uint32_t>(*gateRank);
    tilingData->gateUpInvocation.secondRankColumns = static_cast<uint32_t>(*upRank);
    tilingData->downInvocation.rankColumns = static_cast<uint32_t>(*downRank);
    tilingData->gateUpInvocation.inputColumnOffset = static_cast<uint32_t>(*gateRankOffset);
    tilingData->gateUpInvocation.secondInputColumnOffset = static_cast<uint32_t>(*upRankOffset);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SVDQLowRankDebugReadbackCheckDType(gert::TilingContext* context)
{
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_ROUTED_X_INDEX, "debug routedX", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug routedX dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_HIDDEN_INDEX, "debug hidden", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug hidden dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_GATE_UP_SVDQ_L1_INDEX, "debug gateUpSvdqL1", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpSvdqL1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_GATE_SVDQ_L2_INDEX, "debug gateSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_UP_SVDQ_L2_INDEX, "debug upSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug upSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_DOWN_SVDQ_L1_INDEX, "debug downSvdqL1", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL1 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_DOWN_SVDQ_L2_INDEX, "debug downSvdqL2", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL2 dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRequiredInputDType(context, DEBUG_EXPERT_TOKEN_NUMS_INDEX, "debug expertTokenNums", ge::DT_INT32) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug expertTokenNums dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, DEBUG_GATE_UP_OUTPUT_INDEX, "debug gateUpOutput", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpOutput dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, DEBUG_DOWN_OUTPUT_INDEX, "debug downOutput", ge::DT_BF16) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downOutput dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, DEBUG_GATE_UP_ACCUMULATOR_INDEX, "debug gateUpAccumulator", ge::DT_FLOAT) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpAccumulator dtype check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckOutputDType(context, DEBUG_DOWN_ACCUMULATOR_INDEX, "debug downAccumulator", ge::DT_FLOAT) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downAccumulator dtype check failed."), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckDebugOutputRank2Shape(
    gert::TilingContext* context, uint32_t index, const char* name, int64_t expectedDim0, int64_t expectedDim1)
{
    const gert::StorageShape* shape = context->GetOutputShape(index);
    OP_TILING_CHECK(shape == nullptr, OP_LOGE(K_INNER_DEBUG, "%s shape is null.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(shape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(K_INNER_DEBUG, "%s must be rank-2.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expectedDim0 >= 0 && shape->GetStorageShape().GetDim(0) != expectedDim0,
        OP_LOGE(K_INNER_DEBUG, "%s dim0 mismatch.", name), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(expectedDim1 >= 0 && shape->GetStorageShape().GetDim(1) != expectedDim1,
        OP_LOGE(K_INNER_DEBUG, "%s dim1 mismatch.", name), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void SetDebugLowRankInvocation(DispatchFFNCombineW4A8SVDQImpl::SVDQFusedDownUpTiling& invocation,
    uint32_t invocationId, uint32_t downFactorId, uint32_t upFactorId, uint32_t secondUpFactorId,
    uint32_t m, uint32_t inputColumns, uint32_t rankColumns, uint32_t secondRankColumns, uint32_t outputColumns,
    uint32_t inputColumnOffset, uint32_t outputColumnOffset, uint32_t secondInputColumnOffset,
    uint32_t secondOutputColumnOffset, uint32_t coreCount, uint32_t accumulatorRegionId)
{
    invocation.invocationId = invocationId;
    invocation.inputRegionId = SVDQ_INVALID_ID;
    invocation.outputRegionId = SVDQ_INVALID_ID;
    invocation.downFactorId = downFactorId;
    invocation.upFactorId = upFactorId;
    invocation.secondUpFactorId = secondUpFactorId;
    invocation.m = m;
    invocation.inputColumns = inputColumns;
    invocation.rankColumns = rankColumns;
    invocation.secondRankColumns = secondRankColumns;
    invocation.outputColumns = outputColumns;
    invocation.inputColumnOffset = inputColumnOffset;
    invocation.outputColumnOffset = outputColumnOffset;
    invocation.secondInputColumnOffset = secondInputColumnOffset;
    invocation.secondOutputColumnOffset = secondOutputColumnOffset;
    invocation.rowTile = SVDQ_LOWRANK_ROW_TILE;
    invocation.outputColumnTile = SVDQ_LOWRANK_OUTPUT_COLUMN_TILE;
    invocation.kTile = SVDQ_LOWRANK_K_TILE;
    invocation.coreCount = coreCount;
    invocation.accumulatorRegionId = accumulatorRegionId;
}

static ge::graphStatus SVDQLowRankDebugReadbackCheckShapeAndSetTiling(
    gert::TilingContext* context, SVDQLowRankDebugTilingData* tilingData)
{
    const gert::StorageShape* routedXShape = context->GetInputShape(DEBUG_ROUTED_X_INDEX);
    const gert::StorageShape* hiddenShape = context->GetInputShape(DEBUG_HIDDEN_INDEX);
    const gert::StorageShape* gateUpL1Shape = context->GetInputShape(DEBUG_GATE_UP_SVDQ_L1_INDEX);
    const gert::StorageShape* gateL2Shape = context->GetInputShape(DEBUG_GATE_SVDQ_L2_INDEX);
    const gert::StorageShape* upL2Shape = context->GetInputShape(DEBUG_UP_SVDQ_L2_INDEX);
    const gert::StorageShape* downL1Shape = context->GetInputShape(DEBUG_DOWN_SVDQ_L1_INDEX);
    const gert::StorageShape* downL2Shape = context->GetInputShape(DEBUG_DOWN_SVDQ_L2_INDEX);

    OP_TILING_CHECK(routedXShape == nullptr || hiddenShape == nullptr || gateUpL1Shape == nullptr ||
        gateL2Shape == nullptr || upL2Shape == nullptr || downL1Shape == nullptr || downL2Shape == nullptr,
        OP_LOGE(K_INNER_DEBUG, "debug shapes must be non-null."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(routedXShape->GetStorageShape().GetDimNum() != 2 ||
        hiddenShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(K_INNER_DEBUG, "debug routedX and hidden must be rank-2."), return ge::GRAPH_FAILED);

    const int64_t routedRowsDim = routedXShape->GetStorageShape().GetDim(0);
    const int64_t hiddenSizeDim = routedXShape->GetStorageShape().GetDim(1);
    const int64_t hiddenRowsDim = hiddenShape->GetStorageShape().GetDim(0);
    const int64_t intermediateSizeDim = hiddenShape->GetStorageShape().GetDim(1);
    OP_TILING_CHECK(routedRowsDim <= 0 || hiddenSizeDim <= 0 || hiddenRowsDim <= 0 || intermediateSizeDim <= 0,
        OP_LOGE(K_INNER_DEBUG, "debug routed shape is invalid."), return ge::GRAPH_FAILED);

    const uint32_t routedRows = static_cast<uint32_t>(routedRowsDim);
    const uint32_t hiddenSize = static_cast<uint32_t>(hiddenSizeDim);
    const uint32_t intermediateSize = static_cast<uint32_t>(intermediateSizeDim);
    const uint32_t gateRank = tilingData->gateUpInvocation.rankColumns;
    const uint32_t upRank = tilingData->gateUpInvocation.secondRankColumns;
    const uint32_t downRank = tilingData->downInvocation.rankColumns;

    OP_TILING_CHECK(hiddenRowsDim != static_cast<int64_t>(routedRows),
        OP_LOGE(K_INNER_DEBUG, "debug hidden row dim mismatch."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(CheckRank3Shape(context, DEBUG_GATE_UP_SVDQ_L1_INDEX, "debug gateUpSvdqL1", -1) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpSvdqL1 shape check failed."), return ge::GRAPH_FAILED);
    tilingData->expertPerRank = static_cast<uint32_t>(gateUpL1Shape->GetStorageShape().GetDim(0));
    OP_TILING_CHECK(tilingData->expertPerRank == 0,
        OP_LOGE(K_INNER_DEBUG, "debug expertPerRank must be positive."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DEBUG_GATE_SVDQ_L2_INDEX, "debug gateSvdqL2", tilingData->expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateSvdqL2 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DEBUG_UP_SVDQ_L2_INDEX, "debug upSvdqL2", tilingData->expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug upSvdqL2 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DEBUG_DOWN_SVDQ_L1_INDEX, "debug downSvdqL1", tilingData->expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL1 shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckRank3Shape(context, DEBUG_DOWN_SVDQ_L2_INDEX, "debug downSvdqL2", tilingData->expertPerRank) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL2 shape check failed."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(gateUpL1Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(gateRank + upRank) ||
        gateUpL1Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(hiddenSize),
        OP_LOGE(K_INNER_DEBUG, "debug gateUpSvdqL1 shape mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(gateL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(intermediateSize) ||
        gateL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(gateRank),
        OP_LOGE(K_INNER_DEBUG, "debug gateSvdqL2 shape mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(upL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(intermediateSize) ||
        upL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(upRank),
        OP_LOGE(K_INNER_DEBUG, "debug upSvdqL2 shape mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL1Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(downRank) ||
        downL1Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(intermediateSize),
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL1 shape mismatch."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(downL2Shape->GetStorageShape().GetDim(1) != static_cast<int64_t>(hiddenSize) ||
        downL2Shape->GetStorageShape().GetDim(2) != static_cast<int64_t>(downRank),
        OP_LOGE(K_INNER_DEBUG, "debug downSvdqL2 shape mismatch."), return ge::GRAPH_FAILED);

    const gert::StorageShape* expertTokenNumsShape = context->GetInputShape(DEBUG_EXPERT_TOKEN_NUMS_INDEX);
    OP_TILING_CHECK(expertTokenNumsShape == nullptr || expertTokenNumsShape->GetStorageShape().GetDimNum() != 1 ||
        expertTokenNumsShape->GetStorageShape().GetDim(0) != static_cast<int64_t>(tilingData->expertPerRank),
        OP_LOGE(K_INNER_DEBUG, "debug expertTokenNums shape mismatch."), return ge::GRAPH_FAILED);

    const uint32_t gateUpOutputColumns = intermediateSize * 2;
    OP_TILING_CHECK(CheckDebugOutputRank2Shape(context, DEBUG_GATE_UP_OUTPUT_INDEX, "debug gateUpOutput",
        routedRows, gateUpOutputColumns) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpOutput shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDebugOutputRank2Shape(context, DEBUG_DOWN_OUTPUT_INDEX, "debug downOutput",
        routedRows, hiddenSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downOutput shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDebugOutputRank2Shape(context, DEBUG_GATE_UP_ACCUMULATOR_INDEX, "debug gateUpAccumulator",
        routedRows, gateUpOutputColumns) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug gateUpAccumulator shape check failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(CheckDebugOutputRank2Shape(context, DEBUG_DOWN_ACCUMULATOR_INDEX, "debug downAccumulator",
        routedRows, hiddenSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(K_INNER_DEBUG, "debug downAccumulator shape check failed."), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t aicNum = ascendcPlatform.GetCoreNumAic();
    const uint32_t aivNum = ascendcPlatform.GetCoreNumAiv();
    const uint32_t blockDim = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);
    context->SetBlockDim(blockDim);
    context->SetTilingKey(0);

    SetDebugLowRankInvocation(tilingData->gateUpInvocation, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_GATE_UP,
        SVDQ_FACTOR_GATE_UP_L1, SVDQ_FACTOR_GATE_L2, SVDQ_FACTOR_UP_L2, routedRows, hiddenSize, gateRank, upRank,
        gateUpOutputColumns, tilingData->gateUpInvocation.inputColumnOffset, 0,
        tilingData->gateUpInvocation.secondInputColumnOffset, intermediateSize, blockDim,
        SVDQ_REGION_LOWRANK_ACCUMULATOR_1);
    SetDebugLowRankInvocation(tilingData->downInvocation, DispatchFFNCombineW4A8SVDQImpl::SVDQ_LOWRANK_INVOCATION_DOWN,
        SVDQ_FACTOR_DOWN_L1, SVDQ_FACTOR_DOWN_L2, SVDQ_INVALID_ID, routedRows, intermediateSize, downRank, 0,
        hiddenSize, 0, 0, 0, 0, blockDim, SVDQ_REGION_LOWRANK_ACCUMULATOR_2);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SVDQLowRankDebugReadbackTilingFunc(gert::TilingContext* context)
{
    const char* nodeName = context->GetNodeName();
    SVDQLowRankDebugTilingData* tilingData = context->GetTilingData<SVDQLowRankDebugTilingData>();
    OP_TILING_CHECK(tilingData == nullptr,
        OP_LOGE(nodeName, "debug tilingData is nullptr."), return ge::GRAPH_FAILED);

    OP_TILING_CHECK(SVDQLowRankDebugReadbackCheckAttr(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "debug CheckAttr failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SVDQLowRankDebugReadbackCheckDType(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "debug CheckDType failed."), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(SVDQLowRankDebugReadbackCheckShapeAndSetTiling(context, tilingData) != ge::GRAPH_SUCCESS,
        OP_LOGE(nodeName, "debug CheckShapeAndSetTiling failed."), return ge::GRAPH_FAILED);

    size_t* workSpaces = context->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workSpaces == nullptr,
        OP_LOGE(nodeName, "debug workSpaces is nullptr."), return ge::GRAPH_FAILED);
    workSpaces[0] = SVDQ_SYSTEM_WORKSPACE;
    return ge::GRAPH_SUCCESS;
}

struct SVDQLowRankDebugReadbackCompileInfo {};

ge::graphStatus TilingParseForSVDQLowRankDebugReadback(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SVDQLowRankDebugReadback)
    .Tiling(SVDQLowRankDebugReadbackTilingFunc)
    .TilingParse<SVDQLowRankDebugReadbackCompileInfo>(TilingParseForSVDQLowRankDebugReadback);
}  // namespace optiling
