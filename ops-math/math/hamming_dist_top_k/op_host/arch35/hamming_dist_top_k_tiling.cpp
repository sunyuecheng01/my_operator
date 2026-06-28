/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hamming_dist_top_k_tiling.cpp
 * \brief HammingDistTopK tiling.
 */
#include <algorithm>
#include "hamming_dist_top_k_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr size_t QUERY_IDX = 0;
constexpr size_t KEY_IDX = 1;
constexpr size_t K_IDX = 2;
constexpr size_t SEQ_LEN_IDX = 3;
constexpr size_t CHUNK_SIZE_IDX = 4;
constexpr size_t KEY_BLOCK_TABLE_IDX = 5;
constexpr size_t INDICES_IDX = 0;
constexpr size_t DIM_BATCH = 0;
constexpr size_t DIM_QHEAD = 1;
constexpr size_t DIM_HEAD = 1;
constexpr size_t DIM_SEQ = 2;
constexpr size_t DIM_BYTES = 3;
constexpr int64_t DEFAULT_TOPK = 128;
constexpr int64_t MIN_OUTPUT_CHUNK_LEN = 128;
constexpr int64_t TILE_N1_BASIC = 254;
constexpr int64_t TILE_N2_BASIC = 3328;
constexpr int64_t TILE_N1_SPLIT_S = 128;
constexpr int64_t TILE_N2_SPLIT_S = 4 * 1024;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr size_t WORKSPACE_ALIGN = 512;
constexpr int64_t CUBE_BASE_M = 16;
constexpr int64_t CUBE_BASE_N = 256;
constexpr int64_t CUBE_BASE_K = 128;

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

inline size_t AlignUp(size_t x, size_t align)
{
    return align == 0 ? x : (x + align - 1) / align * align;
}

int64_t GetTopKAttr(gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return DEFAULT_TOPK;
    }
    const int64_t* topk = attrs->GetInt(0);
    if (topk == nullptr || *topk <= 0) {
        return DEFAULT_TOPK;
    }
    return *topk;
}

int64_t CalcOutputChunkLen(int64_t maxSeqLen, int64_t topk)
{
    const int64_t maxChunkSize = maxSeqLen >= 1024 ? 16 : (maxSeqLen >= 128 ? 8 : 1);
    const int64_t baseLen = std::max(CeilDiv(maxSeqLen, maxChunkSize), MIN_OUTPUT_CHUNK_LEN);
    const int64_t formulaLen = baseLen * topk / 100 + 2;
    return std::max({formulaLen, MIN_OUTPUT_CHUNK_LEN, topk});
}

bool CheckInt32VectorInput(gert::TilingContext* context, size_t index, const char* inputName, int64_t batch)
{
    auto desc = context->GetInputDesc(index);
    auto shape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context->GetNodeName(), "%s must be DT_INT32.", inputName), return false);
    OP_CHECK_IF(shape->GetStorageShape().GetDimNum() != 1 || shape->GetStorageShape().GetDim(0) != batch,
        OP_LOGE(context->GetNodeName(), "%s shape must be [batch].", inputName), return false);
    return true;
}

bool SetCubeMatmulTiling(gert::TilingContext* context, HammingDistTopKTilingData& tiling,
    int64_t tileN2, int64_t expandedDim)
{
    matmul_tiling::MatmulApiTiling mmTiling;
    mmTiling.SetAType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mmTiling.SetBType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT8, false);
    mmTiling.SetCType(
        matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    mmTiling.SetBias(false);
    mmTiling.SetOrgShape(1, tileN2, expandedDim);
    mmTiling.SetShape(1, tileN2, expandedDim);
    mmTiling.SetFixSplit(CUBE_BASE_M, std::min<int64_t>(tileN2, CUBE_BASE_N),
        std::min<int64_t>(expandedDim, CUBE_BASE_K));
    mmTiling.SetBufferSpace(-1, -1, -1);
    OP_CHECK_IF(mmTiling.GetTiling(tiling.mmTiling) == -1,
        OP_LOGE(context->GetNodeName(), "HammingDistTopK cube matmul tiling failed."), return false);
    return true;
}
} // namespace

static ge::graphStatus TilingPrepareForHammingDistTopK(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<HammingDistTopKCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForHammingDistTopK(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("HammingDistTopK", "Tiling context is nullptr.");
        return ge::GRAPH_FAILED;
    }
    auto queryDesc = context->GetInputDesc(QUERY_IDX);
    auto keyDesc = context->GetInputDesc(KEY_IDX);
    auto outDesc = context->GetOutputDesc(INDICES_IDX);
    auto queryShape = context->GetInputShape(QUERY_IDX);
    auto keyShape = context->GetInputShape(KEY_IDX);
    auto outShape = context->GetOutputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_CHECK_IF(queryDesc->GetDataType() != ge::DT_UINT8 || keyDesc->GetDataType() != ge::DT_UINT8,
        OP_LOGE(context->GetNodeName(), "query and key_compressed must be DT_UINT8."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(outDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context->GetNodeName(), "indices must be DT_INT32."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDimNum() != 4 || keyShape->GetStorageShape().GetDimNum() != 4,
        OP_LOGE(context->GetNodeName(), "query and key_compressed must be 4-D tensors."),
        return ge::GRAPH_FAILED);

    const int64_t batch = queryShape->GetStorageShape().GetDim(DIM_BATCH);
    const int64_t qHead = queryShape->GetStorageShape().GetDim(DIM_QHEAD);
    const int64_t head = keyShape->GetStorageShape().GetDim(DIM_HEAD);
    const int64_t dimBytes = queryShape->GetStorageShape().GetDim(DIM_BYTES);
    OP_CHECK_IF(batch <= 0 || qHead <= 0 || head <= 0 || dimBytes <= 0,
        OP_LOGE(context->GetNodeName(), "query/key shapes must be static positive values."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_BYTES) != dimBytes,
        OP_LOGE(context->GetNodeName(), "query and key_compressed last dim must be the same."),
        return ge::GRAPH_FAILED);
    const int64_t expandedDim = dimBytes * 8;
    OP_CHECK_IF(expandedDim % 32 != 0,
        OP_LOGE(context->GetNodeName(), "expanded bit dimension must be aligned to 32 for int8 cube matmul."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(queryShape->GetStorageShape().GetDim(DIM_SEQ) != 1,
        OP_LOGE(context->GetNodeName(), "query dim2 must be 1."), return ge::GRAPH_FAILED);

    if (!CheckInt32VectorInput(context, K_IDX, "k", batch) ||
        !CheckInt32VectorInput(context, SEQ_LEN_IDX, "seq_len", batch)) {
        return ge::GRAPH_FAILED;
    }

    int64_t hasChunkSize = 0;
    auto chunkShape = context->GetOptionalInputShape(CHUNK_SIZE_IDX);
    auto chunkDesc = context->GetOptionalInputDesc(CHUNK_SIZE_IDX);
    if (chunkShape != nullptr && chunkDesc != nullptr) {
        OP_CHECK_IF(chunkDesc->GetDataType() != ge::DT_INT32,
            OP_LOGE(context->GetNodeName(), "chunk_size must be DT_INT32."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(chunkShape->GetStorageShape().GetDimNum() != 1 || chunkShape->GetStorageShape().GetDim(0) != batch,
            OP_LOGE(context->GetNodeName(), "chunk_size shape must be [batch]."), return ge::GRAPH_FAILED);
        hasChunkSize = 1;
    }

    int64_t blockCount = 0;
    int64_t blockSize = keyShape->GetStorageShape().GetDim(DIM_SEQ);
    int64_t maxSeqLen = blockSize;
    int64_t hasKeyBlockTable = 0;
    auto blockTableShape = context->GetOptionalInputShape(KEY_BLOCK_TABLE_IDX);
    auto blockTableDesc = context->GetOptionalInputDesc(KEY_BLOCK_TABLE_IDX);
    if (blockTableShape != nullptr && blockTableDesc != nullptr) {
        OP_CHECK_IF(blockTableDesc->GetDataType() != ge::DT_INT32,
            OP_LOGE(context->GetNodeName(), "key_block_table must be DT_INT32."), return ge::GRAPH_FAILED);
        OP_CHECK_IF(blockTableShape->GetStorageShape().GetDimNum() != 2 ||
                    blockTableShape->GetStorageShape().GetDim(0) != batch,
            OP_LOGE(context->GetNodeName(), "key_block_table shape must be [batch, blockCount]."),
            return ge::GRAPH_FAILED);
        blockCount = blockTableShape->GetStorageShape().GetDim(1);
        OP_CHECK_IF(blockCount <= 0 || blockSize <= 0,
            OP_LOGE(context->GetNodeName(), "blockCount and blockSize must be positive when key_block_table is set."),
            return ge::GRAPH_FAILED);
        maxSeqLen = blockCount * blockSize;
        hasKeyBlockTable = 1;
    } else {
        blockCount = 0;
        blockSize = maxSeqLen;
        OP_CHECK_IF(keyShape->GetStorageShape().GetDim(DIM_BATCH) != batch,
            OP_LOGE(context->GetNodeName(), "key_compressed dim0 must equal batch when key_block_table is absent."),
            return ge::GRAPH_FAILED);
    }

    const int64_t outputChunkLenFromShape = outShape->GetStorageShape().GetDimNum() >= 3 ?
        outShape->GetStorageShape().GetDim(2) : 0;
    const int64_t topkAttr = GetTopKAttr(context);
    const int64_t outputChunkLen = outputChunkLenFromShape > 0 ?
        outputChunkLenFromShape : CalcOutputChunkLen(maxSeqLen, topkAttr);
    OP_CHECK_IF(outputChunkLen <= 0,
        OP_LOGE(context->GetNodeName(), "outputChunkLen must be positive."), return ge::GRAPH_FAILED);

    HammingDistTopKTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_qHead(qHead);
    tiling.set_head(head);
    tiling.set_maxSeqLen(maxSeqLen);
    tiling.set_dimBytes(dimBytes);
    tiling.set_outputChunkLen(outputChunkLen);
    tiling.set_blockCount(blockCount);
    tiling.set_blockSize(blockSize);
    tiling.set_totalPairs(batch * head);
    tiling.set_topkAttr(topkAttr);
    tiling.set_hasChunkSize(hasChunkSize);
    tiling.set_hasKeyBlockTable(hasKeyBlockTable);

    int64_t tileN2 = TILE_N2_BASIC;
    if (hasKeyBlockTable != 0) {
        tiling.set_tileN1(blockSize);
        tileN2 = TILE_N2_BASIC;
    } else if (maxSeqLen > 26 * 1024 || (batch < 16 && maxSeqLen > 8 * 1024)) {
        tiling.set_tileN1(TILE_N1_SPLIT_S);
        tileN2 = TILE_N2_SPLIT_S;
    } else {
        tiling.set_tileN1(TILE_N1_BASIC);
        tileN2 = TILE_N2_BASIC;
    }
    tiling.set_tileN2(tileN2);
    tiling.set_expandedDim(expandedDim);

    const auto compileInfo = reinterpret_cast<const HammingDistTopKCompileInfo*>(context->GetCompileInfo());
    const int64_t coreNum = (compileInfo != nullptr && compileInfo->coreNum > 0) ? compileInfo->coreNum : 1;
    const int64_t usedCoreNum = std::max<int64_t>(1, std::min<int64_t>(batch * head, coreNum));
    tiling.set_usedCoreNum(usedCoreNum);
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    if (!SetCubeMatmulTiling(context, tiling, tileN2, expandedDim)) {
        return ge::GRAPH_FAILED;
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    size_t userWorkspaceSize = 0;
    // score buffer is per-batch now (final TopK is head-agnostic).
    userWorkspaceSize += static_cast<size_t>(batch * outputChunkLen * sizeof(int32_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * expandedDim * sizeof(int8_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * tileN2 * expandedDim * sizeof(int8_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * tileN2 * sizeof(int32_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    // per-chunk best similarity is persisted per (batch,head) pair so all heads can
    // be summed per batch in phase 2; sized batch*head*maxSeqLen (chunkSize>=1).
    userWorkspaceSize += static_cast<size_t>(batch * head * maxSeqLen * sizeof(int32_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    workspaces[0] = SYS_WORKSPACE_SIZE + userWorkspaceSize;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HammingDistTopK)
    .Tiling(TilingForHammingDistTopK)
    .TilingParse<HammingDistTopKCompileInfo>(TilingPrepareForHammingDistTopK);
} // namespace optiling
