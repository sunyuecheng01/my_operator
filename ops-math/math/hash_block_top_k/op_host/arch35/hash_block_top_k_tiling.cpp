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
 * \file hash_block_top_k_tiling.cpp
 * \brief HashBlockTopK tiling.
 */
#include <algorithm>
#include "hash_block_top_k_tiling.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace optiling {
namespace {
constexpr size_t HASH_Q_IDX = 0;
constexpr size_t HASH_K_CACHE_IDX = 1;
constexpr size_t K_IDX = 2;
constexpr size_t HASH_K_BLOCK_TABLE_IDX = 3;
constexpr size_t SEQ_LEN_IDX = 4;
constexpr size_t NEW_BLOCK_TABLE_IDX = 0;

constexpr size_t Q_DIM_BATCH = 0;
constexpr size_t Q_DIM_SEQ = 1;
constexpr size_t Q_DIM_HEAD = 2;
constexpr size_t Q_DIM_BYTES = 3;
constexpr size_t K_DIM_PHYSICAL_BLOCK = 0;
constexpr size_t K_DIM_BATCH = 1;
constexpr size_t K_DIM_BLOCK_SIZE = 2;
constexpr size_t K_DIM_HEAD = 3;
constexpr size_t K_DIM_BYTES = 4;
constexpr size_t TABLE_DIM_BATCH = 0;
constexpr size_t TABLE_DIM_BLOCK = 1;
constexpr int64_t TILE_N2_BASIC = 1024;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr size_t WORKSPACE_ALIGN = 512;
constexpr int64_t CUBE_BASE_M = 16;
constexpr int64_t CUBE_BASE_N = 256;
constexpr int64_t CUBE_BASE_K = 128;

inline size_t AlignUp(size_t x, size_t align)
{
    return align == 0 ? x : (x + align - 1) / align * align;
}

bool IsScalarOrBatchVector(gert::TilingContext* context, size_t index, const char* inputName, int64_t batch,
    int64_t& isScalar)
{
    auto desc = context->GetInputDesc(index);
    auto shape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, desc);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    OP_CHECK_IF(desc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context->GetNodeName(), "%s must be DT_INT32.", inputName), return false);

    const auto& storageShape = shape->GetStorageShape();
    if (storageShape.GetDimNum() == 0 || (storageShape.GetDimNum() == 1 && storageShape.GetDim(0) == 1)) {
        isScalar = 1;
        return true;
    }
    OP_CHECK_IF(storageShape.GetDimNum() != 1 || storageShape.GetDim(0) != batch,
        OP_LOGE(context->GetNodeName(), "%s shape must be scalar or [batch].", inputName), return false);
    isScalar = 0;
    return true;
}

bool SetCubeMatmulTiling(gert::TilingContext* context, HashBlockTopKTilingData& tiling,
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
        OP_LOGE(context->GetNodeName(), "HashBlockTopK cube matmul tiling failed."), return false);
    return true;
}
} // namespace

static ge::graphStatus TilingPrepareForHashBlockTopK(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<HashBlockTopKCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingForHashBlockTopK(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("HashBlockTopK", "Tiling context is nullptr.");
        return ge::GRAPH_FAILED;
    }

    auto qDesc = context->GetInputDesc(HASH_Q_IDX);
    auto kDesc = context->GetInputDesc(HASH_K_CACHE_IDX);
    auto tableDesc = context->GetInputDesc(HASH_K_BLOCK_TABLE_IDX);
    auto outDesc = context->GetOutputDesc(NEW_BLOCK_TABLE_IDX);
    auto qShape = context->GetInputShape(HASH_Q_IDX);
    auto kShape = context->GetInputShape(HASH_K_CACHE_IDX);
    auto tableShape = context->GetInputShape(HASH_K_BLOCK_TABLE_IDX);
    auto outShape = context->GetOutputShape(NEW_BLOCK_TABLE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, kDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    OP_CHECK_IF(qDesc->GetDataType() != ge::DT_UINT8 || kDesc->GetDataType() != ge::DT_UINT8,
        OP_LOGE(context->GetNodeName(), "hash_q and hash_k_cache must be DT_UINT8."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(tableDesc->GetDataType() != ge::DT_INT32 || outDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE(context->GetNodeName(), "hash_k_block_table and new_block_table must be DT_INT32."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(qShape->GetStorageShape().GetDimNum() != 4 || kShape->GetStorageShape().GetDimNum() != 5 ||
                    tableShape->GetStorageShape().GetDimNum() != 2,
        OP_LOGE(context->GetNodeName(),
            "hash_q must be [B,Sq,QH,D/8], hash_k_cache must be [PB,B,BS,H,D/8], table must be [B,BN]."),
        return ge::GRAPH_FAILED);

    const int64_t batch = qShape->GetStorageShape().GetDim(Q_DIM_BATCH);
    const int64_t qSeqLen = qShape->GetStorageShape().GetDim(Q_DIM_SEQ);
    const int64_t qHead = qShape->GetStorageShape().GetDim(Q_DIM_HEAD);
    const int64_t dimBytes = qShape->GetStorageShape().GetDim(Q_DIM_BYTES);
    const int64_t physicalBlockCount = kShape->GetStorageShape().GetDim(K_DIM_PHYSICAL_BLOCK);
    const int64_t keyBatch = kShape->GetStorageShape().GetDim(K_DIM_BATCH);
    const int64_t blockSize = kShape->GetStorageShape().GetDim(K_DIM_BLOCK_SIZE);
    const int64_t head = kShape->GetStorageShape().GetDim(K_DIM_HEAD);
    const int64_t keyDimBytes = kShape->GetStorageShape().GetDim(K_DIM_BYTES);
    const int64_t tableBatch = tableShape->GetStorageShape().GetDim(TABLE_DIM_BATCH);
    const int64_t blockCount = tableShape->GetStorageShape().GetDim(TABLE_DIM_BLOCK);
    OP_CHECK_IF(batch <= 0 || qSeqLen <= 0 || qHead <= 0 || dimBytes <= 0 || physicalBlockCount <= 0 ||
                    keyBatch <= 0 || blockSize <= 0 || head <= 0 || blockCount <= 0,
        OP_LOGE(context->GetNodeName(), "all static shape dimensions must be positive."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyBatch != batch || tableBatch != batch,
        OP_LOGE(context->GetNodeName(), "batch dimensions of hash_q/hash_k_cache/hash_k_block_table must match."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyDimBytes != dimBytes,
        OP_LOGE(context->GetNodeName(), "hash_q and hash_k_cache last dim must be the same."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(outShape->GetStorageShape().GetDimNum() != 2 ||
                    outShape->GetStorageShape().GetDim(0) != batch ||
                    outShape->GetStorageShape().GetDim(1) != blockCount,
        OP_LOGE(context->GetNodeName(), "new_block_table shape must be [batch, block_num]."),
        return ge::GRAPH_FAILED);

    const int64_t expandedDim = dimBytes * 8;
    OP_CHECK_IF(expandedDim % 32 != 0,
        OP_LOGE(context->GetNodeName(), "expanded bit dimension must be aligned to 32 for int8 cube matmul."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!((qHead >= head && qHead % head == 0) || (head > qHead && head % qHead == 0)),
        OP_LOGE(context->GetNodeName(), "qhead_num and head_num must be divisible for MHA/GQA/MQA mapping."),
        return ge::GRAPH_FAILED);

    int64_t kIsScalar = 0;
    int64_t seqLenIsScalar = 0;
    if (!IsScalarOrBatchVector(context, K_IDX, "k", batch, kIsScalar) ||
        !IsScalarOrBatchVector(context, SEQ_LEN_IDX, "seqlen", batch, seqLenIsScalar)) {
        return ge::GRAPH_FAILED;
    }

    const int64_t pairCount = std::max(qHead, head);
    const int64_t tileN2 = std::min<int64_t>(blockSize, TILE_N2_BASIC);
    const auto compileInfo = reinterpret_cast<const HashBlockTopKCompileInfo*>(context->GetCompileInfo());
    const int64_t coreNum = (compileInfo != nullptr && compileInfo->coreNum > 0) ? compileInfo->coreNum : 1;
    const int64_t totalTasks = batch * blockCount;
    const int64_t usedCoreNum = std::max<int64_t>(1, std::min<int64_t>(totalTasks, coreNum));

    HashBlockTopKTilingData tiling;
    tiling.set_batch(batch);
    tiling.set_qSeqLen(qSeqLen);
    tiling.set_qHead(qHead);
    tiling.set_head(head);
    tiling.set_physicalBlockCount(physicalBlockCount);
    tiling.set_blockCount(blockCount);
    tiling.set_blockSize(blockSize);
    tiling.set_dimBytes(dimBytes);
    tiling.set_pairCount(pairCount);
    tiling.set_totalTasks(totalTasks);
    tiling.set_tileN2(tileN2);
    tiling.set_expandedDim(expandedDim);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_kIsScalar(kIsScalar);
    tiling.set_seqLenIsScalar(seqLenIsScalar);
    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(1);
    if (!SetCubeMatmulTiling(context, tiling, tileN2, expandedDim)) {
        return ge::GRAPH_FAILED;
    }

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    size_t userWorkspaceSize = 0;
    userWorkspaceSize += static_cast<size_t>(batch * blockCount * sizeof(int32_t)); // block scores
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * expandedDim * sizeof(int8_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * tileN2 * expandedDim * sizeof(int8_t));
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * tileN2 * sizeof(int32_t)); // matmul output
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    userWorkspaceSize += static_cast<size_t>(usedCoreNum * tileN2 * sizeof(int32_t)); // token aggregate
    userWorkspaceSize = AlignUp(userWorkspaceSize, WORKSPACE_ALIGN);
    workspaces[0] = SYS_WORKSPACE_SIZE + userWorkspaceSize;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(HashBlockTopK)
    .Tiling(TilingForHashBlockTopK)
    .TilingParse<HashBlockTopKCompileInfo>(TilingPrepareForHashBlockTopK);
} // namespace optiling
