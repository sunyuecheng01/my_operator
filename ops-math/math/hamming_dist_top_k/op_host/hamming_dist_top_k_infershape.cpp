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
 * \file hamming_dist_top_k_infershape.cpp
 * \brief HammingDistTopK shape inference.
 */
#include <algorithm>
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t QUERY_IDX = 0;
constexpr size_t KEY_IDX = 1;
constexpr size_t KEY_BLOCK_TABLE_IDX = 5;
constexpr size_t INDICES_IDX = 0;
constexpr size_t DIM_BATCH = 0;
constexpr size_t DIM_QHEAD = 1;
constexpr size_t DIM_HEAD = 1;
constexpr size_t DIM_SEQ = 2;
constexpr int64_t DEFAULT_TOPK = 128;
constexpr int64_t MIN_OUTPUT_CHUNK_LEN = 128;

inline int64_t CeilDiv(int64_t x, int64_t y)
{
    return y == 0 ? x : (x + y - 1) / y;
}

int64_t GetTopKAttr(gert::InferShapeContext* context)
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
    if (maxSeqLen < 0) {
        return -1;
    }
    const int64_t maxChunkSize = maxSeqLen >= 1024 ? 16 : (maxSeqLen >= 128 ? 8 : 1);
    const int64_t baseLen = std::max(CeilDiv(maxSeqLen, maxChunkSize), MIN_OUTPUT_CHUNK_LEN);
    const int64_t formulaLen = baseLen * topk / 100 + 2;
    return std::max({formulaLen, MIN_OUTPUT_CHUNK_LEN, topk});
}
} // namespace

static ge::graphStatus InferShapeForHammingDistTopK(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to infer HammingDistTopK shape.");
    const gert::Shape* queryShape = context->GetInputShape(QUERY_IDX);
    const gert::Shape* keyShape = context->GetInputShape(KEY_IDX);
    gert::Shape* outShape = context->GetOutputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (queryShape->GetDimNum() != 4 || keyShape->GetDimNum() != 4) {
        OP_LOGE(context->GetNodeName(), "query and key_compressed must be 4-D tensors.");
        return GRAPH_FAILED;
    }

    const gert::Shape* blockTableShape = context->GetOptionalInputShape(KEY_BLOCK_TABLE_IDX);
    const bool hasBlockTable = blockTableShape != nullptr && blockTableShape->GetDimNum() >= 2;

    const int64_t batch = queryShape->GetDim(DIM_BATCH);
    const int64_t head = keyShape->GetDim(DIM_HEAD);
    int64_t maxSeqLen = keyShape->GetDim(DIM_SEQ);
    if (hasBlockTable) {
        const int64_t blockCount = blockTableShape->GetDim(1);
        const int64_t blockSize = keyShape->GetDim(DIM_SEQ);
        maxSeqLen = (blockCount >= 0 && blockSize >= 0) ? blockCount * blockSize : -1;
    }

    outShape->SetDimNum(3);
    outShape->SetDim(0, batch);
    outShape->SetDim(1, head);
    outShape->SetDim(2, CalcOutputChunkLen(maxSeqLen, GetTopKAttr(context)));
    OP_LOGD(context->GetNodeName(), "End to infer HammingDistTopK shape.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HammingDistTopK).InferShape(InferShapeForHammingDistTopK);
} // namespace ops
