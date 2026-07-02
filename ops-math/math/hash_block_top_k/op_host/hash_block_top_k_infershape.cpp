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
 * \file hash_block_top_k_infershape.cpp
 * \brief HashBlockTopK shape inference.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr size_t HASH_Q_IDX = 0;
constexpr size_t HASH_K_CACHE_IDX = 1;
constexpr size_t HASH_K_BLOCK_TABLE_IDX = 3;
constexpr size_t NEW_BLOCK_TABLE_IDX = 0;
constexpr size_t Q_DIM_BATCH = 0;
constexpr size_t K_DIM_BATCH = 1;
constexpr size_t K_DIM_BYTES = 4;
constexpr size_t TABLE_DIM_BATCH = 0;
constexpr size_t TABLE_DIM_BLOCK = 1;
constexpr size_t Q_DIM_BYTES = 3;
} // namespace

static ge::graphStatus InferShapeForHashBlockTopK(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to infer HashBlockTopK shape.");
    const gert::Shape* qShape = context->GetInputShape(HASH_Q_IDX);
    const gert::Shape* kShape = context->GetInputShape(HASH_K_CACHE_IDX);
    const gert::Shape* tableShape = context->GetInputShape(HASH_K_BLOCK_TABLE_IDX);
    gert::Shape* outShape = context->GetOutputShape(NEW_BLOCK_TABLE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    if (qShape->GetDimNum() != 4 || kShape->GetDimNum() != 5 || tableShape->GetDimNum() != 2) {
        OP_LOGE(context->GetNodeName(),
            "hash_q must be 4-D, hash_k_cache must be 5-D, and hash_k_block_table must be 2-D.");
        return GRAPH_FAILED;
    }

    const int64_t batch = qShape->GetDim(Q_DIM_BATCH);
    const int64_t tableBatch = tableShape->GetDim(TABLE_DIM_BATCH);
    const int64_t keyBatch = kShape->GetDim(K_DIM_BATCH);
    if (batch >= 0 && tableBatch >= 0 && batch != tableBatch) {
        OP_LOGE(context->GetNodeName(), "hash_q dim0 must equal hash_k_block_table dim0.");
        return GRAPH_FAILED;
    }
    if (batch >= 0 && keyBatch >= 0 && batch != keyBatch) {
        OP_LOGE(context->GetNodeName(), "hash_q dim0 must equal hash_k_cache dim1.");
        return GRAPH_FAILED;
    }
    if (qShape->GetDim(Q_DIM_BYTES) >= 0 && kShape->GetDim(K_DIM_BYTES) >= 0 &&
        qShape->GetDim(Q_DIM_BYTES) != kShape->GetDim(K_DIM_BYTES)) {
        OP_LOGE(context->GetNodeName(), "hash_q and hash_k_cache last dim must match.");
        return GRAPH_FAILED;
    }

    outShape->SetDimNum(2);
    outShape->SetDim(0, tableShape->GetDim(TABLE_DIM_BATCH));
    outShape->SetDim(1, tableShape->GetDim(TABLE_DIM_BLOCK));
    OP_LOGD(context->GetNodeName(), "End to infer HashBlockTopK shape.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(HashBlockTopK).InferShape(InferShapeForHashBlockTopK);
} // namespace ops
