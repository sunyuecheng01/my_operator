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
 * \file hamming_dist_top_k.cpp
 * \brief HammingDistTopK level-0 op implementation. Builds the AICORE launch task.
 */
#include "hamming_dist_top_k.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(HammingDistTopK);

const aclTensor* HammingDistTopK(const aclTensor* query, const aclTensor* keyCompressed, const aclTensor* k,
    const aclTensor* seqLen, const aclTensor* chunkSizeOptional, const aclTensor* keyBlockTableOptional, int64_t topk,
    const aclTensor* indices, aclOpExecutor* executor)
{
    L0_DFX(HammingDistTopK, query, keyCompressed, k, seqLen, topk);

    // Output shape is fully determined by the infershape stage and carried by `indices`.
    auto out = executor->AllocTensor(indices->GetViewShape(), op::DataType::DT_INT32, op::Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc indices tensor failed.");
        return nullptr;
    }

    // chunkSizeOptional / keyBlockTableOptional may be nullptr; positional slots are preserved.
    ADD_TO_LAUNCHER_LIST_AICORE(HammingDistTopK,
        OP_INPUT(query, keyCompressed, k, seqLen, chunkSizeOptional, keyBlockTableOptional), OP_OUTPUT(out),
        OP_ATTR(topk));
    return out;
}
} // namespace l0op
