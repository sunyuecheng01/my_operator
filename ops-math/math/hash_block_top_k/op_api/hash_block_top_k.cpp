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
 * \file hash_block_top_k.cpp
 * \brief HashBlockTopK level-0 op implementation. Builds the AICORE launch task.
 */
#include "hash_block_top_k.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(HashBlockTopK);

const aclTensor* HashBlockTopK(const aclTensor* hashQ, const aclTensor* hashKCache, const aclTensor* k,
    const aclTensor* hashKBlockTable, const aclTensor* seqLen, const aclTensor* newBlockTable,
    aclOpExecutor* executor)
{
    L0_DFX(HashBlockTopK, hashQ, hashKCache, k, hashKBlockTable, seqLen);

    auto out = executor->AllocTensor(newBlockTable->GetViewShape(), op::DataType::DT_INT32, op::Format::FORMAT_ND);
    if (out == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "alloc new_block_table tensor failed.");
        return nullptr;
    }

    ADD_TO_LAUNCHER_LIST_AICORE(HashBlockTopK,
        OP_INPUT(hashQ, hashKCache, k, hashKBlockTable, seqLen), OP_OUTPUT(out));
    return out;
}
} // namespace l0op
