/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hash_block_top_k.h"
#include "hash_block_top_k.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor* hashQ, const aclTensor* hashKCache, const aclTensor* k,
    const aclTensor* hashKBlockTable, const aclTensor* seqLen, const aclTensor* newBlockTable)
{
    OP_CHECK_NULL(hashQ, return false);
    OP_CHECK_NULL(hashKCache, return false);
    OP_CHECK_NULL(k, return false);
    OP_CHECK_NULL(hashKBlockTable, return false);
    OP_CHECK_NULL(seqLen, return false);
    OP_CHECK_NULL(newBlockTable, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* hashQ, const aclTensor* hashKCache, const aclTensor* k,
    const aclTensor* hashKBlockTable, const aclTensor* seqLen, const aclTensor* newBlockTable)
{
    OP_CHECK_DTYPE_NOT_MATCH(hashQ, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(hashKCache, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(k, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(hashKBlockTable, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(seqLen, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(newBlockTable, op::DataType::DT_INT32, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* hashQ, const aclTensor* hashKCache, const aclTensor* k,
    const aclTensor* hashKBlockTable, const aclTensor* seqLen, const aclTensor* newBlockTable)
{
    CHECK_RET(CheckNotNull(hashQ, hashKCache, k, hashKBlockTable, seqLen, newBlockTable), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(hashQ, hashKCache, k, hashKBlockTable, seqLen, newBlockTable), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHashBlockTopKGetWorkspaceSize(const aclTensor* hashQ, const aclTensor* hashKCache,
    const aclTensor* k, const aclTensor* hashKBlockTable, const aclTensor* seqLen, aclTensor* newBlockTable,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHashBlockTopK,
        DFX_IN(hashQ, hashKCache, k, hashKBlockTable, seqLen), DFX_OUT(newBlockTable));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(hashQ, hashKCache, k, hashKBlockTable, seqLen, newBlockTable);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto hashQContiguous = l0op::Contiguous(hashQ, uniqueExecutor.get());
    CHECK_RET(hashQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto hashKCacheContiguous = l0op::Contiguous(hashKCache, uniqueExecutor.get());
    CHECK_RET(hashKCacheContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
    CHECK_RET(kContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto blockTableContiguous = l0op::Contiguous(hashKBlockTable, uniqueExecutor.get());
    CHECK_RET(blockTableContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto seqLenContiguous = l0op::Contiguous(seqLen, uniqueExecutor.get());
    CHECK_RET(seqLenContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::HashBlockTopK(hashQContiguous, hashKCacheContiguous, kContiguous, blockTableContiguous,
        seqLenContiguous, newBlockTable, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, newBlockTable, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHashBlockTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHashBlockTopK);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
