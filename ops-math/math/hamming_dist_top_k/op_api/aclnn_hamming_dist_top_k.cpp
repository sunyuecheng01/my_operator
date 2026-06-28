/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_hamming_dist_top_k.h"
#include "hamming_dist_top_k.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor* query, const aclTensor* keyCompressed, const aclTensor* k,
    const aclTensor* seqLen, const aclTensor* indices)
{
    OP_CHECK_NULL(query, return false);
    OP_CHECK_NULL(keyCompressed, return false);
    OP_CHECK_NULL(k, return false);
    OP_CHECK_NULL(seqLen, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* query, const aclTensor* keyCompressed, const aclTensor* k,
    const aclTensor* seqLen, const aclTensor* chunkSizeOptional, const aclTensor* keyBlockTableOptional,
    const aclTensor* indices)
{
    OP_CHECK_DTYPE_NOT_MATCH(query, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(keyCompressed, op::DataType::DT_UINT8, return false);
    OP_CHECK_DTYPE_NOT_MATCH(k, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(seqLen, op::DataType::DT_INT32, return false);
    OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT32, return false);
    if (chunkSizeOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(chunkSizeOptional, op::DataType::DT_INT32, return false);
    }
    if (keyBlockTableOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_MATCH(keyBlockTableOptional, op::DataType::DT_INT32, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* query, const aclTensor* keyCompressed, const aclTensor* k,
    const aclTensor* seqLen, const aclTensor* chunkSizeOptional, const aclTensor* keyBlockTableOptional,
    const aclTensor* indices)
{
    // 1. 检查必选参数是否为空指针
    CHECK_RET(CheckNotNull(query, keyCompressed, k, seqLen, indices), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否符合算子要求
    CHECK_RET(CheckDtypeValid(query, keyCompressed, k, seqLen, chunkSizeOptional, keyBlockTableOptional, indices),
        ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHammingDistTopKGetWorkspaceSize(const aclTensor* query, const aclTensor* keyCompressed,
    const aclTensor* k, const aclTensor* seqLen, const aclTensor* chunkSizeOptional,
    const aclTensor* keyBlockTableOptional, int64_t topk, aclTensor* indices, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnHammingDistTopK,
        DFX_IN(query, keyCompressed, k, seqLen, chunkSizeOptional, keyBlockTableOptional, topk), DFX_OUT(indices));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(query, keyCompressed, k, seqLen, chunkSizeOptional, keyBlockTableOptional, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 入参转换为连续张量；可选输入为空时跳过
    auto queryContiguous = l0op::Contiguous(query, uniqueExecutor.get());
    CHECK_RET(queryContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto keyContiguous = l0op::Contiguous(keyCompressed, uniqueExecutor.get());
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto kContiguous = l0op::Contiguous(k, uniqueExecutor.get());
    CHECK_RET(kContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto seqLenContiguous = l0op::Contiguous(seqLen, uniqueExecutor.get());
    CHECK_RET(seqLenContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* chunkSizeContiguous = nullptr;
    if (chunkSizeOptional != nullptr) {
        chunkSizeContiguous = l0op::Contiguous(chunkSizeOptional, uniqueExecutor.get());
        CHECK_RET(chunkSizeContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* keyBlockTableContiguous = nullptr;
    if (keyBlockTableOptional != nullptr) {
        keyBlockTableContiguous = l0op::Contiguous(keyBlockTableOptional, uniqueExecutor.get());
        CHECK_RET(keyBlockTableContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::HammingDistTopK(queryContiguous, keyContiguous, kContiguous, seqLenContiguous,
        chunkSizeContiguous, keyBlockTableContiguous, topk, indices, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, indices, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnHammingDistTopK(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnHammingDistTopK);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
