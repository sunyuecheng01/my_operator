/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclnn 调用示例：HashBlockTopK。
//
//   hash_q            : uint8 [2, 1, 4, 8]
//   hash_k_cache      : uint8 [8, 2, 16, 2, 8]
//   k                 : int32 [2] = {3, 3}
//   hash_k_block_table: int32 [2, 4]
//   seqlen            : int32 [2] = {64, 48}
//   new_block_table   : int32 [2, 4]

#include <iostream>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_hash_block_top_k.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t BATCH = 2;
constexpr int64_t Q_SEQ_LEN = 1;
constexpr int64_t Q_HEAD = 4;
constexpr int64_t HEAD = 2;
constexpr int64_t DIM_BYTES = 8;
constexpr int64_t PHYSICAL_BLOCK_COUNT = 8;
constexpr int64_t BLOCK_COUNT = 4;
constexpr int64_t BLOCK_SIZE = 16;
constexpr int64_t TOP_K_BLOCKS = 3;
constexpr int32_t PRINT_LEN = 4;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    return size;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::mt19937 gen(42);
    std::uniform_int_distribution<int> byteDist(0, 255);

    std::vector<int64_t> qShape = {BATCH, Q_SEQ_LEN, Q_HEAD, DIM_BYTES};
    std::vector<uint8_t> qHost(GetShapeSize(qShape));
    for (auto& v : qHost) {
        v = static_cast<uint8_t>(byteDist(gen));
    }

    std::vector<int64_t> kCacheShape = {PHYSICAL_BLOCK_COUNT, BATCH, BLOCK_SIZE, HEAD, DIM_BYTES};
    std::vector<uint8_t> kCacheHost(GetShapeSize(kCacheShape));
    for (auto& v : kCacheHost) {
        v = static_cast<uint8_t>(byteDist(gen));
    }

    std::vector<int64_t> batchVecShape = {BATCH};
    std::vector<int32_t> kHost(BATCH, static_cast<int32_t>(TOP_K_BLOCKS));
    std::vector<int32_t> seqLenHost = {64, 48};

    std::vector<int64_t> tableShape = {BATCH, BLOCK_COUNT};
    std::vector<int32_t> blockTableHost = {
        0, 1, 2, 3,
        4, 5, 6, 7,
    };
    std::vector<int32_t> newBlockTableHost(GetShapeSize(tableShape), -1);

    aclTensor* hashQ = nullptr;
    aclTensor* hashKCache = nullptr;
    aclTensor* k = nullptr;
    aclTensor* hashKBlockTable = nullptr;
    aclTensor* seqLen = nullptr;
    aclTensor* newBlockTable = nullptr;
    void* qAddr = nullptr;
    void* kCacheAddr = nullptr;
    void* kAddr = nullptr;
    void* blockTableAddr = nullptr;
    void* seqLenAddr = nullptr;
    void* newBlockTableAddr = nullptr;

    ret = CreateAclTensor(qHost, qShape, &qAddr, aclDataType::ACL_UINT8, &hashQ);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kCacheHost, kCacheShape, &kCacheAddr, aclDataType::ACL_UINT8, &hashKCache);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, batchVecShape, &kAddr, aclDataType::ACL_INT32, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(blockTableHost, tableShape, &blockTableAddr, aclDataType::ACL_INT32, &hashKBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqLenHost, batchVecShape, &seqLenAddr, aclDataType::ACL_INT32, &seqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(newBlockTableHost, tableShape, &newBlockTableAddr, aclDataType::ACL_INT32, &newBlockTable);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnHashBlockTopKGetWorkspaceSize(hashQ, hashKCache, k, hashKBlockTable, seqLen, newBlockTable,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnHashBlockTopKGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnHashBlockTopK(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHashBlockTopK failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    std::vector<int32_t> resultData(GetShapeSize(tableShape), -1);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), newBlockTableAddr,
        resultData.size() * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("HashBlockTopK new_block_table[batch=0], first %d entries:\n", PRINT_LEN);
    for (int32_t i = 0; i < PRINT_LEN; i++) {
        LOG_PRINT("  [%d] = %d\n", i, resultData[i]);
    }

    aclDestroyTensor(hashQ);
    aclDestroyTensor(hashKCache);
    aclDestroyTensor(k);
    aclDestroyTensor(hashKBlockTable);
    aclDestroyTensor(seqLen);
    aclDestroyTensor(newBlockTable);

    aclrtFree(qAddr);
    aclrtFree(kCacheAddr);
    aclrtFree(kAddr);
    aclrtFree(blockTableAddr);
    aclrtFree(seqLenAddr);
    aclrtFree(newBlockTableAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
