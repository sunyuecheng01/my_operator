/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// aclnn 调用示例：HammingDistTopK，连续 KV 布局（不传 key_block_table）。
//
// 构造参数（小规模便于验证）：
//   batch=2, qHead=2, head=2, dim=64 -> dim/8=8, chunk_size=32, seq_len=256
//   query          : uint8 [2, 2, 1, 8]
//   key_compressed : uint8 [2, 2, 256, 8]      (连续布局)
//   k              : int32 [2] = {4, 4}        (每个请求选 4 个 chunk)
//   seq_len        : int32 [2] = {256, 256}
//   chunk_size     : int32 [2] = {32, 32}
//   key_block_table: 不传 -> nullptr
//   topk(attr)     : 128 (默认)
//   indices(out)   : int32 [2, 2, 165]         (165 = CalcOutputChunkLen(256, 128))

#include <iostream>
#include <vector>
#include <random>
#include "acl/acl.h"
#include "aclnnop/aclnn_hamming_dist_top_k.h"

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
constexpr int64_t Q_HEAD = 2;
constexpr int64_t HEAD = 2;
constexpr int64_t DIM_BYTES = 8;          // dim / 8, dim = 64
constexpr int64_t SEQ_LEN = 256;
constexpr int64_t CHUNK_SIZE = 32;
constexpr int64_t TOP_K_CHUNKS = 4;       // runtime k (chunk count)
constexpr int64_t TOPK_ATTR = 128;        // static attr
constexpr int64_t OUTPUT_CHUNK_LEN = 165; // CalcOutputChunkLen(256, 128)
constexpr int32_t PRINT_LEN = 16;

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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
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

    // ---- 构造输入数据 ----
    std::mt19937 gen(42);
    std::uniform_int_distribution<int> byteDist(0, 255);

    std::vector<int64_t> queryShape = {BATCH, Q_HEAD, 1, DIM_BYTES};
    std::vector<uint8_t> queryHost(GetShapeSize(queryShape));
    for (auto& v : queryHost) {
        v = static_cast<uint8_t>(byteDist(gen));
    }

    std::vector<int64_t> keyShape = {BATCH, HEAD, SEQ_LEN, DIM_BYTES}; // 连续布局，无 block_table
    std::vector<uint8_t> keyHost(GetShapeSize(keyShape));
    for (auto& v : keyHost) {
        v = static_cast<uint8_t>(byteDist(gen));
    }

    std::vector<int64_t> vecShape = {BATCH};
    std::vector<int32_t> kHost(BATCH, static_cast<int32_t>(TOP_K_CHUNKS));
    std::vector<int32_t> seqLenHost(BATCH, static_cast<int32_t>(SEQ_LEN));
    std::vector<int32_t> chunkSizeHost(BATCH, static_cast<int32_t>(CHUNK_SIZE));

    std::vector<int64_t> indicesShape = {BATCH, HEAD, OUTPUT_CHUNK_LEN};
    std::vector<int32_t> indicesHost(GetShapeSize(indicesShape), -1);

    aclTensor* query = nullptr;
    aclTensor* keyCompressed = nullptr;
    aclTensor* k = nullptr;
    aclTensor* seqLen = nullptr;
    aclTensor* chunkSize = nullptr;
    aclTensor* indices = nullptr;
    void* queryAddr = nullptr;
    void* keyAddr = nullptr;
    void* kAddr = nullptr;
    void* seqLenAddr = nullptr;
    void* chunkSizeAddr = nullptr;
    void* indicesAddr = nullptr;

    ret = CreateAclTensor(queryHost, queryShape, &queryAddr, aclDataType::ACL_UINT8, &query);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(keyHost, keyShape, &keyAddr, aclDataType::ACL_UINT8, &keyCompressed);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(kHost, vecShape, &kAddr, aclDataType::ACL_INT32, &k);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(seqLenHost, vecShape, &seqLenAddr, aclDataType::ACL_INT32, &seqLen);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(chunkSizeHost, vecShape, &chunkSizeAddr, aclDataType::ACL_INT32, &chunkSize);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHost, indicesShape, &indicesAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // ---- 两段式 aclnn 调用 ----
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // key_block_table 传 nullptr -> 走连续 KV 路径
    ret = aclnnHammingDistTopKGetWorkspaceSize(query, keyCompressed, k, seqLen, chunkSize, /*keyBlockTable=*/nullptr,
        TOPK_ATTR, indices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnHammingDistTopKGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnHammingDistTopK(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnHammingDistTopK failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // ---- 取回结果并打印 ----
    auto outSize = GetShapeSize(indicesShape);
    std::vector<int32_t> resultData(outSize, -1);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(int32_t), indicesAddr,
        outSize * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("HammingDistTopK indices[batch=0, head=0], first %d entries:\n", PRINT_LEN);
    for (int32_t i = 0; i < PRINT_LEN; i++) {
        LOG_PRINT("  [%d] = %d\n", i, resultData[i]);
    }

    // ---- 释放资源 ----
    aclDestroyTensor(query);
    aclDestroyTensor(keyCompressed);
    aclDestroyTensor(k);
    aclDestroyTensor(seqLen);
    aclDestroyTensor(chunkSize);
    aclDestroyTensor(indices);

    aclrtFree(queryAddr);
    aclrtFree(keyAddr);
    aclrtFree(kAddr);
    aclrtFree(seqLenAddr);
    aclrtFree(chunkSizeAddr);
    aclrtFree(indicesAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
