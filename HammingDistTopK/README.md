# HammingDistTopK 算子（CANN 8.5.0 / Ascend910B）

计算 query 与压缩 key 库的汉明距离，返回距离最小的 Top-K 个 key 索引（chunk 级）。
面向 ANN 检索 / 向量库搜索场景。本工程按 AscendC `OpDef` 自定义算子框架组织。

## 目录结构

```
HammingDistTopK/
├── op_host/
│   ├── hamming_dist_top_k_tiling.h   # TilingData 定义
│   └── hamming_dist_top_k.cpp        # OpDef + InferShape/InferDataType + TilingFunc
├── op_kernel/
│   └── hamming_dist_top_k.cpp        # AscendC 核函数
├── op_info_cfg/ascend910b/
│   └── hamming_dist_top_k.ini        # 算子信息库
├── CMakeLists.txt                    # 构建片段（需挂接到上层 custom op 工程）
└── README.md
```

## 接口（与规格一致）

| 名称 | 角色 | dtype | 形状 |
|---|---|---|---|
| query | 输入(必选) | uint8 | [batch, qHead, 1, dim/8] |
| key_compressed | 输入(必选) | uint8 | [batch, head, maxSeqLen, dim/8] |
| k | 输入(必选) | int32 | [batch] |
| seq_len | 输入(必选) | int32 | [batch] |
| chunk_size | 输入(可选) | int32 | [batch] |
| key_block_table | 输入(可选) | int32 | [batch, blockCount] |
| indices | 输出(必选) | int32 | [batch, head, maxOutputChunkLen] |
| topk | 属性 | int | Top-K 值 |

## 规格 → 代码映射

- **uint8 → int4b_t 解包**：`op_kernel` 的 `SelectAndCast()`（Select+Cast，每 bit 展开为 int4）。
- **汉明距离 = Matmul(query, key)**：`ComputeDistance()`，int4b_t 矩阵乘、half 输出。
- **chunk ReduceMax**：`ReduceChunk()`，chunk_size>1 时块内取最大汉明距离。
- **TopK 取最小 K**：`TopKSmallest()`，TOPK_NORMAL，对距离取负求最大等价取最小。
- **block_table 寻址**：`KeyPhysOffset()`，continFlag=1 走查表，=0 走连续 offset。
- **tiling 两分支 / 参数计算**：`op_host` 的 `SelectBranch()` / `CalcMaxChunkSize()` /
  `CalcMaxOutputChunkLen()`；effectLen 在 kernel `ProcessOne()` 计算。
- **tilingKey = branch*10 + continFlag**，kernel 用 `TILING_KEY_IS` 分发 4 个模板特化。

## tiling 分支（规格表）

| 条件 | branch | tileN1 | tileN2 |
|---|---|---|---|
| maxSeqLen>26K 或 (batch<16 且 maxSeqLen>8K) | SplitS | 128 | 4*1024 |
| 其他 | Basic/Parallel | 254 | 3328 |
| continFlag=true（非连续） | — | tileN1=blockSize | — |

## 构建（在 CANN 8.5.0 环境）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 放入你的 custom op 工程后，按工程方式构建，例如：
bash build.sh                 # 工程脚本会调用 opbuild + ascendc 编译
# 产物：custom_opp_*.run（含 op_proto / tiling / device bin / 信息库）
```

## 上机待调点（无法在离线静态代码中确定，需在 910B 上验证）

1. **SelectAndCast 的位展开实现**：uint8 每 bit → int4b_t 的具体向量指令序列
   （SelectCustom / 位提取 / Cast 链）需结合 8.5.0 vec API 落地，并确认 0/1 还是
   ±1 编码能让 int4 Matmul 结果正确等价于 POPCOUNT(q XOR k)。
2. **TopK 模板参数**：`lib/sort/topk/topk.h` 的实际签名（TopKInfo / 是否需排序 buffer /
   取最小的实现方式）按版本头文件对齐；当前给的是调用骨架。
3. **Matmul 成员可见性**：`REGIST_MATMUL_OBJ` 需访问 `op.mm_`，集成时把 `mm_` 设为
   public 或按工程惯例用 friend 宏暴露。
4. **ReduceChunk 向量化**：当前用标量循环表达语义，大序列应替换为 `WholeReduceMax`
   等向量指令提速。
5. **SplitS 分支**：tileN2=4*1024 的二级切分在超长序列下的 UB 切分与流水需实测调优。
6. **workspace 大小**：host 侧按 16MB 粗估，应据 tileN1×dim 解包缓冲与 matmul 中间量收紧。
7. **末尾不完整 chunk 填充**：`effectLen` 跳过后，输出末尾的填充值/填充长度需与下游
   约定对齐（规格“输出索引含义”）。

## 输出语义

`indices` 为 **chunk 级**索引；对应 token 区间为
`[indices[i]*chunkSize, (indices[i]+1)*chunkSize)`；末尾不完整 chunk 填充到输出末尾。
