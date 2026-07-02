# HashBlockTopK

`HashBlockTopK` scores paged hash-K cache blocks with compressed hash-Q, then builds a compact block table
containing the Top-K physical block ids for each batch.

## Inputs

- `hash_q`: `uint8`, shape `[batch, qSeqLen, qHeadNum, headDim / 8]`.
- `hash_k_cache`: `uint8`, shape `[physicalBlockCount, batch, blockSize, headNum, headDim / 8]`.
- `k`: `int32`, scalar or shape `[batch]`. Runtime number of blocks selected per batch.
- `hash_k_block_table`: `int32`, shape `[batch, blockCount]`.
  Maps logical block indices to physical block ids in `hash_k_cache`.
- `seqlen`: `int32`, scalar or shape `[batch]`. Runtime valid KV token length.

## Output

- `new_block_table`: `int32`, shape `[batch, blockCount]`.

The first `min(k, blockCount)` entries contain selected physical block ids. Unused entries are `-1`.
Invalid logical blocks, negative physical block ids, physical ids outside `[0, physicalBlockCount)`, and tokens beyond
`seqlen` do not participate in Top-K.

## Scoring

The kernel expands packed bits to signed `int8` values (`0 -> -1`, `1 -> +1`) and uses CUBE Matmul to compute
hash similarity. Larger scores mean smaller Hamming distance:

```text
similarity = hashBits - 2 * hamming_distance
```

For one logical block:

1. For every query token and every valid key token in the block, compute hash similarity.
2. Aggregate similarities over head-pairs.
3. Use the highest aggregated QK score inside the block as the block score.
4. Select Top-K logical blocks by block score and output their physical block ids from `hash_k_block_table`.

## Head Mapping

This op does not expose a custom head-map tensor. It supports the common divisible head-count families:

- MHA: `qHeadNum == headNum`.
- GQA/MQA: `qHeadNum >= headNum && qHeadNum % headNum == 0`.
- Shared-query fallback: `headNum > qHeadNum && headNum % qHeadNum == 0`.

Internally, the number of scored head-pairs is `max(qHeadNum, headNum)`.

For fully arbitrary head mapping, add explicit `q_head_map` and `kv_head_map` inputs and make the tiling dimension
`pairCount = map_length`.

## Tiling

- Multi-core main split: `batch * blockCount`.
- Sequence split: each task processes one logical block and splits the block's token dimension by `tileN2`.
- Hash dimension: `expandedDim = headDim` is consumed as the CUBE K dimension and is not split by this version.
- Target SoC: Ascend 910B.

## Notes

- `seqlen` is still required for paged KV cache because `hash_k_block_table` only maps logical blocks to physical
  blocks. It does not tell the kernel which tokens in the tail block are valid.
- The current implementation prioritizes a working aclnn-direct CUBE path. A later vector path can replace the
  bit expansion + CUBE score stage with XOR + popcount while preserving the same public API.
