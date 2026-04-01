# Evaluation: Streaming Segmentation

## Grading Breakdown

| Component | Weight | Description |
|-----------|--------|-------------|
| Correctness | 35% | Identical output to batch algorithm on all test cases |
| Streaming design | 25% | Proper bounded memory, segment finalization logic |
| Memory benchmarks | 20% | Demonstrated constant memory on 1K–100K+ inputs |
| Integration | 10% | StreamingOfflinePipeline with micro-batch embedding |
| Edge cases | 10% | Empty input, single sentence, all-identical embeddings |

## What "Done" Looks Like

### A-level
- Streaming and batch produce bit-identical segments on all test corpora
- Memory usage is provably O(lookback_k · D), demonstrated empirically
- Clean iterator/generator API for segment emission
- Micro-batch embedding pipeline implemented
- Handles edge cases gracefully
- Writeup includes the correctness proof from BACKGROUND.md verified empirically

### B-level
- Correctness on most test cases (may have off-by-one in finalization)
- Memory is bounded but may be slightly larger than optimal
- Basic integration

### C-level
- DP works in streaming mode but finalization is incorrect
- Memory savings not demonstrated
