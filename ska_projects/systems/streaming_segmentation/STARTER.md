# Starter: Streaming Segmentation

## Step 0: Understand the Batch Algorithm

Before writing any code, trace through `GeometryLearner.learn_geometry()`
by hand on a small example (N=20 sentences). Print dp, parent, and
boundaries at each step. Understand why lookback_k bounds the state.

```python
# Run this to see the batch algorithm in action on synthetic data
import numpy as np
from ska_agent.core.geometry import GeometryLearner
from ska_agent.utils.math_utils import MathUtils

np.random.seed(42)

# Create synthetic embeddings with clear cluster structure
# 3 clusters of 10 sentences each, with gaps between clusters
cluster1 = np.random.randn(10, 64) * 0.1 + np.random.randn(64)
cluster2 = np.random.randn(10, 64) * 0.1 + np.random.randn(64)
cluster3 = np.random.randn(10, 64) * 0.1 + np.random.randn(64)
embeddings = np.vstack([cluster1, cluster2, cluster3])
sentences = [f"Sentence {i}" for i in range(30)]

learner = GeometryLearner(lambda_seg=0.3, lookback_k=15,
                           min_segment_size=2, max_segment_size=15)
segments = learner.learn_geometry(embeddings, sentences, verbose=True)

print(f"\nSegments found: {len(segments)}")
for seg in segments:
    print(f"  [{seg.start_idx}, {seg.end_idx}): "
          f"cost={seg.internal_cost:.4f}, len={len(seg)}")
```

## Step 1: Circular Buffer Implementation

```python
class CircularBuffer:
    """Fixed-size circular buffer for streaming DP state."""

    def __init__(self, capacity: int, dtype=np.float64):
        self.capacity = capacity
        self._data = np.zeros(capacity, dtype=dtype)
        self._start = 0   # index of oldest element
        self._count = 0

    def append(self, value):
        idx = (self._start + self._count) % self.capacity
        self._data[idx] = value
        if self._count < self.capacity:
            self._count += 1
        else:
            self._start = (self._start + 1) % self.capacity

    def __getitem__(self, offset):
        """Get element by offset from the oldest element."""
        if offset < 0 or offset >= self._count:
            raise IndexError(f"Offset {offset} out of range [0, {self._count})")
        return self._data[(self._start + offset) % self.capacity]

    def get_absolute(self, global_idx, global_start):
        """Get element by absolute index (global_idx - global_start = offset)."""
        offset = global_idx - global_start
        return self[offset]

    @property
    def oldest_global_idx(self):
        """Override in subclass to track global indices."""
        raise NotImplementedError
```

## Step 2: Streaming DP Core

```python
class StreamingGeometryLearner:
    """
    Streaming segmentation with O(lookback_k) memory.

    Usage:
        learner = StreamingGeometryLearner(lambda_seg=0.3)

        for embedding, sentence in stream:
            segment = learner.feed(embedding, sentence)
            if segment is not None:
                # This segment is finalized
                process(segment)

        # Flush remaining segments
        for segment in learner.flush():
            process(segment)
    """

    def __init__(
        self,
        lambda_seg: float,
        lookback_k: int = 50,
        min_segment_size: int = 2,
        max_segment_size: int = 15,
    ):
        self.lambda_seg = lambda_seg
        self.lookback_k = lookback_k
        self.min_seg = min_segment_size
        self.max_seg = max_segment_size

        # DP state, circular buffers
        K = lookback_k + 1
        self._dp = np.full(K, np.inf)
        self._parent = np.zeros(K, dtype=int)
        self._dp[0] = 0.0  # dp[0] = 0

        # Distance prefix sums, circular buffer
        self._prefix_dist = np.zeros(K + 1)

        # Sentence + embedding buffer for building segments
        self._sentences = []   # bounded to lookback_k
        self._embeddings = []  # bounded to lookback_k

        # Global position tracking
        self._pos = 0           # next position to process
        self._window_start = 0  # oldest position in buffers
        self._prev_embedding = None

        # Finalized segments waiting to be emitted
        self._pending_segments = []

    def feed(self, embedding: np.ndarray, sentence: str):
        """
        Feed one sentence + embedding. Returns a Segment if one
        was finalized, else None.

        TODO: Implement the core streaming DP step.
        """
        # 1. Compute distance to previous embedding
        # 2. Update prefix_dist buffer
        # 3. Compute dp[pos] using lookback window
        # 4. Check if any segment can be finalized
        # 5. Return finalized segment if available

        self._pos += 1
        return None  # TODO

    def flush(self):
        """
        Finalize all remaining segments after the stream ends.

        TODO: Backtrack through parent pointers to extract
        remaining segments.
        """
        return []  # TODO
```

## Step 3: Correctness Test

```python
def test_streaming_matches_batch():
    """Verify streaming produces identical segments to batch."""
    np.random.seed(42)
    N = 500
    D = 64

    # Random embeddings with some structure
    embeddings = np.random.randn(N, D) * 0.1
    for i in range(0, N, 50):
        embeddings[i:i+50] += np.random.randn(D) * 0.5
    sentences = [f"S{i}" for i in range(N)]

    lambda_val = 0.3

    # Batch
    from ska_agent.core.geometry import GeometryLearner
    batch_learner = GeometryLearner(lambda_seg=lambda_val)
    batch_segments = batch_learner.learn_geometry(
        embeddings, sentences, verbose=False)

    # Streaming
    stream_learner = StreamingGeometryLearner(lambda_seg=lambda_val)
    stream_segments = []
    for i in range(N):
        seg = stream_learner.feed(embeddings[i], sentences[i])
        if seg is not None:
            stream_segments.append(seg)
    stream_segments.extend(stream_learner.flush())

    # Compare
    assert len(batch_segments) == len(stream_segments), \
        f"Count mismatch: {len(batch_segments)} vs {len(stream_segments)}"

    for b, s in zip(batch_segments, stream_segments):
        assert b.start_idx == s.start_idx, \
            f"Start mismatch: {b.start_idx} vs {s.start_idx}"
        assert b.end_idx == s.end_idx, \
            f"End mismatch: {b.end_idx} vs {s.end_idx}"

    print(f"PASSED: {len(batch_segments)} segments match exactly")
```

## Step 4: Memory Benchmark

```python
import tracemalloc

def benchmark_memory():
    """Show that streaming uses constant memory."""
    for N in [1_000, 10_000, 100_000]:
        np.random.seed(42)

        # Streaming
        tracemalloc.start()
        learner = StreamingGeometryLearner(lambda_seg=0.3)
        for i in range(N):
            emb = np.random.randn(64) * 0.1
            learner.feed(emb, f"S{i}")
        learner.flush()
        _, stream_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"N={N:>7d}: streaming peak = {stream_peak / 1024:.1f} KB")
```

Expected: streaming peak should be roughly constant (~50-100 KB)
regardless of N.
