# Project: OfficeQA Demo App + Retrieval Strategy Comparison

## Summary

Build a web application where a user can upload a PDF and ask questions,
then use it to run a systematic comparison between SKA's pricing-guided
retrieval and standard baselines (top-k cosine, BM25, fixed-size chunking).

**Area:** Applied
**Tier:** Starter
**GPU:** Minimal (CPU embedder is fine)
**Duration:** 8 weeks
**Team size:** 1–2

## Motivation

The SKA-Agent codebase has all the pieces for document QA, but there's
no way to actually use it without writing scripts. A demo app makes the
system tangible. More importantly, the claim that pricing-guided retrieval
beats standard retrieval hasn't been tested in this codebase, the second
half of the project validates that claim empirically.

## Deliverables

### Phase 1 (Weeks 1–4): Demo App

1. **PDF upload and processing:** Drop PDF → extract text → segment
   with progress bar. Display sentence count, segment count, boundaries.

2. **Query interface:** Text input. Show retrieved segments with scores,
   generated answer. Expandable detail: mode selected, actions taken.

3. **Segment browser:** Table of all segments. Click to expand. Highlight
   retrieved segments for current query.

4. **Pipeline visualization:** Step-by-step: PDF → Sentences → Segments
   → Query → Retrieve → Answer, with timing at each step.

### Phase 2 (Weeks 5–8): Retrieval Comparison Study

5. **Baseline retrievers:** Implement 3 baselines alongside the pricing engine:
   - **Top-k cosine:** Embed query, return k most similar segments (standard RAG)
   - **Top-k cosine with fixed-size chunks:** Split document into 512-token chunks
     instead of DP segments, then do top-k
   - **BM25:** Sparse keyword retrieval over segments

6. **Side-by-side comparison UI:** For a given query, show retrieval results
   from all 4 methods in parallel columns. User can compare which segments
   each method chose and how they differ.

7. **Quantitative evaluation:** On a set of 50+ queries with ground-truth
   answers (either manually created or from an existing QA dataset):
   - Retrieval precision: what fraction of retrieved segments are relevant?
   - Answer accuracy: `score_answer()` on each method's output
   - Efficiency: segments retrieved per query (fewer = more efficient)
   - λ sensitivity: how does the pricing engine's performance change with λ?

8. **Written comparison report:** 2–3 pages analyzing:
   - Where pricing-guided retrieval wins (and by how much)
   - Where baselines are competitive or better
   - The effect of DP segmentation vs. fixed-size chunking
   - Recommendations for default configuration

## Where This Fits

```
ska_agent/pipeline.py
    OfflinePipeline.process()       ← text → segments
    RetrievalPipeline()             ← segments + query → context

ska_agent/core/pricing.py
    PricingEngine.retrieve()        ← pricing-guided retrieval

ska_agent/core/geometry.py
    GeometryLearner.learn_geometry() ← DP segmentation

ska_agent/evaluation/officeqa.py
    DocumentProcessor.process_pdf()  ← PDF → text + tables
    score_answer()                   ← accuracy scoring
```

## Milestones

| Week | Milestone |
|------|-----------|
| 1 | PDF upload → text extraction → sentence display. |
| 2 | Segmentation with progress. Segment browser. |
| 3 | Query → retrieval with segment highlighting. |
| 4 | Answer generation (real LLM or mock). Pipeline visualization. |
| 5 | Implement top-k cosine and fixed-chunk baselines. |
| 6 | Implement BM25 baseline. Side-by-side UI. |
| 7 | Quantitative evaluation on 50+ queries. |
| 8 | Written comparison report. Polish. |
