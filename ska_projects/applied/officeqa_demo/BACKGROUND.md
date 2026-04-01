# Background: The SKA Document QA Pipeline

## The Two Stages

**Stage I (Offline): Structure Learning**

Raw text → sentences → embeddings → segments

The system splits a document into sentences, embeds each sentence as a
dense vector, then uses dynamic programming to find optimal "segments", groups of consecutive sentences that are semantically coherent. Unlike
fixed-size chunks (e.g. 512 tokens), these segments adapt to the
document's natural structure.

**Stage II (Online): Pricing-Guided Retrieval**

Query → embed → select segments → generate answer

Given a question, the system embeds it and selects the most relevant
segments using a "pricing" algorithm. The pricing engine adds segments
one at a time, only including a segment if its information gain exceeds
a cost threshold (λ). This prevents retrieving redundant or irrelevant
context.

## Why Not Just Use Vector Similarity?

Standard RAG systems retrieve the top-k most similar chunks. This has
two problems:
1. **Redundancy:** The top-k chunks may all say the same thing
2. **Fixed k:** Some questions need 1 chunk, others need 10

The pricing engine solves both: it stops when adding more segments
doesn't help (adaptive k), and it penalizes redundancy (η term).

## The PDF Processing Pipeline

```
PDF file
  → pdfplumber (extract text + tables)
  → TextPreprocessor.split_sentences() (NLTK sentence tokenizer)
  → Embedder.embed() (dense vectors, ~384 or 1536 dimensions)
  → GeometryLearner.learn_geometry() (DP segmentation)
  → List[Segment] (each with text, centroid vector, boundaries)
```

## References

No math papers needed for this project. The relevant code files are:
- `pipeline.py` (OfflinePipeline, RetrievalPipeline)
- `evaluation/officeqa.py` (DocumentProcessor)
- `core/pricing.py` (PricingEngine)
