# Starter: OfficeQA Demo

## Step 0: Dependencies

```bash
pip install streamlit pdfplumber nltk numpy
# For embeddings (CPU is fine):
pip install sentence-transformers
# For LLM (optional, can mock):
# pip install transformers torch
```

## Step 1: PDF Upload and Text Extraction

```python
import streamlit as st
from ska_agent.evaluation.officeqa import DocumentProcessor

st.title("📄 SKA-Agent Document QA")

uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded:
    # Save to temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(uploaded.read())
        tmp_path = f.name

    # Process
    processor = DocumentProcessor()
    with st.spinner("Extracting text from PDF..."):
        result = processor.process_pdf(tmp_path)

    st.success(f"Extracted {len(result['text'])} characters, "
               f"{len(result['tables'])} tables")

    with st.expander("Raw Text (first 2000 chars)"):
        st.text(result['text'][:2000])

    if result['graph']:
        with st.expander("Graph Representation"):
            st.code(result['graph'][:3000])

    os.unlink(tmp_path)

    # Store in session state for later
    st.session_state['doc_text'] = result['text']
    st.session_state['doc_tables'] = result['tables']
```

## Step 2: Segmentation

```python
if 'doc_text' in st.session_state:
    from ska_agent.pipeline import OfflinePipeline

    if st.button("Run Segmentation"):
        # Use a lightweight embedder
        with st.spinner("Loading embedder..."):
            pipeline = OfflinePipeline(embedding_model='all-MiniLM-L6-v2')

        progress = st.progress(0, text="Processing...")

        with st.spinner("Segmenting document..."):
            segments, embeddings, sentences = pipeline.process(
                st.session_state['doc_text'], verbose=True)

        progress.progress(100, text="Done!")

        st.success(f"Found {len(segments)} segments from {len(sentences)} sentences")

        # Display segments
        for i, seg in enumerate(segments):
            with st.expander(f"Segment {i} ({len(seg)} sentences, "
                             f"cost={seg.internal_cost:.3f})"):
                st.write(seg.text)

        st.session_state['segments'] = segments
        st.session_state['embedder'] = pipeline.embedder
```

## Step 3: Query and Retrieval

```python
if 'segments' in st.session_state:
    query = st.text_input("Ask a question about the document:")

    if query:
        from ska_agent.core.pricing import PricingEngine

        engine = PricingEngine(
            segments=st.session_state['segments'],
            embed_fn=lambda q: st.session_state['embedder'].embed_single(q),
            lambda_sparsity=0.05,
            max_segments=5,
        )

        result = engine.retrieve(query, verbose=False)

        st.subheader(f"Retrieved {len(result.segments)} segments")
        for seg, rc in zip(result.segments, result.reduced_costs):
            st.markdown(f"**Reduced cost: {rc:.4f}**")
            st.info(seg.text)

        # Generate answer (mock or real)
        context = result.get_context()
        st.subheader("Answer")
        st.write(f"*Based on {len(result.segments)} retrieved segments:*")
        # For a real answer, use LLMGenerator here
        # For mock: just show the context
        st.write(context[:500])
```

## Step 4: Pipeline Visualization

```python
# Simple step indicator
steps = ["Upload PDF", "Extract Text", "Segment", "Query", "Retrieve", "Answer"]
current_step = 3  # update based on progress

cols = st.columns(len(steps))
for i, (col, step) in enumerate(zip(cols, steps)):
    if i < current_step:
        col.markdown(f"✅ **{step}**")
    elif i == current_step:
        col.markdown(f"🔄 **{step}**")
    else:
        col.markdown(f"⬜ {step}")
```
