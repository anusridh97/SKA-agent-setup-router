# Starter: Router Dashboard

## Step 0: Install Dependencies

```bash
pip install streamlit plotly numpy
# You also need the ska_agent package
tar xf ska_agent-1_0_0-7.tar
pip install -e .  # or add to PYTHONPATH
```

## Step 1: Minimal Working Dashboard (30 minutes)

Create `dashboard.py`:

```python
import streamlit as st
import numpy as np

st.set_page_config(page_title="SKA Router Dashboard", layout="wide")
st.title("🧭 SKA-Agent Router Dashboard")

# --- Query Input ---
query = st.text_input("Enter a query:", "What was the total federal debt in 2023?")

if st.button("Route"):
    # For now, mock the router output
    # We'll wire up the real router in Step 2
    mode_probs = np.array([0.7, 0.1, 0.15, 0.05])
    mode_names = ["LOOKUP", "MULTI_DOC", "COMPUTE", "MULTI_STEP"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Mode Selection")
        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            x=mode_probs,
            y=mode_names,
            orientation='h',
            marker_color=['#2ecc71' if i == np.argmax(mode_probs)
                          else '#bdc3c7' for i in range(4)]
        ))
        fig.update_layout(xaxis_title="Probability", height=250,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Selected Mode")
        selected = mode_names[np.argmax(mode_probs)]
        st.metric("Mode", selected, f"{mode_probs.max():.1%} confidence")

# Run with: streamlit run dashboard.py
```

**Run this now.** You should see a working (mocked) dashboard in ~30 minutes.

## Step 2: Wire Up the Real Router

Replace the mock with actual router calls. The router needs
`sentence-transformers` for the query encoder, which is ~100MB.

```python
# At the top of dashboard.py
import sys
sys.path.insert(0, ".")  # or wherever ska_agent is

from ska_agent.router.adaptive_router import AdaptiveRouter
from ska_agent.core.structures import SystemConfig, MODE_TEMPLATES

@st.cache_resource
def load_router():
    """Load once, reuse across queries."""
    config = SystemConfig()
    return AdaptiveRouter(config=config)

router = load_router()

# In the button handler:
if st.button("Route"):
    # Get mode selection
    e_Q = router.encoder.encode(query)
    mode, mode_probs = router.mode_selector.predict(e_Q)

    # Display mode probabilities (same chart code as above,
    # but with real probs)

    # Get action scores
    template = MODE_TEMPLATES[mode]
    current_node = "start"
    candidates = router._enumerate_candidates(current_node, template)

    st.subheader("Action Candidates")
    for action in candidates:
        score = router.scorer.score_action(e_Q, action)
        st.write(f"**{action.model}** → {action.target}: "
                 f"score={score:.4f} "
                 f"(reward={action.predicted_reward:.4f})")

    # Run the full routing
    mock_specialists = {
        'ska_retriever': lambda q: "[Retrieved context]",
        'code_executor': lambda q: "42",
        'reasoner': lambda q: f"[Reasoned about: {q[:50]}]",
        'parser': lambda q: f"[Parsed: {q[:50]}]",
    }
    results = router.route(query, specialists=mock_specialists, verbose=False)

    st.subheader("Execution Trace")
    for i, r in enumerate(results):
        with st.expander(f"Step {i+1}: {r.action.model} → {r.action.target}"):
            st.write(f"**Output:** {r.output[:200]}")
            st.write(f"**Cost:** {r.actual_cost}")
```

## Step 3: PID Dynamics Panel

```python
# Track PID state across queries using session state
if 'pid_history' not in st.session_state:
    st.session_state.pid_history = []

# After routing, record the PID state
st.session_state.pid_history.append(router.pid.lambda_vec.copy())

# Plot
if len(st.session_state.pid_history) > 1:
    st.subheader("PID Cost Dynamics")
    import plotly.graph_objects as go

    history = np.array(st.session_state.pid_history)
    dim_names = ["λ_in", "λ_out", "λ_lat", "λ_$", "λ_meta"]

    fig = go.Figure()
    for i, name in enumerate(dim_names):
        fig.add_trace(go.Scatter(
            y=history[:, i], name=name, mode='lines+markers'
        ))
    fig.update_layout(xaxis_title="Query #", yaxis_title="λ value",
                      height=300)
    st.plotly_chart(fig, use_container_width=True)
```

## Step 4: DAG Visualization

```python
# Use graphviz or a simple HTML/CSS grid
def render_dag(template, executed_nodes):
    """Render a collaboration mode DAG."""
    dot = "digraph {\n  rankdir=LR;\n"
    for node, children in template.items():
        color = "green" if node in executed_nodes else "lightgray"
        dot += f'  "{node}" [style=filled, fillcolor={color}];\n'
        for child in children:
            dot += f'  "{node}" -> "{child}";\n'
    dot += "}"
    st.graphviz_chart(dot)

# Usage:
render_dag(MODE_TEMPLATES[mode],
           executed_nodes={r.action.target for r in results})
```
