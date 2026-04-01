# Starter: Memory Inspector

## Step 1: Minimal Spectrum Viewer (20 minutes)

```python
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from ska_agent.shared_memory.spectral_memory import SharedSpectralMemory

st.title("🔬 Spectral Memory Inspector")

# Create a shared memory and write some data
rank = 64
mem = SharedSpectralMemory(rank=rank, ridge_eps=1e-3)

# Simulate writes
n_writes = st.slider("Number of writes", 1, 50, 10)
np.random.seed(42)
for i in range(n_writes):
    keys = np.random.randn(5, rank) * 0.1
    mem.write(keys, source_agent=f"agent_{i % 3}")

# Get operator
op = mem.operator
if op is not None:
    # Singular values
    svs = np.linalg.svd(op.A_w, compute_uv=False)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Operator Spectrum")
        fig = go.Figure(go.Bar(y=svs, marker_color='steelblue'))
        fig.update_layout(xaxis_title="Index", yaxis_title="σ_i",
                          height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Health Metrics")
        st.metric("Condition Number κ(G)", f"{op.condition_number:.1f}",
                  delta="OK" if op.condition_number < 1e4 else "⚠️ HIGH")
        st.metric("Tokens Seen", op.num_tokens_seen)
        st.metric("Spectral Radius", f"{svs[0]:.4f}")
        st.metric("Spectral Gap", f"{svs[0] - svs[1]:.4f}")
```

## Step 2: Condition Number Timeline

```python
# Track condition number across writes
st.subheader("Condition Number Over Time")

mem2 = SharedSpectralMemory(rank=rank, ridge_eps=1e-3)
kappas = []
np.random.seed(42)
for i in range(100):
    keys = np.random.randn(3, rank) * 0.1
    mem2.write(keys)
    op = mem2.operator
    kappas.append(op.condition_number)

fig = go.Figure()
fig.add_trace(go.Scatter(y=kappas, mode='lines', name='κ(G)'))
fig.add_hline(y=1e4, line_dash="dash", line_color="red",
              annotation_text="Alert Threshold")
fig.update_layout(xaxis_title="Write #", yaxis_title="κ(G)",
                  yaxis_type="log", height=300)
st.plotly_chart(fig, use_container_width=True)
```

## Step 3: Read Test Panel

```python
st.subheader("Read Test")
st.write("Send a random query through the operator and see the output.")

if st.button("Send Random Query"):
    query = np.random.randn(1, rank) * 0.1
    for K in [1, 2, 3, 5]:
        mem_test = SharedSpectralMemory(rank=rank, power_K=K)
        # Copy operator state
        mem_test._operator = mem.operator
        mem_test._stale = False
        try:
            output = mem_test.read(query)
            st.write(f"**K={K}:** ||output|| = {np.linalg.norm(output):.6f}")
        except Exception as e:
            st.error(f"K={K}: {e}")
```
