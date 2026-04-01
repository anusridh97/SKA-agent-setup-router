# Starter: Retrieval Feedback

## Step 1: Feedback UI Component

```python
import streamlit as st
import json
from datetime import datetime

def segment_feedback(segments, reduced_costs):
    """Display segments with feedback buttons."""
    ratings = {}
    for i, (seg, rc) in enumerate(zip(segments, reduced_costs)):
        col1, col2, col3 = st.columns([6, 1, 1])
        with col1:
            st.write(seg.text[:200] + ("..." if len(seg.text) > 200 else ""))
            st.caption(f"Reduced cost: {rc:.4f}")
        with col2:
            if st.button("👍", key=f"up_{i}"):
                ratings[i] = 1
        with col3:
            if st.button("👎", key=f"down_{i}"):
                ratings[i] = -1
    return ratings

def answer_feedback():
    """Star rating for the overall answer."""
    return st.slider("Rate the answer quality:", 1, 5, 3,
                     help="1 = wrong, 3 = partial, 5 = perfect")
```

## Step 2: Feedback Storage

```python
import sqlite3

def init_db(path="feedback.db"):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            query_embedding BLOB,
            mode_selected TEXT,
            lambda_used REAL,
            num_segments INTEGER,
            segment_ratings TEXT,  -- JSON: {idx: rating}
            answer_rating INTEGER,
            answer_text TEXT
        )
    """)
    conn.commit()
    return conn

def save_feedback(conn, query, q_emb, mode, lam, n_segs,
                  seg_ratings, answer_rating, answer):
    import numpy as np
    conn.execute(
        "INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), query, q_emb.tobytes(), mode,
         lam, n_segs, json.dumps(seg_ratings), answer_rating, answer)
    )
    conn.commit()
```

## Step 3: Generate Training Data

```python
def generate_reward_training_data(db_path="feedback.db"):
    """Convert feedback into RewardPredictor format."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM feedback").fetchall()

    training_data = []
    for row in rows:
        q_emb = np.frombuffer(row[3], dtype=np.float64)
        answer_rating = row[7]  # 1-5

        # Normalize to [-1, 1]
        delta_r = (answer_rating - 3) / 2.0

        training_data.append({
            'query_embedding': q_emb,
            'model_idx': 0,      # which specialist (map from mode)
            'base_model_idx': 0, # baseline
            'delta_r': delta_r,
        })

    return training_data
```
