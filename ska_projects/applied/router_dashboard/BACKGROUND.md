# Background: The SKA Router (No Math Required)

This document explains what the router does at a conceptual level.
You do NOT need to understand Koopman operators or spectral theory.

## 1. What the Router Does

The router answers one question: **given a user's query, which
specialists should handle it, and in what order?**

It has 4 specialists available:
- **Parser**: Extracts structure from documents
- **SKA Retriever**: Finds relevant segments from a document corpus
- **Code Executor**: Runs Python for numerical computation
- **Reasoner**: Multi-step reasoning and answer synthesis

## 2. The Four Modes

The router first classifies the query into one of four modes, each
with a predefined workflow:

**LOOKUP** (simple factual):
"What was the total federal debt in 2023?"
→ Parse → Retrieve → Extract

**MULTI_DOC** (comparison):
"How did healthcare spending change from 2022 to 2023?"
→ Parse → Retrieve A → Retrieve B → Compare

**COMPUTE** (calculation needed):
"What percentage of GDP was military spending?"
→ Parse → Retrieve → Extract → Code → Answer

**MULTI_STEP** (complex):
"Which department had the highest year-over-year growth after inflation?"
→ Decompose → [Retrieve/Code]* → Synthesize

## 3. How It Decides

At each step, the router:
1. Lists all possible next actions (based on the mode's DAG template)
2. Scores each action: **S(a) = predicted_quality_gain - cost_penalty**
3. Executes the highest-scoring action if S > 0
4. Stops if no action has positive score

The **cost penalty** is controlled by a PID controller that learns
from past queries. If the system has been spending too much (tokens,
latency, dollars), the PID raises the cost penalty, making the router
more conservative. If it's under budget, penalties decrease.

## 4. The 5-Dimensional Cost Vector

Every action has a cost with 5 components:
1. Input tokens consumed
2. Output tokens generated
3. Latency (milliseconds)
4. Dollar cost
5. Meta overhead (shared memory operations)

The PID controller maintains a separate "price" (λ) for each dimension.
When the price for latency is high, the router avoids slow specialists.

## 5. What You're Visualizing

Your dashboard shows:
- **Mode probabilities**: The 4 softmax outputs from the mode selector
- **Action scores**: For each candidate action, the score breakdown
- **PID trajectory**: How the 5 λ values evolve over queries
- **Execution trace**: The sequence of actions actually taken
- **DAG view**: The workflow template with execution status
