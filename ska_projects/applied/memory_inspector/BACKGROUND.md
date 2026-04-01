# Background: Shared Spectral Memory (Plain English)

## What It Is

Imagine a shared whiteboard between agents. One agent writes notes, another
reads them. But instead of text, the notes are encoded as a small matrix
(64×64). This is more efficient than passing text around, the matrix is
always the same size, regardless of how much information has been written.

## Key Concepts

**Write:** An agent produces a batch of "key vectors" (think: compressed
summaries of what it found). These get accumulated into the shared matrix.

**Read:** An agent sends a "query vector" (what am I looking for?). The
matrix transforms the query into a "retrieved value" (here's what the
other agent found that's relevant).

**Condition Number (κ):** A health metric. κ ≈ 1 is perfectly healthy.
κ > 10,000 means the matrix is numerically sick, reads may return garbage.
The codebase prints a warning when this happens but doesn't show it visually.

**Singular Values:** The operator matrix has 64 singular values (σ₁ ≥ σ₂ ≥
... ≥ σ₆₄). These tell you the "strength" of the matrix in different
directions. A healthy operator has a few strong directions and many weak
ones (rapid decay). A degenerate operator has all values clustered together
(no filtering ability).

## What Can Go Wrong

1. **Condition number blowup:** If agents write very similar keys, the
   matrix becomes ill-conditioned. Solution: the system adds ridge
   regularization (ε·I), but if the data is too correlated, even that
   isn't enough.

2. **Operator goes to zero:** If spectral normalization is too aggressive
   (γ << 1), all singular values get squashed and reads return near-zero.

3. **Slot interference:** With multi-head Koopman (4 parallel operators),
   different agents write to different "slots." If slots get mixed up,
   an agent may read another slot's stale data.
