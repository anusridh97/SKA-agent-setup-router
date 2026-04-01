# Background: Human Feedback for Retrieval Systems

## 1. Explicit vs Implicit Feedback

**Explicit:** User clicks thumbs up/down, gives star rating.
Unambiguous but requires effort, users may not bother.

**Implicit:** User behavior (clicked on a segment, spent time reading,
reformulated query). Rich signal but noisy.

For this project, focus on explicit feedback. Implicit is a stretch goal.

## 2. Feedback Granularity

Three levels of feedback, each more useful:

1. **Answer-level:** "Was the final answer correct?" (easiest to collect,
   least diagnostic)
2. **Segment-level:** "Was this retrieved segment relevant?" (more effort,
   but tells you if retrieval or generation failed)
3. **Comparative:** "Was answer A or answer B better?" (most useful for
   reward modeling, but requires showing two answers)

This project focuses on levels 1 and 2.

## 3. From Ratings to Training Signal

The reward predictor expects a quality "delta", how much better is
specialist A than specialist B on this query? To derive this from ratings:

- If the same query is run twice (e.g. with different λ), and one gets
  4 stars while the other gets 2, delta_r = (4-2)/5 = 0.4
- If only one run exists, compare against a default baseline score
  (e.g. 3 stars = neutral, so 5 stars → delta_r = +0.4, 1 star → -0.4)

## 4. References

1. Ouyang et al. "Training Language Models to Follow Instructions with
   Human Feedback." NeurIPS 2022. (RLHF at scale)
2. Stiennon et al. "Learning to Summarize from Human Feedback." NeurIPS 2020.
