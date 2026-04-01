# Evaluation: Mode Calibration

## Grading

| Component | Weight |
|-----------|--------|
| Calibration dataset (200+ labeled queries) | 15% |
| ECE/MCE computed, reliability diagrams plotted | 25% |
| Temperature scaling implemented and evaluated | 20% |
| 1+ additional method (Platt, histogram, isotonic) | 20% |
| Before/after comparison with clear conclusions | 20% |

### A-level
- Well-designed calibration dataset covering all 4 modes
- Clear reliability diagrams showing miscalibration
- Temperature scaling and Platt scaling both implemented
- ECE reduction quantified (e.g. "ECE reduced from 0.15 to 0.03")
- Discussion of downstream routing impact

### B-level
- Basic dataset. Temperature scaling works. Some ECE improvement.

### C-level
- Incomplete dataset. Only one calibration method. Unclear results.
