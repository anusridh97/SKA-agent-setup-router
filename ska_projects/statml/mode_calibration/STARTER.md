# Starter: Mode Calibration

The SPEC.md contains the dataset generation and ECE computation code.
Here's the temperature scaling implementation:

```python
import numpy as np
from scipy.optimize import minimize_scalar

def temperature_scale(logits, labels, n_classes=4):
    """
    Find optimal temperature T by minimizing NLL on calibration set.

    logits: (N, n_classes) raw logits
    labels: (N,) integer class labels
    """
    def nll(T):
        scaled = logits / T
        # Stable softmax
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        exp_s = np.exp(shifted)
        probs = exp_s / exp_s.sum(axis=1, keepdims=True)
        # NLL
        log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-15)
        return -log_probs.mean()

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x

def apply_temperature(logits, T):
    scaled = logits / T
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(shifted)
    return exp_s / exp_s.sum(axis=1, keepdims=True)

# Usage:
# T_opt = temperature_scale(val_logits, val_labels)
# calibrated_probs = apply_temperature(test_logits, T_opt)
```

## Reliability Diagram Plotting

```python
import matplotlib.pyplot as plt

def plot_reliability_diagram(probs, labels, n_bins=10, title=""):
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i+1])
        if mask.sum() > 0:
            bin_accs.append(accuracies[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_counts.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8), height_ratios=[3, 1])

    ax1.bar(bin_confs, bin_accs, width=1/n_bins, alpha=0.7, edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.legend()

    ax2.bar(bin_confs, bin_counts, width=1/n_bins, alpha=0.7, color='gray')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')

    plt.tight_layout()
    return fig
```
