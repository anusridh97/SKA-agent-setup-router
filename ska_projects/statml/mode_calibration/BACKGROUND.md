# Background: Neural Network Calibration

## 1. Calibration Defined

A probabilistic classifier outputs p̂(y|x). It is calibrated if:

    P(Y = y | p̂(Y = y | X) = p) = p  for all p ∈ [0, 1]

Modern neural networks are poorly calibrated: they tend to be
overconfident (predict p̂ = 0.95 when true probability is 0.7).

## 2. Expected Calibration Error (ECE)

Partition predictions into M equal-width bins by confidence.

    ECE = Σ_{m=1}^{M} (n_m / N) |acc_m - conf_m|

where acc_m = accuracy in bin m, conf_m = average confidence in bin m.

## 3. Temperature Scaling (Guo et al. 2017)

After training, learn a single scalar T > 0:

    p̂_cal(y | x) = softmax(z(x) / T)

where z(x) are the logits. Fit T by minimizing NLL on validation data.

Simple, effective, preserves top-1 predictions.

## 4. Platt Scaling

Learn affine parameters: p̂_cal = softmax(a · z + b)

More expressive, but can overfit on small calibration sets.

## 5. Reliability Diagrams

Plot bin accuracy vs bin confidence. Perfectly calibrated = diagonal.
Below diagonal = overconfident. Above = underconfident.
Include bar chart showing number of samples per bin (shows coverage).

## 6. References

1. Guo, Pleiss, Sun, Weinberger. "On Calibration of Modern Neural Networks."
   ICML 2017.
2. Niculescu-Mizil & Caruana. "Predicting Good Probabilities with Supervised
   Learning." ICML 2005.
