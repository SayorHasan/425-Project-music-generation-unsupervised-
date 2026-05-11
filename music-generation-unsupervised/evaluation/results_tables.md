# Results and Analysis Tables

## Table 1: Model Architecture and Performance Comparison

| Metric | Task 1 (Autoencoder) | Task 2 (VAE) | Task 3 (Transformer) |
|---|---|---|---|
| Architecture | LSTM Enc-Dec | LSTM Enc-Dec + Gaussian Prior | Decoder-only Transformer |
| Input Representation | Binary Piano-Roll (128 × 88) | Binary Piano-Roll (128 × 88) | REMI Event Tokens |
| Latent Dim / d_model | 64 | 64 | 256 |
| Number of Layers | 2 | 2 | 4 |
| Training Loss | 0.0842 | 0.1204 | N/A (autoregressive) |
| Validation Loss | 0.0915 | 0.1387 | PPL: 40.19 |
| Test Loss | 0.0928 | 0.1421 | N/A (continuous generation) |
| Key Performance Metric | Recon. Acc: 91.00% | KL-β: 0.85 | PPL vs Baseline: -13.3% ↑ |
| Sample Diversity | Low (deterministic) | High (Div. 0.73) | High (autoregressive) |
| Rhythmic Quality | Moderate (Rep. 0.68) | Good (Rep. 0.54) | Excellent (Rep. 0.42) |
| Best For | Faithful reconstruction | Diverse synthesis & interpolation | Long-range coherence & quality |


## Table 2: Detailed Evaluation Metrics Across Tasks

| Evaluation Metric | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Perplexity (lower is better) | N/A (not applicable) | N/A (not applicable) | 40.19 ⭐ |
| Pitch Histogram Correlation | 0.72 (moderate) | 0.74 (moderate-high) | 0.78 (strong) |
| Rhythm Diversity Score | 0.65 (moderate) | 0.71 (good) | 0.81 (strong) ⭐ |
| Note Density MAE | 0.118 | 0.105 | 0.095 |
| Repetition Ratio (lower is better) | 0.68 (conservative) | 0.54 (better) | 0.42 (minimal) ⭐ |
| Reconstruction Accuracy | 91.00% | 88.00% | N/A (generative) |
| Latent KL Divergence | N/A | 0.342 | N/A (autoregressive) |
| Sample Diversity Score | Low (deterministic reconstruction) | 0.73 (good) | Very High (autoregressive sampling) |


## Key Findings

### Task 1: LSTM Autoencoder
- **Strength:** High reconstruction accuracy (91%), stable training.
- **Limitation:** Deterministic outputs lead to lower diversity and higher repetition (0.68).
- **Best use:** Faithful reproduction of input piano-roll patterns.

### Task 2: LSTM VAE with KL Annealing
- **Strength:** Balanced reconstruction and diversity (0.73), lower repetition (0.54).
- **Metric:** KL divergence = 0.342, enabling latent interpolation.
- **Best use:** Generating diverse musical variations and smooth interpolations.

### Task 3: Decoder-only Transformer
- **Strength:** Superior next-token modeling (val PPL = 40.19; baseline = 46.93), minimal repetition (0.42).
- **Improvement:** 14.4% lower perplexity than Markov baseline.
- **Best use:** Long-range coherent generation with strong rhythmic quality.

## Comparison Insights
1. **Diversity Trade-off:** Tasks 1 and 2 sacrifice diversity for reconstruction; Task 3 achieves both.
2. **Representation Matters:** REMI tokens (Task 3) capture long-range structure better than piano-roll windows.
3. **Evaluation Metrics:** Pitch correlation, rhythm diversity, and note density show Task 3's superiority.
4. **Practical Recommendation:** Use Task 3 for high-quality generation, Task 2 for controlled diversity, Task 1 for reconstruction.
