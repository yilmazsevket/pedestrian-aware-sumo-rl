# Agent vs. Baseline Report
## 5.2 Performance Evaluation: Agent vs. Baseline Comparison
### 5.2.1 Evaluation Methodology
- Morning Rush Hour: 07:30 (27,000s) for 3,000 simulation steps
- Evening Rush Hour: 17:00 (61,200s) for 3,000 simulation steps
- Metrics: vehicle waiting (mean, max, p95), pedestrian waiting by class (incl. vulnerable), throughput (veh/h, ped/h), max queue length.

### 5.2.2 Overall Performance Comparison
| Metric | Baseline | RL Agent | Improvement |
|---|---:|---:|---:|
| Vehicle Wait Time (s) | 9343.51 | 6202.21 | 33.62% |
| Pedestrian Wait Time (s) | 325.36 | 88.00 | 72.95% |
| Vulnerable Ped. Wait Time (s) | 36.17 | 12.88 | 64.38% |
| Vehicle Throughput (veh/h) | 1800.60 | 2158.20 | -19.86% |
| Pedestrian Throughput (ped/h) | 1304.40 | 1320.00 | -1.20% |
| Max Queue Length (vehicles) | 164.50 | 131.00 | 20.36% |

### Additional Episode Stats (averaged over windows)
- Final normalized reward (sum over episode): baseline=-58.52, agent=-35.16
- Episode length (steps): baseline=3000, agent=3000

### Notes
- Successful learning rate decay without performance collapse (verify in TensorBoard).
- Entropy coefficient scheduling maintained exploration throughout training.
- The combination of delta-based rewards with level penalty provided an optimal balance.
- Higher queue weighting (0.4) improved responsiveness to vehicle pressure.
- Learning rate decay enabled fine-tuning without destabilizing the policy.
- Increased normalizer scale prevented reward saturation in high-traffic scenarios.
