# Experiment 003: Turnover Penalty

## Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | PPO |
| **Data** | Train split |
| **Varied** | `turnover_penalty` ∈ {0, 0.01, 0.05} |
| **Timesteps per trial** | 40k |
| **Evaluation** | Validation set, 10 episodes |

Reward: `(log_return - turnover_penalty × Σ|Δw|) × reward_scale`

---

## Findings

| turnover_penalty | Val Sharpe | Val Return |
|-----------------|------------|------------|
| 0.0 | -14.48 | -99% |
| 0.01 | -13.28 | -98% |
| 0.05 | -12.43 | -98% |

**Observation**: Higher turnover penalty improves validation Sharpe (less negative). Encourages holding; reduces churning. Best: `turnover_penalty=0.05`.

Training with best config for 200k steps still yields -100% on test — regime shift or overfitting.
