# Experiment 004: Hyperparameter Tuning

## Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | PPO |
| **Data** | Train (train), Val (eval) |
| **Grid** | turnover_penalty × learning_rate × reward_scale |
| **Timesteps per trial** | 40k |
| **n_trials** | 6 |
| **Command** | `python scripts/tune_hyperparams.py --timesteps-per-trial 40000 --n-trials 6` |

Grid (prioritized):
- turnover_penalty: 0.05, 0.01, 0.0
- learning_rate: 1e-4, 3e-4
- reward_scale: 5.0, 10.0

---

## Findings

| Config | Val Sharpe |
|--------|------------|
| turnover=0.05, lr=1e-4, scale=10 | -12.43 (best) |
| turnover=0.05, lr=1e-4, scale=5 | -12.94 |
| turnover=0.01, lr=3e-4, scale=10 | -12.99 |
| turnover=0.01, lr=1e-4, scale=10 | -13.28 |
| turnover=0.05, lr=3e-4, scale=10 | -13.84 |
| turnover=0.0, lr=1e-4, scale=10 | -14.48 |

**Observation**: Turnover penalty is most impactful. Best config trained 200k steps → still -100% on test.
