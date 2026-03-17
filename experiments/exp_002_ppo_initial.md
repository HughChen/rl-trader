# Experiment 002: PPO Agent (Initial)

## Setup

| Parameter | Value |
|-----------|-------|
| **Algorithm** | PPO (Stable-Baselines3) |
| **Data** | KuCoin, 10 symbols, 1h bars, train split |
| **Episode length** | 252 |
| **Total timesteps** | 50k–200k |
| **turnover_penalty** | 0 |
| **reward_scale** | 10 |
| **slippage_sigma** | 0.05 |
| **Network** | [256, 256] |
| **Learning rate** | 1e-4 |
| **Command** | `python scripts/train_agent.py --split train --total-timesteps 200000` |

---

## Findings

| Timesteps | Test Sharpe | Test Return |
|-----------|-------------|-------------|
| 50k | -12.8 | -100% |
| 200k | -12.7 | -100% |

**Observation**: Agent consistently underperforms baseline. Collapses to -100% (blow-up). Likely causes: over-concentration, extreme rebalancing, regime mismatch (train bull, test bear).
