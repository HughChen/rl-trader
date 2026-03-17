# Experiments

Each experiment has a separate report with **setup** and **findings**.

| Experiment | Description |
|------------|-------------|
| [exp_001_baseline](exp_001_baseline.md) | Equal-weight buy-and-hold baseline |
| [exp_002_ppo_initial](exp_002_ppo_initial.md) | PPO agent, initial runs (no turnover penalty) |
| [exp_003_turnover_penalty](exp_003_turnover_penalty.md) | Effect of turnover penalty on validation |
| [exp_004_hyperparam_tuning](exp_004_hyperparam_tuning.md) | Grid search on validation set |
| [exp_005_exchange_restrictions](exp_005_exchange_restrictions.md) | CCXT exchange data limits |

---

## Common Data Config

Shared across experiments (unless noted):

- **Exchange**: KuCoin
- **Symbols**: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT (USDT)
- **Frequency**: 1h bars
- **Splits**: Train to 2024-12-31, Val to 2025-06-30, Test after

---

## Lessons Learned (cross-cutting)

1. **Agent blow-ups**: Policy can allocate 100% to one asset and blow up. Action constraints (max weight per asset) recommended.
2. **Turnover penalty helps**: Reduces churning; improves validation metrics.
3. **Regime mismatch**: Train (2024) mostly bull; test (2025–2026) bear. Agent may not generalize.
4. **Evaluation rigor**: Single train/val/test split is fragile. Walk-forward testing recommended.
5. **Baseline context**: Negative baseline on test = bear market. Beating baseline means losing less than -33%.

Planned improvements: see [PLANNING.md §13](../PLANNING.md).
