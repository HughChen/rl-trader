# Experiment 011: High-Impact Changes (Full Costs, Rebalancing Constraint, Turnover Cap, Terminal Reward)

## Setup

Implements the four high-priority next steps:

| Change | Implementation |
|--------|----------------|
| **Train with full costs** | `cost_model="full"` (fee + slippage) — match eval, avoid train/eval mismatch |
| **Min rebalance interval** | `min_rebalance_interval=6` — force hold for 6 steps between rebalances |
| **Max turnover per step** | `max_turnover_per_step=0.2` — cap turnover at 20% per step |
| **Terminal reward** | `terminal_reward_scale=1.0` — Sharpe-like bonus at episode end (long-horizon cost signal) |

### Commands

```bash
python scripts/train_and_evaluate.py --total-timesteps 200000 --cost-analysis --lr-decay
```

To disable high-impact changes: `--no-high-impact`

---

## Findings (eval on saved model)

| Metric | Rebalancing | Buy-and-hold | PPO Agent |
|--------|-------------|--------------|-----------|
| Val Sharpe | -3.50 | -0.47 | -5.04 |
| Val Total Ret | -87.65% | -33.58% | -100% |
| Test Sharpe | -3.33 | -0.44 | -4.50 |
| Test Total Ret | -81.99% | -29.41% | -100% |

### Cost attribution (test, agent)

| Config | Total Return | Cum Fee | Cum Slippage |
|--------|--------------|---------|--------------|
| Frictionless | -39.66% | — | — |
| Fees only | -41.81% | 3.62% | — |
| Slippage only | -92.75% | — | 211.78% |
| Full costs | -92.91% | 3.62% | 210.51% |

- **Fee impact**: 3.6% of frictionless value
- **Slippage impact**: 88% of frictionless value

---

## Observations

1. **Agent still blows up** (-100%) with full costs; constraints did not prevent collapse.
2. **Frictionless return negative** (-39.66%) — allocation quality worse than prior runs (+57% frictionless). Constraints may have pushed agent toward holding with suboptimal weights.
3. **Cum fee reduced** (3.62% vs 545% in exp_007) — min_rebalance_interval and max_turnover are cutting turnover.
4. **Slippage still dominant** (211% cum) — even with caps, agent churns enough to destroy value.
5. **No config beats baseline** (-29% to -34% buy-and-hold).

---

## Conclusion

High-impact changes reduce turnover (fee 3.6% vs 545%) but agent still collapses. Allocation quality degraded (frictionless -40% vs +57%). Possible next steps: stronger constraints (e.g. min_rebalance_interval=12, max_turnover=0.1), curriculum (short → long episodes), or regime-aware training.

See [exp_012_next_steps](exp_012_next_steps.md) for stronger constraints + curriculum + regime.
