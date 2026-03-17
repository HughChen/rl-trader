# Experiment 009: Turnover Penalty & Notional Sweep

## Setup

Tested higher turnover penalty and smaller notional to reduce churning and slippage.

| Config | turnover_penalty | notional | slippage_sigma | Frictionless | Full costs |
|--------|------------------|----------|----------------|--------------|------------|
| 1 | 0.15 | 1e6 | 0.05 | +15.6% | -100% |
| 2 | 0.25 | 1e6 | 0.05 | +3.6% | -100% |
| 3 | 0.25 | 100k | 0.05 | +6.7% | -100% |
| 4 | 0.5 | 50k | 0.05 | +21.8% | -100% |
| 5 | 1.0 | 50k | 0.05 | +37.7% | -99.99% |
| 6 | 2.0 | 25k | 0.02 | +46.2% | -99.95% |
| 7 | 3.0 | 10k | 0.01 | +47.4% | -99.80% |
| 8 | 10.0 | 10k | 0.01 | **+49.8%** | -99.78% |

---

## Findings

- **Higher turnover penalty** → lower frictionless return (agent trades less) but still blows up with costs
- **Smaller notional + lower sigma** → less slippage, but fees still dominate (cum_fee 545%+)
- **Best frictionless**: +49.8% (turnover=10, notional=10k, sigma=0.01)
- **No config beat baseline** (-32.75%) with full costs; agent still churns too much

### Cost breakdown (config 8)

- Frictionless: +49.8%
- Fees only: -99.36% (cum_fee=545%)
- Slippage only: -87.3%
- Full: -99.78%

---

## Conclusion

Turnover penalty and notional tweaks improve frictionless performance but do not stop blow-up. Fees from constant rebalancing remain the main cost. Possible next steps: rebalancing frequency constraint (e.g. every N steps), or PGPortfolio-style architecture.
