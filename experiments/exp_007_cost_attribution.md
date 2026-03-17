# Experiment 007: Cost Attribution (Fee vs Slippage)

> **Note**: Pre-exp_008 (slippage had units bug). See [exp_008](exp_008_slippage_notional.md) for corrected slippage.

## Setup

| Parameter | Value |
|-----------|-------|
| **Tool** | `python scripts/evaluate.py --cost-analysis` |
| **Method** | Replay agent trajectory with 4 cost configs |
| **Agent** | PPO (trained 200k steps, exp_006 config) |
| **Split** | Test (full period) |

Replay configs:
- **Frictionless**: fee_rate=0, slippage_sigma=0
- **Fees only**: fee_rate=0.001, slippage_sigma=0
- **Slippage only**: fee_rate=0, slippage_sigma=0.1
- **Full costs**: fee_rate=0.001, slippage_sigma=0.1

---

## Findings

| Config | Total Return | Cumulative Fee | Cumulative Slippage |
|--------|--------------|----------------|---------------------|
| Frictionless | **+57.07%** | — | — |
| Fees only | -3.52% | 664% | — |
| Slippage only | -99.80% | — | 49% |
| Full costs | -99.82% | 664% | 13% |

**Impact (value lost vs frictionless)**:
- Fee impact: 38.6% of frictionless value
- Slippage impact: 99.9% of frictionless value

### Key insight

**The agent would be +57% without costs.** Allocation decisions are profitable; the blow-up is entirely from transaction costs.

**Slippage dominates**: With slippage only (no fees), return is -99.8%. Slippage from large, frequent rebalancing destroys value. Fees contribute ~39% of the loss; slippage contributes ~100% (nearly all of it).

**Implication**: The agent churns with large trades. Square-root market impact (slippage) scales with trade size — big rebalances are extremely costly. Higher turnover penalty or position-size constraints may help more than fee tweaks.
