# Experiment 010: Full Episodes (Short-Horizon Hypothesis Test)

## Hypothesis

Agent trained on 100-step episodes optimizes for short horizons; costs compound over 6000 steps at eval. Training with full episodes should teach long-horizon cost of churning.

## Setup

| Config | episode_length | turnover | notional | sigma | timesteps |
|--------|----------------|----------|----------|-------|-----------|
| Baseline (exp_009) | 100 | 10 | 10k | 0.01 | 200k |
| Full | 0 (full) | 10 | 10k | 0.01 | 200k |
| Medium | 500 | 10 | 10k | 0.01 | 200k |
| Full + more | 0 | 5 | 10k | 0.01 | 400k |

---

## Findings

| Config | Frictionless | Cum fee | Cum slip | Full costs |
|--------|--------------|---------|----------|------------|
| 100-step (baseline) | **+49.8%** | 545% | 108% | -99.78% |
| Full (200k) | -20.3% | 383% | 105% | -99.40% |
| 500-step | -29.9% | 444% | 97% | -99.69% |
| Full 400k, turn=5 | -23.6% | **328%** | 102% | -98.96% |

### Key observations

1. **Full episodes reduce churning**: Cum_fee drops from 545% (100-step) to 328–383% (full) — agent trades ~25–40% less.
2. **Allocation quality worsens**: Frictionless return goes from +50% (100-step) to -20% to -24% (full). The full-episode agent holds more but makes worse allocation decisions.
3. **Trade-off**: Short episodes → good allocation, excessive churning. Full episodes → less churning, poor allocation.
4. **No config beats baseline** (-32.75%) with full costs.

---

## Conclusion

**Hypothesis partially supported**: Full episodes do reduce churning. But they also hurt allocation quality — the agent may over-converge to "hold" with suboptimal weights, or long episodes worsen credit assignment for allocation learning.

**Refined hypothesis**: Need both (a) diverse short windows for allocation learning and (b) long-horizon cost signal. Possible next steps: curriculum (short → long), or hybrid reward (per-step + terminal).
