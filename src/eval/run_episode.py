"""Run agents through the environment and collect results."""

import numpy as np

from .metrics import compute_metrics, equity_curve


def run_episode(
    env,
    get_action,
    seed: int | None = None,
    *,
    return_actions: bool = False,
) -> tuple[np.ndarray, float, dict] | tuple[np.ndarray, float, dict, list[np.ndarray], int]:
    """
    Run one episode, collecting rewards.

    Args:
        env: PortfolioEnv instance
        get_action: callable(obs, info) -> action array
        seed: Optional random seed for env.reset
        return_actions: If True, also return (actions, start_t) for cost replay

    Returns:
        (rewards, final_value, info_dict) or
        (rewards, final_value, info_dict, actions, start_t) if return_actions
    """
    obs, info = env.reset(seed=seed)
    start_t = info.get("t", 0)
    rewards = []
    actions_list = [] if return_actions else None
    done = False
    while not done:
        action = get_action(obs, info)
        if return_actions:
            actions_list.append(np.asarray(action, dtype=np.float64))
        obs, reward, done, truncated, info = env.step(action)
        # Use raw_reward for metrics when available (unscaled log return)
        rewards.append(info.get("raw_reward", reward))
        done = done or truncated

    rewards = np.array(rewards)
    result = (rewards, info.get("portfolio_value", np.exp(np.sum(rewards))), info)
    if return_actions:
        return (*result, actions_list, start_t)
    return result


def _prev_weights_from_obs(obs, n_assets: int) -> np.ndarray:
    """Extract previous weights from observation (Dict or flat)."""
    if isinstance(obs, dict):
        return np.asarray(obs["prev_w"], dtype=np.float32)
    return np.asarray(obs[-n_assets:], dtype=np.float32)


def run_equal_weight(env, seed: int | None = None) -> tuple[np.ndarray, float, dict]:
    """Run equal-weight rebalancing baseline (rebalances to 1/n every step, has friction)."""
    n_assets = env.n_assets
    action = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

    def get_action(obs, info):
        return action

    return run_episode(env, get_action, seed)


def run_equal_weight_buy_and_hold(env, seed: int | None = None) -> tuple[np.ndarray, float, dict]:
    """Run equal-weight buy-and-hold baseline (no rebalancing, zero friction)."""
    def get_action(obs, info):
        return None  # Hold: env uses drifted weights as target (zero turnover)

    return run_episode(env, get_action, seed)


def run_episodes(
    env,
    get_action,
    n_episodes: int = 10,
    seed: int | None = None,
) -> tuple[list[np.ndarray], list[float], dict]:
    """
    Run multiple episodes (for env with episode_length set).

    Returns:
        (list of reward arrays, list of final values, aggregate metrics)
    """
    rng = np.random.default_rng(seed)
    all_rewards = []
    all_final_values = []

    for i in range(n_episodes):
        rewards, final_val, _ = run_episode(env, get_action, seed=int(rng.integers(0, 2**31)))
        all_rewards.append(rewards)
        all_final_values.append(final_val)

    # Aggregate: concatenate all rewards for overall metrics
    concat_rewards = np.concatenate(all_rewards)
    metrics = compute_metrics(concat_rewards)

    return all_rewards, all_final_values, metrics
