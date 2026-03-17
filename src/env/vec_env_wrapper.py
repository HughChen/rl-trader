"""Wrapper to use VecEnv as a single env for evaluation."""

import numpy as np


class VecEnvToGymWrapper:
    """Use a VecEnv (e.g. VecNormalize with n_envs=1) as a single Gymnasium env."""

    def __init__(self, vec_env):
        assert vec_env.num_envs == 1, "Only supports n_envs=1"
        self.vec_env = vec_env

    def _get_underlying_env(self):
        venv = getattr(self.vec_env, "venv", self.vec_env)
        return venv.envs[0]

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._get_underlying_env()._rng = np.random.default_rng(seed)
        result = self.vec_env.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs, info = result, {}
        info_dict = {k: v[0] if hasattr(v, "__len__") and len(v) == 1 else v for k, v in info.items()} if info else {}
        return obs[0], info_dict

    def step(self, action):
        result = self.vec_env.step([action])
        if len(result) == 5:
            obs, rewards, dones, truncateds, infos = result
            truncated = bool(truncateds[0])
        else:
            obs, rewards, dones, infos = result
            truncated = False
        info = infos[0] if isinstance(infos, list) and infos else {}
        return obs[0], float(rewards[0]), bool(dones[0]), truncated, info

    @property
    def n_assets(self):
        # VecNormalize wraps DummyVecEnv; get underlying env
        venv = getattr(self.vec_env, "venv", self.vec_env)
        return venv.envs[0].n_assets

    @property
    def action_space(self):
        return self.vec_env.action_space
