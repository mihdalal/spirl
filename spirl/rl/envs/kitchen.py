import numpy as np
from collections import defaultdict
import d4rl
import rlkit.envs.primitives_make_env as primitives_make_env
from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv


class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-mixed-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info.pop("completed_tasks")
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info


class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]

class PrimitivesEnv(GymEnv):
    def __init__(self, config):
        self._hp = self._default_hparams().overwrite(config)
        self._env = self._make_env(self._hp)

        from mujoco_py.builder import MujocoException
        self._mj_except = MujocoException

    def _make_env(self, hp):
        """Instantiates the environment given the ID."""
        env = primitives_make_env.make_env(self._hp.env_suite, self._hp.env_name, self._hp.env_kwargs)
        return env

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        return info

    def render(self, mode='rgb_array'):
        # TODO make env render in the correct size instead of downsizing after for performance
        img = self._render_raw(mode=mode)
        return np.array(img) / 255.

    def render_raw(self, mode='rgb_array'):
        img = self._render_raw(mode=mode, imwidth=self._hp.screen_width, imheight=self._hp.screen_height)
        return img
