import os

from spirl.utils.general_utils import AttrDict
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.policies.mlp_policies import ConvPolicy, MLPPolicy
from spirl.rl.components.critic import ConvCritic, MLPCritic
from spirl.rl.components.replay_buffer import ImageUniformReplayBuffer, UniformReplayBuffer
from spirl.rl.envs.kitchen import KitchenEnv, PrimitivesEnv
from spirl.rl.components.normalization import Normalizer
from spirl.configs.default_data_configs.kitchen import data_spec


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'non-hierarchical RL experiments in kitchen env'

configuration = {
    'seed': 42,
    'agent': SACAgent,
    'environment': PrimitivesEnv,
    'data_dir': '.',
    'num_epochs': 1000,
    'max_rollout_len': 280,
    'n_steps_per_epoch': 1e3,
    'n_warmup_steps': 2.5e3,
}
configuration = AttrDict(configuration)
data_spec.n_actions = 26
# Policy
policy_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=0,
    nz_mid=256,
    max_action_range=1.,
    unused_obs_size=0,
    input_res=data_spec.res,
    input_nc=3,
    ngf=8,
    nz_enc=128,
)

# Critic
critic_params = AttrDict(
    action_dim=policy_params.action_dim,
    input_dim=0,
    output_dim=1,
    action_input=True,
    unused_obs_size=0,
    input_res=data_spec.res,
    input_nc=3,
    ngf=8,
    nz_enc=128,
)

# Replay Buffer
replay_params = AttrDict(
    capacity=2.5e6,
    dump_replay=False,
)

# Observation Normalization
obs_norm_params = AttrDict(
)

# Agent
agent_config = AttrDict(
    policy=ConvPolicy,
    policy_params=policy_params,
    critic=ConvCritic,
    critic_params=critic_params,
    replay=ImageUniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
    batch_size=256,
    log_video_caption=True,
)

# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
)

