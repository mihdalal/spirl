from spirl.rl.policies.cl_model_policies import ACClModelPolicy
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
import os
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl
from spirl.rl.policies.mlp_policies import SplitObsMLPPolicy
from spirl.rl.components.critic import SplitObsConvCritic, SplitObsMLPCritic
from spirl.rl.components.sampler import MultiImageAugmentedHierarchicalSampler
from spirl.configs.default_data_configs.kitchen_vision import data_spec
import os
import copy
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.utils.general_utils import AttrDict
from spirl.rl.components.agent import FixedIntervalHierarchicalAgent
from spirl.rl.envs.kitchen import KitchenEnv
from spirl.rl.components.replay_buffer import UniformReplayBuffer
from spirl.rl.agents.ac_agent import SACAgent
from spirl.rl.agents.skill_space_agent import ACSkillSpaceAgent


current_dir = os.path.dirname(os.path.realpath(__file__))

notes = 'hierarchical RL on the kitchen vision env'

configuration = {
    'seed': 42,
    'agent': FixedIntervalHierarchicalAgent,
    'environment': KitchenEnv,
    'sampler': MultiImageAugmentedHierarchicalSampler,
    'data_dir': '.',
    'num_epochs': 15,
    'max_rollout_len': 280,
    'n_steps_per_epoch': 1e5,
    'n_warmup_steps': 5e3,
}
configuration = AttrDict(configuration)


# Replay Buffer
replay_params = AttrDict(
)

# Observation Normalization
obs_norm_params = AttrDict(
)

sampler_config = AttrDict(
    n_frames=2,
)

base_agent_params = AttrDict(
    batch_size=256,
    replay=UniformReplayBuffer,
    replay_params=replay_params,
    clip_q_target=False,
)


###### Low-Level ######
# LL Policy
ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=1e-2,
    n_input_frames=2, #2
    prior_input_res=data_spec.res,
    nz_vae=10,
    n_rollout_steps=10,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# LL Policy
ll_policy_params = AttrDict(
    policy_model=ImageClSPiRLMdl,
    policy_model_params=ll_model_params,
    policy_model_checkpoint=os.path.join(os.environ["EXP_DIR"], "skill_prior_learning/kitchen/vision_cl"),
    # initial_log_sigma=-50.,
)
ll_policy_params.update(ll_model_params)

# LL Critic
ll_critic_params = AttrDict(
    action_dim=data_spec.n_actions,
    input_dim=data_spec.state_dim,
    output_dim=1,
    action_input=True,
    unused_obs_size=10,     # ignore HL policy z output in observation for LL critic
)

# LL Agent
ll_agent_config = copy.deepcopy(base_agent_params)
ll_agent_config.update(AttrDict(
    policy=ACClModelPolicy,
    policy_params=ll_policy_params,
    critic=SplitObsConvCritic,
    critic_params=ll_critic_params,
    model=ImageClSPiRLMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/kitchen/vision_cl"),
))


###### High-Level ########
# HL Policy
hl_policy_params = AttrDict(
    action_dim=10,       # z-dimension of the skill VAE
    max_action_range=2.,        # prior is Gaussian with unit variance
    unused_obs_size=60,
    discard_part='front',
    prior_model=ll_policy_params.policy_model,
    prior_model_params=ll_policy_params.policy_model_params,
    prior_model_checkpoint=ll_policy_params.policy_model_checkpoint,
)

# HL Critic
hl_critic_params = AttrDict(
    action_dim=hl_policy_params.action_dim,
    input_dim=0,
    output_dim=1,
    action_input=True,
    unused_obs_size=hl_policy_params.unused_obs_size,
    discard_part=hl_policy_params.discard_part,
    input_res=data_spec.res,
    input_nc=3*2,
    ngf=8,
    nz_enc=128,
)

# HL Agent
hl_agent_config = copy.deepcopy(base_agent_params)
hl_agent_config.update(AttrDict(
    policy=ACLearnedPriorAugmentedPIPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsConvCritic,
    critic_params=hl_critic_params,
))


##### Joint Agent #######
agent_config = AttrDict(
    hl_agent=ActionPriorSACAgent,
    hl_agent_params=hl_agent_config,
    ll_agent=SACAgent,
    ll_agent_params=ll_agent_config,
    hl_interval=ll_model_params.n_rollout_steps,
    log_videos=False,
    update_hl=True,
    update_ll=False,
)

# update agent, set target divergence
agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=5.), #was originally 5
))


# Dataset - Random data
data_config = AttrDict()
data_config.dataset_spec = data_spec

# Environment
env_config = AttrDict(
    reward_norm=1.,
    screen_height=ll_model_params.prior_input_res,
    screen_width=ll_model_params.prior_input_res,
)

from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res**2*3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim   # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False
