from spirl.configs.hrl.kitchen.base_conf import *
from spirl.rl.policies.prior_policies import ACLearnedPriorAugmentedPIPolicy
from spirl.rl.agents.prior_sac_agent import ActionPriorSACAgent
import os
from spirl.models.skill_prior_mdl import ImageSkillPriorMdl

hl_agent_config.update(AttrDict(
    policy=SplitObsMLPPolicy,
    policy_params=hl_policy_params,
    critic=SplitObsMLPCritic,
    critic_params=hl_critic_params,
))
# update policy to use prior model for computing divergence
hl_policy_params.update(AttrDict(
    prior_model=ll_agent_config.model,
    prior_model_params=ll_agent_config.model_params,
    prior_model_checkpoint=ll_agent_config.model_checkpoint,
))
hl_agent_config.policy = ACLearnedPriorAugmentedPIPolicy

# update agent, set target divergence
agent_config.hl_agent = ActionPriorSACAgent
agent_config.hl_agent_params.update(AttrDict(
    td_schedule_params=AttrDict(p=1.), #was originally 5
))

configuration.sampler = ACMultiImageAugmentedHierarchicalSampler
configuration.num_epochs = 15
configuration.environment = ImageKitchenEnv

sampler_config = AttrDict(
    n_frames=2,
)

ll_model_params = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    kl_div_weight=1e-2,
    n_input_frames=2,
    prior_input_res=data_spec.res,
    nz_vae=10,
    n_rollout_steps=10,
)

ll_agent_config.update(AttrDict(
    model=ImageSkillPriorMdl,
    model_params=ll_model_params,
    model_checkpoint=os.path.join(os.environ["EXP_DIR"],
                                  "skill_prior_learning/kitchen/vision/"),
))

# reduce replay capacity because we are training image-based, do not dump (too large)
from spirl.rl.components.replay_buffer import SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay = SplitObsUniformReplayBuffer
agent_config.ll_agent_params.replay_params.unused_obs_size = ll_model_params.prior_input_res**2*3 * 2 + \
                                                             hl_agent_config.policy_params.action_dim   # ignore HL action
agent_config.ll_agent_params.replay_params.dump_replay = False
agent_config.hl_agent_params.replay_params.dump_replay = False
