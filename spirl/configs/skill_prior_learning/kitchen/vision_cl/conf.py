import os

from spirl.models.skill_prior_mdl import ImageSkillPriorMdl,SkillSpaceLogger
from spirl.models.closed_loop_spirl_mdl import ImageClSPiRLMdl
from spirl.utils.general_utils import AttrDict
from spirl.configs.default_data_configs.kitchen_vision import data_spec
from spirl.components.evaluator import TopOfNSequenceEvaluator


current_dir = os.path.dirname(os.path.realpath(__file__))


configuration = {
    'model': ImageClSPiRLMdl,
    'logger': SkillSpaceLogger,
    'data_dir': 'data/kitchen-vision/kitchen-mixed-v0-vision-v2.hdf5',
    'epoch_cycles_train': 10,
    'evaluator': TopOfNSequenceEvaluator,
    'top_of_n_eval': 100,
    'top_comp_metric': 'mse',
}
configuration = AttrDict(configuration)

model_config = AttrDict(
    state_dim=data_spec.state_dim,
    action_dim=data_spec.n_actions,
    n_rollout_steps=10,
    kl_div_weight=1e-2, #try 5e-4 as well
    prior_input_res=data_spec.res,
    n_input_frames=2,
    nz_enc=128,
    nz_mid=128,
    n_processing_layers=5,
    cond_decode=True,
)

# Dataset
data_config = AttrDict()
data_config.dataset_spec = data_spec
data_config.dataset_spec.subseq_len = model_config.n_rollout_steps + model_config.n_input_frames

