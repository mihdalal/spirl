from spirl.utils.general_utils import AttrDict
from spirl.data.kitchen.src.kitchen_data_loader import D4RLVisionSequenceSplitDataset


data_spec = AttrDict(
    dataset_class=D4RLVisionSequenceSplitDataset,
    n_actions=9,
    state_dim=60,
    res=64,
    crop_rand_subseq=True,
)
data_spec.max_seq_len = 280
