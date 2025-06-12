import torch
import torch.nn as nn


class EncoderProprio(nn.Module):
    def __init__(self,
                 proprio_enc_type,
                 use_joint,
                 use_gripper,
                 use_ee,
                 out_dim=256,
                 dropout=0.1):
        # Todo: check if any normalization required
        super().__init__()
        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.total_dim = int(use_joint)*joint_states_dim + \
                         int(use_gripper)*gripper_states_dim + \
                         int(use_ee)*ee_dim

        assert self.total_dim > 0, "ERROR: no proprioceptive data"

        self.proprio_encoder_type = proprio_enc_type

        if proprio_enc_type == "grouped":
            # consider all the proprioceptive observations together
            # as a single tensor and encode it
            self.layers = nn.Sequential(
                nn.Linear(self.total_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, out_dim),
            )
        else:
            raise NotImplementedError

    def get_input_data(self, data):
        torch_list = []
        if self.use_joint:
            torch_list.append(data['joint_states'])
        if self.use_gripper:
            torch_list.append(data['gripper_states'])
        if self.use_ee:
            torch_list.append(data['ee_states'])

        if self.proprio_enc_type == "grouped":
            x = torch.cat(torch_list, dim=-1)
        else:
            raise NotImplementedError
        return x

    def forward(self, x):
        return self.layers(x).unsqueeze(-2)

