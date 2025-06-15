import torch.nn as nn
import torch
from ebm.utils.data_augmentation import (
    TranslationAugmentation,
    BatchWiseImageColorJitter,
    DataAugmentationGroup
)
import robomimic.utils.tensor_utils as TensorUtils


class BasePolicy(nn.Module):
    def __init__(self, cfg, shape_meta):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.shape_meta = shape_meta

        policy_cfg = cfg.policy

        # eval -> function, rest are its params
        # * -> positional unpacking
        # ** -> keyword unpacking

        # Data Augmentation
        color_aug = eval(policy_cfg.color_aug.network)(**policy_cfg.color_aug.network_kwargs)
        policy_cfg.translation_aug.network_kwargs["input_shape"] = shape_meta["all_shapes"][cfg.data.obs.modality.rgb[0]]
        translation_aug = eval(policy_cfg.translation_aug.network)(**policy_cfg.translation_aug.network_kwargs)
        self.img_aug = DataAugmentationGroup((color_aug, translation_aug))

    def forward(self, data):
        raise NotImplementedError

    def get_action_(self, data):
        raise NotImplementedError

    def _get_img_tuple_(self, data):
        # image encoder to be defined in that particular policy class
        img_tuple = tuple(
            [data['obs']['agentview_rgb'], data['obs']['eye_in_hand_rgb']]
        )
        return img_tuple

    def _get_aug_output_dict_(self, out):
        img_dict = {
            img_name: out[idx]
            for idx, img_name in enumerate(['agentview_rgb', 'eye_in_hand_rgb'])
        }
        return img_dict

    def preprocess_input(self, data, train_mode=True):
        if train_mode:
            if self.cfg.train.use_augmentation:
                img_tuple = self._get_img_tuple_(data)
                aug_out = self._get_aug_output_dict_(self.img_aug(img_tuple))
                # replacing images with augmented images
                for img_name in ['agentview_rgb', 'eye_in_hand_rgb']:
                    data["obs"][img_name] = aug_out[img_name]
            return data
        else:
            data = TensorUtils.recursive_dict_list_tuple_apply(
                data, {torch.Tensor: lambda x: x.unsqueeze(dim=1)}  # add time dimension
            )
            data["task_emb"] = data["task_emb"].squeeze(1)
        return data

    def compute_loss(self, data, reduction="mean"):
        raise NotImplementedError

