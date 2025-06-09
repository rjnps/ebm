import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchvision.transforms import ColorJitter
from robomimic.models.base_nets import CropRandomizer


class TranslationAugmentation(nn.Module):
    def __init__(self, input_shape, translation):
        super().__init__()

        # pad and then random crop
        self.pad_translation = translation //2
        pad_output_shape = (
            input_shape[0],
            input_shape[1] + translation,
            input_shape[2] + translation,
        )

        self.crop_randomizer = CropRandomizer(
            input_shape=pad_output_shape,
            crop_height=input_shape[1],
            crop_width=input_shape[2],
        )

    def forward(self, x):
        if self.training:
            batch_size, temporal_length, img_c, img_h, img_w = x.shape
            x = x.reshape(batch_size, temporal_length*img_c, img_h, img_w)
            # pad = (left, right, top, bottom)
            out = F.pad(x, pad=(self.pad_translation,)*4, mode="replicate")
            # random crop
            out = self.crop_randomizer.forward_in(out)
            out = out.reshape(batch_size, temporal_length, img_c, img_h, img_w)
        else:
            out=x
        return out


class BatchWiseImageColorJitter(nn.Module):
    def __init__(self,
                 input_shape,
                 brightness=0.3,
                 contrast=0.3,
                 saturation=0.3,
                 hue=0.3,
                 epsilon=0.1):
        super().__init__()
        self.color_jitter = ColorJitter(brightness=brightness,
                                        contrast=contrast,
                                        saturation=saturation,
                                        hue=hue)
        self.epsilon = epsilon

    def forward(self, x):
        out=[]
        # or torch.unbind to get a tuple of tensors along the batch dimension
        for x_i in torch.split(x, 1):
            if self.training and np.random.random() > self.epsilon:
                out.append(self.color_jitter(x_i))
            else:
                out.append(x_i)
        return torch.cat(out, dim=0)


# To do a bunch of data augmentation together
class DataAugmentationGroup(nn.Module):
    def __init__(self, aug_list):
        super().__init__()
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, x_groups):
        # input -> a tuple of images
        # split this along the channel dimension
        # join them together along channel dimension
        # do augmentations
        # use the split_list to join them again
        split_channels=[]
        for i in range(len(x_groups)):
            split_channels.append(x_groups[i].shape[1])
        if self.training:
            x = torch.cat(x_groups, dim=1)
            out = self.aug_layer(x)
            out = torch.split(out, split_channels, dim=1)
        else:
            out = x_groups
        return out


