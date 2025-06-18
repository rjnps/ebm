import numpy as np
import torch


def update_data_horizon(data, horizon):
    if isinstance(data, dict):
        new_dict = {}
        for key, val in data.items():
            new_dict[key] = update_data_horizon(val, horizon)
        return new_dict

    elif isinstance(data, torch.Tensor):
        if data.dim() > 2:
            return data[:, :horizon, ...]
        else:
            return data

    else:
        return data


def data_numpy_tensor(data, batch, device):
    if isinstance(data, dict):
        new_dict = {}
        for key, val in data.items():
            new_dict[key] = data_numpy_tensor(val, batch, device)
        return new_dict
    elif isinstance(data, np.ndarray):
        data_ = torch.from_numpy(data).to(device).requires_grad_(True)
        expand_shape = (batch,) + data_.shape
        data_ = data_.unsqueeze(0).expand(*expand_shape)
        return data_
    elif isinstance(data, torch.Tensor):
        expand_shape = (batch,) + data.shape
        data_ = data.unsqueeze(0).expand(*expand_shape)
        return data_
    else:
        return data


def data_buffer_cpu(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, val in data.items():
            new_dict[key] = data_buffer_cpu(val)
        return new_dict

    elif isinstance(data, torch.Tensor):
        return data.detach().cpu()

    else:
        return data.detach().cpu()


def data_buffer_gpu(data):
    if isinstance(data, dict):
        new_dict = {}
        for key, val in data.items():
            new_dict[key] = data_buffer_cpu(val)
        return new_dict

    elif isinstance(data, torch.Tensor):
        return data.to("cuda")

    else:
        return data.to("cuda")
