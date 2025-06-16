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
