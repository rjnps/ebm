import torch
import os
import hydra
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict
from ebm.utils.data_loader import LiberoDataset
from ebm.algos.lange_dyn import langevin_dynamics
from ebm.models.energy_model import EnergyModel

from tqdm import trange
import time
from torch.utils.data import DataLoader, RandomSampler


class TrainEBM:
    def __init__(self,
                 benchmark_name="libero_goal",
                 config_path="configs",
                 config_name="config",
                 data_folder_path="/home/nune/ebm/data/LIBERO/libero/datasets",
                 bddl_folder_path="/home/nune/ebm/data/LIBERO/libero/libero/bddl_file",
                 init_states_folder_path="/home/nune/ebm/data/LIBERO/libero/libero/init_files"
                 ):

        hydra.initialize(config_path=config_path)
        hydra_cfg = hydra.compose(config_name=config_name)
        yaml_config = OmegaConf.to_yaml(hydra_cfg)
        self.cfg = EasyDict(yaml.safe_load(yaml_config))

        self.data_sets = LiberoDataset(
            self.cfg,
            benchmark_name,
            data_folder_path,
            bddl_folder_path,
            init_states_folder_path
        )

        # energy model
        self.energy_model = EnergyModel(
            self.cfg,
            self.data_sets.shape_meta,
            self.data_sets.get_batch_size(),
            embed_size_inp=self.cfg.policy.embed_size,
            embed_size_cond=self.cfg.policy.embed_size,
            load_encoded_data=True
            )

        self.best_state_dict = self.energy_model.state_dict()

    def train(self):
        t0 = time.time()
        for i in trange(self.data_sets.n_tasks):
            train_dataloader = DataLoader(
                self.data_sets.datasets[i],
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(self.data_sets.datasets[i]),
                persistent_workers=True,
            )

            for epoch in trange(0, self.cfg.train.n_epochs + 1):
                for (idx, data) in enumerate(train_dataloader):
                    print(idx)


if __name__ == '__main__':
    train = TrainEBM()
    train.train()

