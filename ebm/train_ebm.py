import torch
import os
import hydra
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict
from ebm.utils.data_loader import LiberoDataset
from ebm.algos.lange_dyn import langevin_dynamics
from ebm.models.energy_model import EnergyModel

from ebm.algos.dmp import DMP

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
        self.energy_model.to(self.cfg.device)
        self.best_state_dict = self.energy_model.state_dict()

        self.dmp = DMP(self.cfg.motion_primitives)

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
                for (idx, data_) in enumerate(train_dataloader):
                    if self.cfg.policy.horizon != self.cfg.data.seq_len:
                        assert self.cfg.policy.horizon < self.cfg.data.seq_len, \
                            "horizon must be smaller than sequence length"
                        data = self.data_sets.set_data_horizon(data_, self.cfg.policy.horizon)
                    else:
                        data = data_

                    energy, weights, goal, energy_neg_loc = self.energy_model(data)

                    start_state = []
                    if self.cfg.policy.use_joint:
                        start_state.append(data['obs']['joint_states'][:,  0, :])
                    if self.cfg.policy.use_gripper:
                        start_state.append(data['obs']['gripper_states'][:, 0, :])
                    if self.cfg.policy.use_ee:
                        start_state.append(data['obs']['ee_states'][:, 0, :])

                    y_0 = torch.cat(start_state, dim=-1).to(weights.device)
                    y_dot_0 = torch.zeros_like(y_0, device=y_0.device, requires_grad=True, dtype=y_0.dtype)
                    weights = weights.view(y_0.shape[0], self.cfg.motion_primitives.num_basis_fns, y_0.shape[-1])
                    traj_pos, traj_vel, traj_accln = self.dmp.integrate(goal, weights, y_0, y_dot_0)

                    # loss
                    # recon loss with all the traj steps
                    # global contrastive loss
                    # local contrastive loss
                    # -ve sampling




if __name__ == '__main__':
    train = TrainEBM()
    train.train()

