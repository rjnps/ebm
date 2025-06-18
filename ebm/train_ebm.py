import torch
import os
import hydra
import random
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
from ebm.utils.random_utils import data_buffer_cpu, data_buffer_gpu, data_numpy_tensor
from torch.utils.tensorboard import SummaryWriter


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

        self.writer = SummaryWriter(log_dir="runs/exp1")

    def train(self):
        t0 = time.time()
        optimizer = torch.optim.Adam(lr=5e-5, params=self.energy_model.parameters(), weight_decay=1e-4)
        for i in trange(self.data_sets.n_tasks):
            train_dataloader = DataLoader(
                self.data_sets.datasets[i],
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(self.data_sets.datasets[i]),
                persistent_workers=True,
                drop_last=True
            )

            while True:
                k = random.randint(0, len(self.data_sets.datasets) - 1)
                if k != i:
                    break

            for epoch in trange(0, self.cfg.train.n_epochs + 1):
                for (idx, data_) in enumerate(train_dataloader):
                    if self.cfg.policy.horizon != self.cfg.data.seq_len:
                        assert self.cfg.policy.horizon < self.cfg.data.seq_len, \
                            "horizon must be smaller than sequence length"
                        data = self.data_sets.set_data_horizon(data_, self.cfg.policy.horizon)
                    else:
                        data = data_

                    energy, weights, goal, energy_neg_loc, latent_pos = self.energy_model(data)

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

                    # -- Neg sampling --
                    # get data from a different task

                    # idx_k = random.randint(0, len(self.data_sets.datasets[k]) - 1)
                    # diff_data = self.data_sets.datasets[k][idx_k]
                    # diff_data = data_numpy_tensor(diff_data, y_0.shape[0], "cuda")

                    # get global -ve loss
                    # energy_cross_sample, energy_cross_task = self.energy_model.get_global_neg_energy_loss(data,
                    #                                                                                       diff_data,
                    #                                                                                       latent_pos)
                    #
                    # print("energy_cross_sample", energy_cross_sample)
                    # print("energy_cross_task", energy_cross_task)

                    recons_loss = torch.nn.functional.mse_loss(traj_pos[..., :7].reshape(-1, 7),
                                                               data_['actions'].reshape(-1, 7).to(traj_pos.device))

                    local_energy_loss = torch.mean(energy - energy_neg_loc)

                    energy_reg = 1e-5 * ((energy ** 2).mean() + (energy_neg_loc ** 2).mean())

                    loss = local_energy_loss + recons_loss + energy_reg

                    if idx % 500 == 0:
                        print("-------------LOSS--------------")
                        print("total loss : ", loss)
                        print("energy loss : ", local_energy_loss)
                        print("recons loss : ", recons_loss)
                        print("energy reg : ", energy_reg)
                        print("-------------------------------")

                    self.writer.add_scalar("total_loss", loss.item(), epoch * len(train_dataloader) + idx)
                    self.writer.add_scalar("energy_loss", local_energy_loss.item(), epoch * len(train_dataloader) + idx)
                    self.writer.add_scalar("recons loss", recons_loss.item(), epoch * len(train_dataloader) + idx)
                    self.writer.add_scalar("pos energy", energy.sum().item(), epoch * len(train_dataloader) + idx)
                    self.writer.add_scalar("neg energy", energy_neg_loc.sum().item(), epoch * len(train_dataloader) + idx)
                    self.writer.add_scalar("reg energy", energy_reg.sum().item(), epoch * len(train_dataloader) + idx)

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.energy_model.parameters(), max_norm=1.0)
                    optimizer.step()



                    # loss
                    # recon loss with all the traj steps
                    # global contrastive loss
                    # local contrastive loss
                    # -ve sampling




if __name__ == '__main__':
    train = TrainEBM()
    train.train()

