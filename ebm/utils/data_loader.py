import copy
import os
import hydra
from omegaconf import OmegaConf
import yaml
import pprint
from easydict import EasyDict
from tqdm import trange
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import (GroupedTaskDataset, SequenceVLDataset, get_dataset)
from libero.lifelong.utils import (get_task_embs, safe_device, create_experiment_dir)
from torch.utils.data import DataLoader, RandomSampler
import torchvision.transforms as T
from torchvision.utils import save_image
from ebm.utils.random_utils import update_data_horizon
import torch


pp = pprint.PrettyPrinter(indent=2)


class LiberoDataset():
    def __init__(self,
                 cfg,
                 benchmark_name,
                 data_folder_path,
                 bddl_folder_path,
                 init_states_folder_path):

        cfg.folder = data_folder_path
        cfg.bddl_folder = bddl_folder_path
        cfg.init_states_folder = init_states_folder_path

        cfg.eval.num_procs = 1
        cfg.eval.n_eval = 5

        # load benchmark
        task_order = cfg.data.task_order_index
        cfg.benchmark_name = benchmark_name
        benchmark = get_benchmark(cfg.benchmark_name)(task_order)

        # prepare datasets from benchmark
        datasets = []
        descriptions = []
        shape_meta = None
        self.n_tasks = benchmark.n_tasks

        for i in range(self.n_tasks):
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(cfg.folder, benchmark.get_task_demonstration(i)),
                obs_modality=cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=cfg.data.seq_len,
            )

            descriptions.append(benchmark.get_task(i).language)
            datasets.append(task_i_dataset)

        task_embs = get_task_embs(cfg, descriptions)
        benchmark.set_task_embs(task_embs)

        self.datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(datasets, task_embs)]
        self.n_demos = [data.n_demos for data in datasets]
        self.n_sequences = [data.total_num_sequences for data in datasets]
        self.shape_meta = shape_meta
        self.cfg = cfg

    def get_batch_size(self):
        assert self.n_tasks > 0

        for i in trange(self.n_tasks):
            train_dataloader = DataLoader(
                self.datasets[i],
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(self.datasets[i]),
                persistent_workers=True,
                drop_last=True,
            )
            break

        for (idx, data) in enumerate(train_dataloader):
            batch_size = data['obs']['joint_states'].shape[0]
            break

        return batch_size

    def set_data_horizon(self, data, horizon):
        data_new = copy.deepcopy(data)
        data_new = update_data_horizon(data_new, horizon)
        return data_new

    def experiments(self):
        first_ = True
        fv = None

        data = self.datasets[0][0]
        print(data['obs']['agentview_rgb'][0])
        exit()
        for i in trange(self.n_tasks):
            train_dataloader = DataLoader(
                self.datasets[i],
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(self.datasets[i]),
                persistent_workers=True,
            )

            for (idx, data) in enumerate(train_dataloader):
                print(data['task_emb'].shape[0])
                exit()
            #     if i == 0:
            #         fv = data['task_emb'][0]
            #     if i == 2:
            #         sv = data['task_emb'][0]
            #         cs = torch.cosine_similarity(fv, sv, dim=-1)
            #         print(cs)
            #         exit()

                # images = data['obs']['eye_in_hand_rgb']
                # for batch in range(images.shape[0]):
                #     for seq in range(images.shape[1]):
                #         save_image(images[batch][seq], 'output_{}.png'.format(seq))
                #     exit()
                # print(idx)
                # B, To, C, H, W = data["obs"]["agentview_rgb"].shape
                # img = data["obs"]["agentview_rgb"].view(B*To, C, H, W)
                # img = T.Resize(224)(img)
                # print(img[0, 0, 100, 100])
                # print("prior shape: ", data["obs"]["agentview_rgb"].shape)
                # print("reshaped: ", img.shape)

                # ['actions', 'obs', 'task_emb']
                # obs_keys = ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
                # """
                # data_keys : ['actions', 'obs', 'task_emb']
                # obs_keys : ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
                # actions :  torch.Size([32, 10, 7])
                # agentview_rgb  :  torch.Size([32, 10, 3, 128, 128])
                # eye_in_hand_rgb  :  torch.Size([32, 10, 3, 128, 128])
                # gripper_states  :  torch.Size([32, 10, 2])
                # joint_states  :  torch.Size([32, 10, 7])
                # task_emb :  torch.Size([32, 768])
                # """

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    config_path = "../configs"
    config_name = "config"
    data_folder_path = "/home/nune/ebm/data/LIBERO/libero/datasets"
    bddl_folder_path = "/home/nune/ebm/data/LIBERO/libero/libero/bddl_file"
    init_states_folder_path = "/home/nune/ebm/data/LIBERO/libero/libero/init_files"

    hydra.initialize(config_path=config_path)
    hydra_cfg = hydra.compose(config_name=config_name)
    yaml_config = OmegaConf.to_yaml(hydra_cfg)
    cfg = EasyDict(yaml.safe_load(yaml_config))

    d = LiberoDataset(cfg,
                      "libero_goal",
                      data_folder_path,
                      bddl_folder_path,
                      init_states_folder_path
                      )

    d.experiments()
