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


pp = pprint.PrettyPrinter(indent=2)


class DataSet():
    def __init__(self,
                 config_path,
                 config_name,
                 data_folder_path,
                 bddl_folder_path,
                 init_states_folder_path):

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ['PYOPENGL_PLATFORM'] = 'egl'

        # init hydra
        hydra.initialize(config_path=config_path,)
        hydra_cfg = hydra.compose(config_name=config_name)
        yaml_config = OmegaConf.to_yaml(hydra_cfg)
        cfg = EasyDict(yaml.safe_load(yaml_config))

        pp.pprint(cfg.policy)

        cfg.folder = data_folder_path
        cfg.bddl_folder = bddl_folder_path
        cfg.init_states_folder = init_states_folder_path

        cfg.eval.num_procs = 1
        cfg.eval.n_eval = 5

        # load benchmark
        task_order = cfg.data.task_order_index
        cfg.benchmark_name = "libero_goal"
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

        self.cfg = cfg

    def experiments(self):
        for i in trange(self.n_tasks):
            train_dataloader = DataLoader(
                self.datasets[i],
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                sampler=RandomSampler(self.datasets[i]),
                persistent_workers=True,
            )

            for (idx, data) in enumerate(train_dataloader):
                pp.pprint(data["obs"]["agentview_rgb"])

                # ['actions', 'obs', 'task_emb']
                obs_keys = ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
                """
                data_keys : ['actions', 'obs', 'task_emb']
                obs_keys : ['agentview_rgb', 'eye_in_hand_rgb', 'gripper_states', 'joint_states']
                actions :  torch.Size([32, 10, 7])
                agentview_rgb  :  torch.Size([32, 10, 3, 128, 128])
                eye_in_hand_rgb  :  torch.Size([32, 10, 3, 128, 128])
                gripper_states  :  torch.Size([32, 10, 2])
                joint_states  :  torch.Size([32, 10, 7])
                task_emb :  torch.Size([32, 768])
                """

if __name__ == "__main__":
    d = DataSet(config_path="../configs",
                config_name="config",
                data_folder_path="/home/nune/ebm/data/LIBERO/libero/datasets",
                bddl_folder_path="/home/nune/ebm/data/LIBERO/libero/libero/bddl_file",
                init_states_folder_path="/home/nune/ebm/data/LIBERO/libero/libero/init_files")

    d.experiments()
