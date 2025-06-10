import torch

from ebm.utils.data_loader import DataSet
from ebm.algos.lange_dyn import langevin_dynamics
from ebm.models.energy_model import EnergyModel

def train_ebm(datatset, task_id, benchmark, result_summary):
    train_dataloader = datatset
