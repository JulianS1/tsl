from typing import Optional

import torch
from pytorch_lightning.loggers import Logger, TensorBoardLogger

from tsl.datasets import MetrLA, PemsBay, GPVARDatasetAZ
from tsl.datasets.pems_benchmarks import PeMS03, PeMS04, PeMS07, PeMS08
from tsl.nn import models

try:
    from pytorch_lightning.loggers import WandbLogger
except ImportError:
    WandbLogger = None


class Setup:
    def __init__(self, args):
        self.args = args

    def get_dataset(self, dataset_name):
        if dataset_name == 'la':
            dataset = MetrLA(impute_zeros=True)
        elif dataset_name == 'gpvar':
            dataset = GPVARDatasetAZ()
        elif dataset_name == 'pems3':
            dataset = PeMS03()
        elif dataset_name == 'pems4':
            dataset = PeMS04()
        elif dataset_name == 'pems7':
            dataset = PeMS07()
        elif dataset_name == 'pems8':
            dataset = PeMS08()
        else:
            raise ValueError(f"Dataset '{dataset_name}' not available.")
        return dataset

    def get_model(self, model_str):
        if model_str == 'gwnet':
            model = models.GraphWaveNetModel
        elif model_str == 'dcrnn':
            model = models.DCRNNModel
        elif model_str == 'gru_diffconv':
            model = models.GRUDiffConvModel
        else:
            raise NotImplementedError(f'Model "{model_str}" not available.')
        return model

    def get_logger(self, backend: Optional[str] = None) -> Optional[Logger]:
        if backend is None:
            return None
        if backend == 'wandb':
            if WandbLogger is None:
                raise ImportError("Install wandb: pip install wandb")
            return WandbLogger(save_dir=self.args.log_dir)
        elif backend == 'tensorboard':
            return TensorBoardLogger(save_dir=self.args.log_dir)
        else:
            raise ValueError(f"Logger backend '{backend}' not available. "
                             "Choose 'wandb' or 'tensorboard'.")

    def setup(self):
        dataset = self.get_dataset(self.args.dataset)
        model = self.get_model(self.args.model)
        return dataset, model
