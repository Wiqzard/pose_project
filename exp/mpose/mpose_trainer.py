from pathlib import Path
import torch
from typing import Union

from utils import LOGGER
from utils.torch_utils import (
    torch_distributed_zero_first,
    de_parallel,
    build_dataloader,
    ModelWrapper
)
from engine.trainer import BaseTrainer
from utils import colorstr
from data_tools.bop_dataset import BOPDataset
from data_tools.dummy_dataset import DummyDataset
from utils.flags import Mode

from gat_inf import GraphNet, AttentionMode

class MposeTrainer(BaseTrainer):
    def get_dataset(self, img_path, mode=Mode.TRAIN, use_cache=True, single_object=False):
        return BOPDataset(img_path, mode, use_cache=True, single_object=False)
        

    def build_dataset(self, dataset_path:Union[str, Path], mode:Mode):
        """Builds the dataset from the dataset path."""
        LOGGER.info(colorstr("bold", "red", f"Setting up {mode.name} dataset..."))
        if mode == Mode.TRAIN:
            if not self.trainset:
                dataset = self.get_dataset(dataset_path, mode=Mode.TRAIN, use_cache=True, single_object=False)
                dataset = DummyDataset(bop_dataset=dataset)
                self.trainset = dataset
            else:
                dataset = self.trainset
        elif mode == Mode.TEST:
            if not self.testset:
                dataset = self.get_dataset(dataset_path, mode=Mode.TEST, use_cache=True, single_object=False)
                dataset = DummyDataset(bop_dataset=dataset)
                self.testset = dataset
            else:
                dataset = self.testset
        return dataset

    def get_dataloader(self, dataset, batch_size=16, rank=0, mode=Mode.TRAIN):
        assert mode in [Mode.TRAIN, Mode.TEST]
        if dataset is None: 
            with torch_distributed_zero_first(
                rank
            ):  # init dataset *.cache only once if DDP
                dataset = self.build_dataset(self.dataset_path, mode)
        shuffle = mode == Mode.TRAIN 

        workers = self.args.workers if mode == Mode.TRAIN else self.args.workers * 2
        return build_dataloader(
            dataset, batch_size, workers, shuffle, rank
        )  # return dataloader

        
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = GraphNet(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            channels=cfg.channels,
            n_res_blocks=cfg.n_res_blocks,
            attention_levels=cfg.attention_levels,
            attention_mode=AttentionMode.GAT if cfg.attention_mode == "gat" else None,
            channel_multipliers=cfg.channel_multipliers,
            unpooling_levels=cfg.unpooling_levels,
            n_heads=cfg.n_heads,
            d_cond=cfg.d_cond
        )
        model = ModelWrapper(model)
        
        if weights:
            model.load(weights)
        return model
    
    def get_validator(self):
        return super().get_validator()

    def criterion(self, pred, target):
        #MSE
        return torch.mean((pred - target) ** 2)