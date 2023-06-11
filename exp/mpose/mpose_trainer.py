from utils import LOGGER
from utils.torch_utils import (
    torch_distributed_zero_first,
    de_parallel,
    build_dataloader,
)
from engine.trainer import BaseTrainer


class MposeTrainer(BaseTrainer):
    def get_dataset(self, img_path, mode="train", batch=None):
        pass

    def build_dataset(self, dataset_path, mode, batch_size):
        """Builds the dataset from the dataset path."""
        pass

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """TODO: manage splits differently."""
        # Calculate stride - check if model is initialized
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(
            rank
        ):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"

        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(
            dataset, batch_size, workers, shuffle, rank
        )  # return dataloader
