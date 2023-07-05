import os
import warnings

warnings.filterwarnings("ignore")

import torch

from utils import colorstr, DEFAULT_CFG
from utils.cfg_utils import get_cfg
from utils.torch_utils import clear_cuda_memory

from engine.trainer import BaseTrainer
from exp.mpose.mpose_trainer import MposeTrainer
from exp.hiera.hiera_trainer import HIERATrainer


def main() -> int:
    clear_cuda_memory()
    os.environ["MASTER_ADDR"] = "localhost"
    # if len(args.gpus) == 1:
    #    os.environ["MASTER_PORT"] = "29500"
    torch.backends.cudnn.benchmark = True
    os.environ["OMP_NUM_THREADS"] = "32"
    cfg = get_cfg(
        "/home/bmw/Documents/Sebastian/pose_project/configs/direct_method.yaml"
    )
    trainer = HIERATrainer(cfg)
    trainer.train()
    # print(DEFAULT_CFG)

    return 0

def main2() -> int:
    clear_cuda_memory() 
    os.environ["MASTER_ADDR"] = "localhost"
    trainer = MposeTrainer()
    trainer.train()


if __name__ == "__main__":
    #main()
    main2()
