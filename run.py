import os 
import warnings
warnings.filterwarnings("ignore")

from utils import colorstr, DEFAULT_CFG
from utils.cfg_utils import get_cfg


from engine.trainer import BaseTrainer
from exp.mpose.mpose_trainer import MposeTrainer
from exp.hiera.hiera_trainer import HIERATrainer

def main() -> int:
    #trainer = BaseTrainer()
    #trainer = MposeTrainer()
    #trainer.train()
    os.environ["MASTER_ADDR"] = "localhost"
    #if len(args.gpus) == 1:
    #    os.environ["MASTER_PORT"] = "29500"
    #os.environ["OMP_NUM_THREADS"] = "32"
    cfg = get_cfg("/home/bmw/Documents/Sebastian/pose_project/configs/direct_method.yaml")
    trainer = HIERATrainer(cfg) 
    trainer.train()
    #print(DEFAULT_CFG)
    
    return 0


if __name__ == "__main__":
    main()
