from utils import colorstr, DEFAULT_CFG
from utils.cfg_utils import get_cfg


from engine.trainer import BaseTrainer
from exp.mpose.mpose_trainer import MposeTrainer
from exp.hiera.hiera_trainer import HIERATrainer

def main() -> int:
    #trainer = BaseTrainer()
    #trainer = MposeTrainer()
    #trainer.train()

    cfg = get_cfg("/home/bmw/Documents/Sebastian/pose_project/configs/direct_method.yaml")
    trainer = HIERATrainer(cfg) 
    trainer.train()
    #print(DEFAULT_CFG)
    
    return 0


if __name__ == "__main__":
    main()
