from utils import colorstr, DEFAULT_CFG
from utils.cfg_utils import get_cfg


from engine.trainer import BaseTrainer
from exp.mpose.mpose_trainer import MposeTrainer

def main() -> int:
    #trainer = BaseTrainer()
    trainer = MposeTrainer()
    trainer.train()
    #print(DEFAULT_CFG)
    
    return 0


if __name__ == "__main__":
    main()
