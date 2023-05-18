import os
import logging.config
from pathlib import Path
import platform


# PyTorch Multi-GPU DDP Constants
RANK = int(os.getenv("RANK", -1))
LOCAL_RANK = int(
    os.getenv("LOCAL_RANK", -1)
)  # https://pytorch.org/docs/stable/elastic/run.html
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

# Other Constants
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
# DEFAULT_CFG_PATH = ROOT / 'yolo/cfg/default.yaml'
NUM_THREADS = min(
    8, max(1, os.cpu_count() - 1)
)  # number of YOLOv5 multiprocessing threads
VERBOSE = True  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
LOGGING_NAME = "pose_project"
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])


def set_logging(name=LOGGING_NAME, verbose=True):
    """Sets up logging for the given name."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {name: {"level": level, "handlers": [name], "propagate": False}},
        }
    )


set_logging(LOGGING_NAME, verbose=VERBOSE)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)
