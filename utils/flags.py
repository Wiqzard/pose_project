from enum import Enum, auto


class Mode(Enum):
    TRAIN = auto()
    TEST = auto()
    PREDICT = auto()
class BBoxType(Enum):
    OBJECT = auto()
    VISIBLE = auto()
    JITTER = auto() 

class AttentionMode(Enum):
    pass
 
class DistType(Enum):
    NORMAL = auto()
    UNIFORM = auto()