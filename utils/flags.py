from enum import Enum, auto


class Mode(Enum):
    TRAIN = auto()
    TEST = auto()
    PREDICT = auto()
class BBoxType(Enum):
    OBJECT = auto()
    VISIBLE = auto()
    JITTER = auto() 

class MetricType(Enum):
    ANG_DIST = auto()
    EUCL_DIST = auto()
    ADD = auto()
    ADI = auto()

class AttentionType(Enum):
    pass
 
class DistType(Enum):
    NORMAL = auto()
    UNIFORM = auto()