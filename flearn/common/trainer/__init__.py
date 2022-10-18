from .DistillTrainer import DistillTrainer, LogitsTracker
from .DynTrainer import DynTrainer
from .L2Trainer import L2Trainer
from .MaxTrainer import MaxTrainer
from .MOONTrainer import MOONTrainer
from .ProxTrainer import ProxTrainer
from .Trainer import Trainer

try:
    from TFTrainer import TFTrainer
except:
    pass

__all__ = [
    "Trainer",
    "ProxTrainer",
    "DistillTrainer",
    "DynTrainer",
    "L2Trainer",
    "LogitsTracker",
    "MaxTrainer",
    "MOONTrainer",
]
# "TFTrainer"
