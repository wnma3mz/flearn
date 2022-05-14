from .DistillTrainer import DistillTrainer, LogitTracker
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
    "TFTrainer",
    "DistillTrainer",
    "DynTrainer",
    "L2Trainer",
    "LogitTracker",
    "MaxTrainer",
    "MOONTrainer",
]
