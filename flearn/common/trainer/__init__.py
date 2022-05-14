from .ProxTrainer import ProxTrainer
from .Trainer import Trainer

try:
    from TFTrainer import TFTrainer
except:
    pass

__all__ = ["Trainer", "ProxTrainer", "TFTrainer"]
