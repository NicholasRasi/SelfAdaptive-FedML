from enum import IntEnum
from abc import abstractmethod
from federate_learning.common.model import Model


class ControlType(IntEnum):
    STATIC = 1
    DYNAMIC_1 = 2


class ControlStrategy:

    def __init__(self,
                 num_epochs: int,
                 batch_size: int,
                 num_rounds: int,
                 k_fit: float,
                 k_eval: float,
                 logger=None
                 ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_rounds = num_rounds
        self.k_fit = k_fit
        self.k_eval = k_eval
        self.logger = logger

    @abstractmethod
    def apply_strategy(self, model: Model = None, num_round: int = None):
        """
        Apply the control strategy
        :param model:
        :param num_round:
        :return: k_fit, k_eval, config
        """
