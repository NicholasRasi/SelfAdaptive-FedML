from enum import IntEnum
from abc import abstractmethod


class Target:
    def __init__(self,
                 accuracy=None,
                 num_round=None,
                 network_cost=None):
        self.accuracy = accuracy
        self.num_round = num_round
        self.network_cost = network_cost


class ControlType(IntEnum):
    STATIC = 1
    DYNAMIC_LINEAR_ROUNDS = 2
    DYNAMIC_QUADRATIC_ROUNDS = 3
    DYNAMIC_LINEAR_NETWORK = 4
    DYNAMIC_QUADRATIC_NETWORK = 5


class ControlStrategy:

    def __init__(self,
                 num_rounds: int = None,
                 min_devices: int = None,
                 num_epochs: int = None,
                 batch_size: int = None,
                 k_fit: float = None,
                 k_eval: float = None,
                 target: Target = None,
                 logger=None):
        self.name = str(self.__class__.__name__)
        self.num_rounds = int(num_rounds)
        self.min_devices = int(min_devices)
        self.num_epochs = int(num_epochs)
        self.batch_size = int(batch_size)
        self.k_fit = float(k_fit)
        self.k_eval = float(k_eval)
        self.target = target
        self.logger = logger
        self.metrics = None
        self.accuracy_fun = None
        self.max_epochs = 1024


    @abstractmethod
    def apply_strategy(self, num_round: int = None):
        """
        Apply the control strategy
        :param model:
        :param num_round:
        :return: k_fit, k_eval, config
        """
