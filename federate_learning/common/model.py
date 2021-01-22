from abc import abstractmethod
from typing import Dict, Tuple
from federate_learning.common.parameters import NDArrayList
from federate_learning.common.metrics import Metrics
from federate_learning.orchestrator.control_strategy import ControlStrategy


class Model:
    def __init__(self,
                 name: str,
                 framework: str,
                 control_strategy: ControlStrategy = None,
                 model=None,
                 logger=None):
        self.name = name
        self.framework = framework
        self.logger = logger
        self.model = model
        self.weights = None
        self.metrics = Metrics()
        if control_strategy:
            self.control_strategy = control_strategy
            self.control_strategy.metrics = self.metrics
            self.control_strategy.logger = self.logger

    @abstractmethod
    def get_weights(self) -> NDArrayList:
        """

        :return:
        """

    @abstractmethod
    def fit(self,
            weights: NDArrayList,
            config: Dict[str, str]) -> Tuple[NDArrayList, int, float, float]:
        """

        :param weights:
        :param config:
        :return:
        """

    @abstractmethod
    def evaluate(self,
            weights: NDArrayList,
            config: Dict[str, str]) -> Tuple[int, float, float]:
        """

        :param weights:
        :param config:
        :return:
        """
