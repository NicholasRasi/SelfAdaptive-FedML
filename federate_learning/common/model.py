from abc import abstractmethod
from typing import Dict, Tuple
from federate_learning.common.parameters import NDArrayList
from federate_learning.common.parameters import Parameters
from federate_learning.common.metrics import Metrics


class Model:
    def __init__(self,
                 name: str,
                 framework: str,
                 logger=None,
                 model=None,
                 target_accuracy=1,
                 weights: Parameters = None):
        self.name = name
        self.framework = framework
        self.logger = logger
        self.model = model
        self.weights = None
        self.metrics = Metrics()
        self.target_accuracy = target_accuracy

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
