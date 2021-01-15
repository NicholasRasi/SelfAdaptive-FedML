from federate_learning.common import Model
from federate_learning.orchestrator.control_strategy import ControlStrategy


class Static(ControlStrategy):

    def apply_strategy(self, model: Model = None, num_round: int = None):
        return self.k_fit, self.k_eval, {"epochs": self.num_epochs, "batch_size": self.batch_size}
