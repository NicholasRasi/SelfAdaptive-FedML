from scipy import interpolate
import numpy as np
from federate_learning.orchestrator.control_strategy import ControlStrategy

"""
Compute the target speed with a quadratic function
"""


class DynamicQuadraticRounds(ControlStrategy):

    def apply_strategy(self, num_round: int = None):
        # control parameters
        perc_target = 1.00

        # get the model evaluation accuracies and configs
        accs = self.metrics.accuracies_eval
        configs = self.metrics.device_configs

        # check if at least two points are available, otherwise fallback to init parameters
        if len(accs) <= 1 or len(configs) <= 1:
            return self.k_fit, self.k_eval, {"epochs": self.num_epochs, "batch_size": self.batch_size}

        # compute remaining accuracy and round
        remaining_acc = (self.target.accuracy * perc_target) - accs[-1]
        remaining_rounds = max(0, self.target.num_round - num_round)

        if self.accuracy_fun is None:
            # set the training curve to reach the 0.8 of the target accuracy in 0.5 target rounds
            rounds = [0, 0.5 * self.target.num_round, self.target.num_round]
            accuracies = [0, 0.8 * self.target.accuracy, self.target.accuracy]
            self.accuracy_fun = interpolate.interp1d(rounds, accuracies, kind='quadratic', fill_value='extrapolate')
        target_accuracy = self.accuracy_fun(num_round)

        # compute the current speed (last step)
        current_speed = accs[-1] - accs[-2]

        # cumulate the epochs
        epochs = np.cumsum([config["epochs"] for config in configs])

        # compute the interpolation function
        self.logger.warning("accs: {}, epochs: {}".format(["{0:0.4f}".format(a) for a in accs], epochs))
        epochs_fun = interpolate.interp1d(accs[-2:], epochs[-2:], kind='linear', fill_value='extrapolate')
        new_epochs = int(min(self.max_epochs, max(1, epochs_fun(target_accuracy) - epochs[-1])))

        self.logger.warning("remaining rounds: {}, remaining acc: {:0.4f}, "
                            "target_accuracy: {:0.4f}, new epochs(x): {:0.4f}, "
                            "new epochs(y): {}, current speed: {:0.4f}".format(remaining_rounds, remaining_acc,
                                                                               target_accuracy, target_accuracy,
                                                                               new_epochs, current_speed))

        return self.k_fit, self.k_eval, {"epochs": new_epochs, "batch_size": self.batch_size}
