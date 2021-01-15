from scipy import interpolate
from federate_learning.common import Model
from federate_learning.orchestrator.control_strategy import ControlStrategy


class Dynamic1(ControlStrategy):

    def apply_strategy(self, model: Model = None, num_round: int = None):
        # control parameters
        perc_target = 1.05  # 5% above the set point

        # get the model evaluation accuracies and configs
        accs = model.metrics.eval_accuracies
        configs = model.metrics.device_configs

        # check if at least two points are available, otherwise fallback to init parameters
        if len(accs) <= 1 or len(configs) <= 1:
            return self.k_fit, self.k_eval, {"epochs": self.num_epochs, "batch_size": self.batch_size}

        # compute remaining accuracy and round
        remaining_acc = (model.target_accuracy * perc_target) - accs[-1]
        remaining_rounds = max(0, self.num_rounds - num_round)

        # compute target speed
        if remaining_rounds != 0:
            target_speed = remaining_acc / remaining_rounds
        else:
            target_speed = remaining_acc

        # compute the current speed (last step)
        current_speed = accs[-1] - accs[-2]

        # cumulate the epochs
        epochs = []
        for config in configs:
            if epochs:
                epochs.append(epochs[-1] + config["epochs"])
            else:
                epochs.append(config["epochs"])

        # compute the interpolation function
        epochs_fun = interpolate.interp1d([accs[-2], accs[-1]], [epochs[-2], epochs[-1]], kind='linear',
                                          fill_value='extrapolate')
        self.logger.warning("interpolating x: {}, y: {}".format(["{0:0.4f}".format(a) for a in accs], epochs))
        new_epochs = max(1, int(epochs_fun(accs[-1] + target_speed) - epochs[-1]))

        self.logger.warning(
            "remaining rounds: {}, remaining acc: {:0.4f}, target speed: {:0.4f}, new epochs(x): {:0.4f}, new epochs(y): {}, "
            "current speed: {:0.4f}".format(remaining_rounds, remaining_acc, target_speed, accs[-1] + target_speed,
                                            new_epochs, current_speed))

        return self.k_fit, self.k_eval, {"epochs": new_epochs, "batch_size": self.batch_size}
