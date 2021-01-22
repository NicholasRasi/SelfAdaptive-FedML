from typing import List


class Metrics:

    def __init__(self):
        self.ks_fit: List[float] = []
        self.nums_devices_fit: List[float] = []
        self.losses_fit: List[float] = []
        self.accuracies_fit: List[float] = []
        self.computation_costs_fit: List[float] = []
        self.network_costs_fit: List[float] = []
        self.ks_eval: List[float] = []
        self.nums_devices_eval: List[float] = []
        self.losses_eval: List[float] = []
        self.accuracies_eval: List[float] = []
        self.computation_costs_eval: List[float] = []
        self.network_costs_eval: List[float] = []
        self.device_configs: List[dict] = []

    def get_num_rounds(self):
        return len(self.accuracies_fit)

    def add_metrics(self,
                    k_fit: float,
                    num_devices_fit: int,
                    loss_fit: float,
                    accuracy_fit: float,
                    computation_cost_fit: float,
                    network_cost_fit: float,
                    k_eval: float,
                    num_devices_eval: int,
                    loss_eval: float,
                    accuracy_eval: float,
                    computation_cost_eval: float,
                    network_cost_eval: float,
                    device_config: dict):
        self.ks_fit.append(k_fit)
        self.nums_devices_fit.append(num_devices_fit)
        self.losses_fit.append(loss_fit)
        self.accuracies_fit.append(accuracy_fit)
        self.computation_costs_fit.append(computation_cost_fit)
        self.network_costs_fit.append(network_cost_fit)

        self.ks_eval.append(k_eval)
        self.nums_devices_eval.append(num_devices_eval)
        self.losses_eval.append(loss_eval)
        self.accuracies_eval.append(accuracy_eval)
        self.computation_costs_eval.append(computation_cost_eval)
        self.network_costs_eval.append(network_cost_eval)

        self.device_configs.append(device_config)
