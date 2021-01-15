from typing import List


class Metrics:

    def __init__(self):
        self.train_accuracies: List[float] = []
        self.train_losses: List[float] = []
        self.eval_accuracies: List[float] = []
        self.eval_losses: List[float] = []
        self.device_configs: List[dict] = []
        self.costs: List[float] = []
        self.times: List[float] = []

    def get_num_rounds(self):
        return len(self.train_accuracies)