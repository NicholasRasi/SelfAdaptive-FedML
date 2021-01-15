from functools import reduce
import numpy as np
from typing import List, Tuple
from federate_learning.orchestrator.aggregation_strategy import AggregationStrategy
from federate_learning.common.parameters import NDArrayList


class Aggregate(AggregationStrategy):

    @staticmethod
    def aggregate(results: List[Tuple[int, NDArrayList]]) -> NDArrayList:
        """Compute weighted average."""
        # Calculate the total number of examples used during training
        num_examples_total = sum([num_examples for num_examples, _ in results])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for num_examples, weights in results
        ]

        # Compute average weights of each layer
        weights_prime: NDArrayList = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime

    @staticmethod
    def weighted_loss_avg(samples_losses: List[Tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple clients."""
        num_total_evaluation_examples = sum(
            [num_examples for num_examples, _ in samples_losses]
        )
        weighted_losses = [num_examples * loss for num_examples, loss in samples_losses]
        return sum(weighted_losses) / num_total_evaluation_examples

    @staticmethod
    def weighted_accuracies_avg(samples_accuracies: List[Tuple[int, float]]) -> float:
        """Aggregate evaluation results obtained from multiple clients."""
        num_total_evaluation_examples = sum(
            [num_examples for num_examples, _ in samples_accuracies]
        )
        weighted_accuracies = [num_examples * acc for num_examples, acc in samples_accuracies]
        return sum(weighted_accuracies) / num_total_evaluation_examples

    @staticmethod
    def weighted_cost_sum(samples_conf: List[Tuple[int, dict]]) -> float:
        costs = [(conf["epochs"] * num_examples)/conf["batch_size"] for num_examples, conf in samples_conf]
        return sum(costs)