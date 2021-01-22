from federate_learning.orchestrator.control_strategy import ControlType, Target, Static, DynamicLinearRounds
from federate_learning.orchestrator.control_strategy import DynamicQuadraticRounds, DynamicLinearNetwork, DynamicQuadraticNetwork


class ControlStrategyFactory:

    @staticmethod
    def factory(num_rounds: int = None,
                min_devices: int = None,
                num_epochs: int = None,
                batch_size: int = None,
                k_fit: float = None,
                k_eval: float = None,
                control_type: int = ControlType.STATIC,
                target: Target = None,
                logger=None):
        if control_type == ControlType.DYNAMIC_LINEAR_ROUNDS:
            control_strategy = DynamicLinearRounds(num_rounds=num_rounds,
                                                   min_devices=min_devices,
                                                   num_epochs=num_epochs,
                                                   batch_size=batch_size,
                                                   k_fit=k_fit,
                                                   k_eval=k_eval,
                                                   target=target,
                                                   logger=logger)
        elif control_type == ControlType.DYNAMIC_QUADRATIC_ROUNDS:
            control_strategy = DynamicQuadraticRounds(num_rounds=num_rounds,
                                                       min_devices=min_devices,
                                                       num_epochs=num_epochs,
                                                       batch_size=batch_size,
                                                       k_fit=k_fit,
                                                       k_eval=k_eval,
                                                       target=target,
                                                       logger=logger)
        elif control_type == ControlType.DYNAMIC_LINEAR_NETWORK:
            control_strategy = DynamicLinearNetwork(num_rounds=num_rounds,
                                                    min_devices=min_devices,
                                                    num_epochs=num_epochs,
                                                    batch_size=batch_size,
                                                    k_fit=k_fit,
                                                    k_eval=k_eval,
                                                    target=target,
                                                    logger=logger)
        elif control_type == ControlType.DYNAMIC_QUADRATIC_NETWORK:
            control_strategy = DynamicQuadraticNetwork(num_rounds=num_rounds,
                                                       min_devices=min_devices,
                                                       num_epochs=num_epochs,
                                                       batch_size=batch_size,
                                                       k_fit=k_fit,
                                                       k_eval=k_eval,
                                                       target=target,
                                                       logger=logger)
        else:
            # control_type == ControlType.STATIC:
            control_strategy = Static(num_rounds=num_rounds,
                                      min_devices=min_devices,
                                      num_epochs=num_epochs,
                                      batch_size=batch_size,
                                      k_fit=k_fit,
                                      k_eval=k_eval,
                                      target=target,
                                      logger=logger)
        return control_strategy
