import os
import federate_learning as fl
from federate_learning.orchestrator.control_strategy import ControlType
from federate_learning.orchestrator.control_strategy.control_strategy import Target
from federate_learning.orchestrator.control_strategy.control_strategy_factory import ControlStrategyFactory

num_rounds = int(os.getenv('FL_ROUNDS') or 15)
num_epochs = int(os.getenv('FL_EPOCHS') or 1)
batch_size = int(os.getenv('FL_BATCHSIZE') or 32)
k_fit = float(os.getenv('FL_K_FIT') or 1)
k_eval = float(os.getenv('FL_K_EVAL') or 1)
min_devices = int(os.getenv('FL_MIN') or 10)
control_type = int(os.getenv('FL_CONTROL') or ControlType.DYNAMIC_QUADRATIC_ROUNDS)
target_accuracy = float(os.getenv('FL_TACCURACY') or 0.8)
target_num_rounds = int(os.getenv('FL_TROUNDS') or 10)
target_network_cost = int(os.getenv('FL_TNETWORK') or 100)
model = str(os.getenv('FL_MODEL') or "mnist")

mnist_control_strategy = ControlStrategyFactory.factory(num_rounds=num_rounds,
                                                        min_devices=min_devices,
                                                        num_epochs=num_epochs,
                                                        batch_size=batch_size,
                                                        k_fit=k_fit,
                                                        k_eval=k_eval,
                                                        control_type=control_type,
                                                        target=Target(accuracy=target_accuracy,
                                                                      num_round=target_num_rounds,
                                                                      network_cost=target_network_cost))
mnist_model = fl.Model(name=model,
                       framework="TF",
                       control_strategy=mnist_control_strategy)
models_for_training = [mnist_model]

app = fl.orchestrator.OrchestratorApp(__name__)
app.orchestrator.config(available_models=models_for_training,
                        export_metrics=True)
app.run()
