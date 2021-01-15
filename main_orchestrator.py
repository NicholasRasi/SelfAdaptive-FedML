import federate_learning as fl
from federate_learning.orchestrator.control_strategy import ControlType
import os


num_rounds = os.getenv('FL_R') or 15
target_accuracy = os.getenv('FL_ACC') or 0.8
k_fit = os.getenv('FL_K_FIT') or 1
k_eval = os.getenv('FL_K_EVAL') or 1
min_devices = os.getenv('FL_MIN') or 10

mnist_model = fl.Model(name="Mnist", framework="TF", target_accuracy=0.8)
models_for_training = [mnist_model]

app = fl.orchestrator.OrchestratorApp(__name__)
app.orchestrator.config(available_models=models_for_training,
                        num_rounds=num_rounds,
                        min_devices=min_devices,
                        k_fit=k_fit,
                        k_eval=k_eval,
                        num_epochs=1,
                        batch_size=32,
                        control_type=ControlType.DYNAMIC_1)
app.run()
