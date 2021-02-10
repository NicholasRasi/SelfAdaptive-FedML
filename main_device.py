import random
import federate_learning as fl
from federate_learning.device.dataset_model_loader import DatasetModelLoader
import os
import tensorflow as tf
import statistics


# Tensorflow init
tf.config.optimizer.set_jit(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# device port
port = os.getenv('FL_PORT') or random.randint(5100, 5200)

# local sample size n_k
nk = int(os.getenv('FL_NK') or 100)
model_name = str(os.getenv('FL_MODEL') or "mnist")
server_host = str(os.getenv('FL_SERVERHOST') or 'localhost:5000')

model_loader = DatasetModelLoader(model_name)
x_train, y_train, x_test, y_test = model_loader.get_dataset(nk)
tf_model = model_loader.get_compiled_model(optimizer="adam")


class TrainableModel(fl.Model):
    def get_weights(self):
        return self.model.get_weights()

    def fit(self, weights, config):
        if self.logger:
            self.logger.debug("train on {} samples, config={}".format(len(x_train), config))
        self.model.set_weights(weights)
        history = self.model.fit(x_train, y_train, epochs=config["epochs"], batch_size=config["batch_size"], steps_per_epoch=1)
        mean_acc = statistics.mean(history.history['accuracy'])
        mean_loss = statistics.mean(history.history['loss'])
        return self.model.get_weights(), len(x_train), mean_loss, mean_acc

    def evaluate(self, weights, config):
        self.model.set_weights(weights)
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return len(x_test), loss, accuracy


model = TrainableModel(name=model_name, framework="TF", model=tf_model)

# init available models
available_models = [model]

app = fl.device.DeviceApp(__name__)
app.device.config(host="localhost",
                  port=port,
                  server_host=server_host,
                  available_models=available_models)

app.run()
