import random
import federate_learning as fl
import os
import tensorflow as tf
import statistics
import numpy as np

# Tensorflow init
tf.config.optimizer.set_jit(True)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# device port
port = os.getenv('FL_PORT') or random.randint(5100, 5200)

# local sample size n_k
nk = int(os.getenv('FL_NK') or 100)

# load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# select random samples
indices = np.random.choice(x_train.shape[0], nk, replace=False)
x_train = x_train[indices]
y_train = y_train[indices]

# build and compile Keras model
tf_model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)
tf_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

class MnistModel(fl.Model):
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


mnist_model = MnistModel(name="Mnist", framework="TF", model=tf_model)

# init available models
available_models = [mnist_model]

app = fl.device.DeviceApp(__name__)
app.device.config(host="localhost",
                  port=port,
                  server_host="localhost:5000",
                  available_models=available_models)

app.run()
