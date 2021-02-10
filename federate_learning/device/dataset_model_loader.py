import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import numpy as np


class DatasetModelLoader:

    def __init__(self, model: str):
        self.model = model


    def get_dataset(self, nk: int):
        if self.model == "mnist":
            # load MNIST data
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            # select random samples
            nk = min(nk, 10000)  # 60000 for train, 10000 for evaluation
            x_train, y_train = self.select_random_samples(x_train, y_train, nk)
            x_test, y_test = self.select_random_samples(x_test, y_test, nk)

            return x_train, y_train, x_test, y_test
        elif self.model == "fashion_mnist":  # https://www.tensorflow.org/tutorials/keras/classification
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
            train_images = train_images / 255.0
            test_images = test_images / 255.0

            # select random samples
            nk = min(nk, 10000)  # 60000 for train, 10000 for evaluation
            train_images, train_labels = self.select_random_samples(train_images, train_labels, nk)
            test_images, test_labels = self.select_random_samples(test_images, test_labels, nk)

            return train_images, train_labels, test_images, test_labels
        elif self.model == "cifar10":  # https://www.tensorflow.org/tutorials/images/cnn
            (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
            train_images, test_images = train_images / 255.0, test_images / 255.0

            # select random samples
            nk = min(nk, 10000)  # 50000 for train, 10000 for evaluation
            train_images, train_labels = self.select_random_samples(train_images, train_labels, nk)
            test_images, test_labels = self.select_random_samples(test_images, test_labels, nk)

            return train_images, train_labels, test_images, test_labels
        elif self.model == "imdb_reviews":  # https://builtin.com/data-science/how-build-neural-network-keras
            (training_data, training_targets), (testing_data, testing_targets) = tf.keras.datasets.imdb.load_data(num_words=10000)

            # prepare data
            def vectorize(sequences, dimension=10000):
                results = np.zeros((len(sequences), dimension))

                for i, sequence in enumerate(sequences):
                    results[i, sequence] = 1
                return results

            training_data = vectorize(training_data)
            testing_data = vectorize(testing_data)
            training_targets = np.array(training_targets).astype("float32")
            testing_targets = np.array(testing_targets).astype("float32")

            # select random samples
            nk = min(nk, 25000)  # 25000 for train, 25000 for evaluation
            training_data, training_targets = self.select_random_samples(training_data, training_targets, nk)
            testing_data, testing_targets = self.select_random_samples(testing_data, testing_targets, nk)

            return training_data, training_targets, testing_data, testing_targets

    def get_compiled_model(self, optimizer: str):
        if self.model == "mnist":
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
                optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
            )
            return tf_model
        elif self.model == "fashion_mnist":  # https://www.tensorflow.org/tutorials/keras/classification
            tf_model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10)
            ])
            tf_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
            return tf_model
        elif self.model == "cifar10":  # https://www.tensorflow.org/tutorials/images/cnn
            tf_model = tf.keras.models.Sequential()
            tf_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
            tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            tf_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
            tf_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
            tf_model.add(tf.keras.layers.Flatten())
            tf_model.add(tf.keras.layers.Dense(64, activation='relu'))
            tf_model.add(tf.keras.layers.Dense(10))

            tf_model.compile(optimizer=optimizer,
                             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                             metrics=['accuracy'])
            return tf_model
        elif self.model == "imdb_reviews":  # https://builtin.com/data-science/how-build-neural-network-keras
            # Input - Layer
            tf_model = tf.keras.models.Sequential()
            tf_model.add(tf.keras.layers.Dense(50, activation="relu", input_shape=(10000,)))
            # Hidden - Layers
            tf_model.add(tf.keras.layers.Dropout(0.3, noise_shape=None, seed=None))
            tf_model.add(tf.keras.layers.Dense(50, activation="relu"))
            tf_model.add(tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None))
            tf_model.add(tf.keras.layers.Dense(50, activation="relu"))
            # Output- Layer
            tf_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

            tf_model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )

            return tf_model

    def select_random_samples(self, x_train, y_train, nk):
        indices = np.random.choice(x_train.shape[0], nk, replace=False)
        x_train = x_train[indices]
        y_train = y_train[indices]
        return x_train, y_train


