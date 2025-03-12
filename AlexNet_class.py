from typing import Tuple, Dict
import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPool2D
from dataset_class import Dataset


class AlexNet:
    def __init__(self):
        # Layer params
        self.activation_function: str = "relu"
        self.output_activation_function: str = "softmax"
        self.output_neurons: int = 10
        self.input_img_shape: Tuple[int, int, int] = (32, 32, 3)
        self.dropout: float = 0.5
        self.learning_rate: float = 0.01
        self.loss_function: str = "categorical_crossentropy"
        self.optimizer: SGD = SGD(learning_rate=self.learning_rate)

        self.l1_kernel_size: Tuple[int, int] = (3, 3)
        self.l1_num_kernel: int = 48
        self.l1_strides: Tuple[int, int] = (2, 2)
        self.l1_padding: str = "same"
        self.l1_pool_size: Tuple[int, int] = (2, 2)
        self.l1_pool_strides: Tuple[int, int] = (1, 1)
        self.l1_pool_pad: str = "same"

        self.l2_kernel_size: Tuple[int, int] = (3, 3)
        self.l2_num_kernel: int = 96
        self.l2_strides: Tuple[int, int] = (2, 2)
        self.l2_padding: str = "same"
        self.l2_pool_size: Tuple[int, int] = (2, 2)
        self.l2_pool_strides: Tuple[int, int] = (1, 1)
        self.l2_pool_pad: str = "same"

        self.l3_kernel_size: Tuple[int, int] = (3, 3)
        self.l3_num_kernel: int = 192
        self.l3_strides: Tuple[int, int] = (1, 1)
        self.l3_padding: str = "same"

        self.l4_kernel_size: Tuple[int, int] = (3, 3)
        self.l4_num_kernel: int = 192
        self.l4_strides: Tuple[int, int] = (1, 1)
        self.l4_padding: str = "same"

        self.l5_kernel_size: Tuple[int, int] = (3, 3)
        self.l5_num_kernel: int = 256
        self.l5_strides: Tuple[int, int] = (2, 2)
        self.l5_padding: str = "same"
        self.l5_pool_strides: Tuple[int, int] = (1, 1)
        self.l5_pool_size: Tuple[int, int] = (2, 2)
        self.l5_pool_pad: str = "same"

        self.l6_num_neurons: int = 512
        self.l7_num_neurons: int = 256

        self.model: Sequential = self.initialize_model()

        # Training params
        self.train_status: bool = False
        self.dataset: Dataset = Dataset()
        self.training_history: Dict = None
        self.num_epochs: int = 20
        self.batch_size: int = 256
        self.shuffle_training_data: bool = True

    def initialize_model(self) -> Sequential:
        """
        Initializes model layers.
        """
        print("Initializing model")
        alexnet_model = Sequential()

        # Layer 1
        print("Building layer 1...")
        alexnet_model.add(
            Conv2D(
                filters=self.l1_num_kernel,
                kernel_size=self.l1_kernel_size,
                strides=self.l1_strides,
                activation=self.activation_function,
                input_shape=self.input_img_shape,
                padding=self.l1_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l1_pool_size,
                strides=self.l1_pool_strides,
                padding=self.l1_pool_pad,
            )
        )

        # Layer 2
        print("Building layer 2...")
        alexnet_model.add(
            Conv2D(
                filters=self.l2_num_kernel,
                kernel_size=self.l2_kernel_size,
                strides=self.l2_strides,
                activation=self.activation_function,
                padding=self.l2_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l2_pool_size,
                strides=self.l2_pool_strides,
                padding=self.l2_pool_pad,
            )
        )

        # Layer 3
        print("Building layer 3...")
        alexnet_model.add(
            Conv2D(
                filters=self.l3_num_kernel,
                kernel_size=self.l3_kernel_size,
                strides=self.l3_strides,
                activation=self.activation_function,
                padding=self.l3_padding,
            )
        )
        alexnet_model.add(BatchNormalization())

        # Layer 4
        print("Building layer 4...")
        alexnet_model.add(
            Conv2D(
                filters=self.l4_num_kernel,
                kernel_size=self.l4_kernel_size,
                strides=self.l4_strides,
                activation=self.activation_function,
                padding=self.l4_padding,
            )
        )
        alexnet_model.add(BatchNormalization())

        # Layer 5
        print("Building layer 5...")
        alexnet_model.add(
            Conv2D(
                filters=self.l5_num_kernel,
                kernel_size=self.l5_kernel_size,
                strides=self.l5_strides,
                activation=self.activation_function,
                padding=self.l5_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l5_pool_size,
                strides=self.l5_pool_strides,
                padding=self.l5_pool_pad,
            )
        )

        # FC layer 6
        print("Building layer 6...")
        alexnet_model.add(Flatten())
        alexnet_model.add(
            Dense(self.l6_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # FC layer 7
        print("Building layer 7...")
        alexnet_model.add(
            Dense(self.l7_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # Output softmax layer
        print("Building output layer...")
        alexnet_model.add(
            Dense(self.output_neurons, activation=self.output_activation_function)
        )

        alexnet_model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=["accuracy"]
        )

        # Check model layers
        print("Model architecture:")
        print(alexnet_model.summary())

        return alexnet_model

    def train_model(self) -> None:
        """
        Trains model using the CIFAR-10 dataset, and caches training history.
        """
        print("Training model")
        training_history = self.model.fit(
            self.dataset.training_dataset,
            self.dataset.training_labels,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data,
            validation_data=(
                self.dataset.validation_dataset,
                self.dataset.validation_labels,
            ),
        )

        self.train_status = True
        self.training_history = training_history

    def eval_model(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns categorical cross entropy results, predicted and true labels for testing dataset.
        """
        print("Evaluating model")
        if not self.train_status:
            raise ValueError("Model has not been trained.")

        # loss function results
        loss, accuracy = self.model.evaluate(
            self.dataset.testing_dataset, self.dataset.testing_labels
        )

        # Classification results, convert prob to class labels
        predictions = self.model.predict(self.dataset.testing_dataset)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.testing_labels, axis=1)

        return loss, accuracy, pred_labels, true_labels

    def generate_confusion_matrix(
        self, pred_labels: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generates a confusion matrix.
        """
        print("Generating confusion matrix")
        matrix = confusion_matrix(true_labels, pred_labels)

        # Categories are indexed in sequential order
        classification_categories = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        matrix_df = pd.DataFrame(
            matrix, index=classification_categories, columns=classification_categories
        )

        return matrix_df


class AlexNet2:
    def __init__(self):
        # Layer params
        self.activation_function: str = "relu"
        self.output_activation_function: str = "softmax"
        self.output_neurons: int = 10
        self.input_img_shape: Tuple[int, int, int] = (32, 32, 3)
        self.dropout: float = 0.5
        self.learning_rate: float = 0.01
        self.loss_function: str = "categorical_crossentropy"
        self.optimizer: SGD = SGD(learning_rate=self.learning_rate)

        self.l1_kernel_size: Tuple[int, int] = (3, 3)
        self.l1_num_kernel: int = 48
        self.l1_strides: Tuple[int, int] = (2, 2)
        self.l1_padding: str = "same"
        self.l1_pool_size: Tuple[int, int] = (2, 2)
        self.l1_pool_strides: Tuple[int, int] = (1, 1)
        self.l1_pool_pad: str = "same"

        self.l2_kernel_size: Tuple[int, int] = (3, 3)
        self.l2_num_kernel: int = 96
        self.l2_strides: Tuple[int, int] = (2, 2)
        self.l2_padding: str = "same"
        self.l2_pool_size: Tuple[int, int] = (2, 2)
        self.l2_pool_strides: Tuple[int, int] = (1, 1)
        self.l2_pool_pad: str = "same"

        self.l3_kernel_size: Tuple[int, int] = (3, 3)
        self.l3_num_kernel: int = 192
        self.l3_strides: Tuple[int, int] = (1, 1)
        self.l3_padding: str = "same"

        self.l4_kernel_size: Tuple[int, int] = (3, 3)
        self.l4_num_kernel: int = 192
        self.l4_strides: Tuple[int, int] = (1, 1)
        self.l4_padding: str = "same"

        self.l6_num_neurons: int = 512
        self.l7_num_neurons: int = 256

        self.model: Sequential = self.initialize_model()

        # Training params
        self.train_status: bool = False
        self.dataset: Dataset = Dataset()
        self.training_history: Dict = None
        self.num_epochs: int = 20
        self.batch_size: int = 256
        self.shuffle_training_data: bool = True

    def initialize_model(self) -> Sequential:
        """
        Initializes model layers.
        """
        print("Initializing model")
        alexnet_model = Sequential()

        # Layer 1
        print("Building layer 1...")
        alexnet_model.add(
            Conv2D(
                filters=self.l1_num_kernel,
                kernel_size=self.l1_kernel_size,
                strides=self.l1_strides,
                activation=self.activation_function,
                input_shape=self.input_img_shape,
                padding=self.l1_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l1_pool_size,
                strides=self.l1_pool_strides,
                padding=self.l1_pool_pad,
            )
        )

        # Layer 2
        print("Building layer 2...")
        alexnet_model.add(
            Conv2D(
                filters=self.l2_num_kernel,
                kernel_size=self.l2_kernel_size,
                strides=self.l2_strides,
                activation=self.activation_function,
                padding=self.l2_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l2_pool_size,
                strides=self.l2_pool_strides,
                padding=self.l2_pool_pad,
            )
        )

        # Layer 3
        print("Building layer 3...")
        alexnet_model.add(
            Conv2D(
                filters=self.l3_num_kernel,
                kernel_size=self.l3_kernel_size,
                strides=self.l3_strides,
                activation=self.activation_function,
                padding=self.l3_padding,
            )
        )
        alexnet_model.add(BatchNormalization())

        # Layer 4
        print("Building layer 4...")
        alexnet_model.add(
            Conv2D(
                filters=self.l4_num_kernel,
                kernel_size=self.l4_kernel_size,
                strides=self.l4_strides,
                activation=self.activation_function,
                padding=self.l4_padding,
            )
        )
        alexnet_model.add(BatchNormalization())

        # Layer 5
        print("Skipping layer 5...")

        # FC layer 6
        print("Building layer 6...")
        alexnet_model.add(Flatten())
        alexnet_model.add(
            Dense(self.l6_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # FC layer 7
        print("Building layer 7...")
        alexnet_model.add(
            Dense(self.l7_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # Output softmax layer
        print("Building output layer...")
        alexnet_model.add(
            Dense(self.output_neurons, activation=self.output_activation_function)
        )

        alexnet_model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=["accuracy"]
        )

        # Check model layers
        print("Model architecture:")
        print(alexnet_model.summary())

        return alexnet_model

    def train_model(self) -> None:
        """
        Trains model using the CIFAR-10 dataset, and caches training history.
        """
        print("Training model")
        training_history = self.model.fit(
            self.dataset.training_dataset,
            self.dataset.training_labels,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data,
            validation_data=(
                self.dataset.validation_dataset,
                self.dataset.validation_labels,
            ),
        )

        self.train_status = True
        self.training_history = training_history

    def eval_model(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns categorical cross entropy results, predicted and true labels for testing dataset.
        """
        print("Evaluating model")
        if not self.train_status:
            raise ValueError("Model has not been trained.")

        # loss function results
        loss, accuracy = self.model.evaluate(
            self.dataset.testing_dataset, self.dataset.testing_labels
        )

        # Classification results, convert prob to class labels
        predictions = self.model.predict(self.dataset.testing_dataset)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.testing_labels, axis=1)

        return loss, accuracy, pred_labels, true_labels

    def generate_confusion_matrix(
        self, pred_labels: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generates a confusion matrix.
        """
        print("Generating confusion matrix")
        matrix = confusion_matrix(true_labels, pred_labels)

        # Categories are indexed in sequential order
        classification_categories = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        matrix_df = pd.DataFrame(
            matrix, index=classification_categories, columns=classification_categories
        )

        return matrix_df
    
class AlexNet3:
    def __init__(self):
        # Layer params
        self.activation_function: str = "relu"
        self.output_activation_function: str = "softmax"
        self.output_neurons: int = 10
        self.input_img_shape: Tuple[int, int, int] = (32, 32, 3)
        self.dropout: float = 0.5
        self.learning_rate: float = 0.01
        self.loss_function: str = "categorical_crossentropy"
        self.optimizer: SGD = SGD(learning_rate=self.learning_rate)

        self.l1_kernel_size: Tuple[int, int] = (3, 3)
        self.l1_num_kernel: int = 48
        self.l1_strides: Tuple[int, int] = (2, 2)
        self.l1_padding: str = "same"
        self.l1_pool_size: Tuple[int, int] = (2, 2)
        self.l1_pool_strides: Tuple[int, int] = (1, 1)
        self.l1_pool_pad: str = "same"

        self.l2_kernel_size: Tuple[int, int] = (3, 3)
        self.l2_num_kernel: int = 96
        self.l2_strides: Tuple[int, int] = (2, 2)
        self.l2_padding: str = "same"
        self.l2_pool_size: Tuple[int, int] = (2, 2)
        self.l2_pool_strides: Tuple[int, int] = (1, 1)
        self.l2_pool_pad: str = "same"

        self.l3_kernel_size: Tuple[int, int] = (3, 3)
        self.l3_num_kernel: int = 192
        self.l3_strides: Tuple[int, int] = (1, 1)
        self.l3_padding: str = "same"

        self.l5_kernel_size: Tuple[int, int] = (3, 3)
        self.l5_num_kernel: int = 256
        self.l5_strides: Tuple[int, int] = (2, 2)
        self.l5_padding: str = "same"
        self.l5_pool_strides: Tuple[int, int] = (1, 1)
        self.l5_pool_size: Tuple[int, int] = (2, 2)
        self.l5_pool_pad: str = "same"

        self.l6_num_neurons: int = 512
        self.l7_num_neurons: int = 256

        self.model: Sequential = self.initialize_model()

        # Training params
        self.train_status: bool = False
        self.dataset: Dataset = Dataset()
        self.training_history: Dict = None
        self.num_epochs: int = 20
        self.batch_size: int = 256
        self.shuffle_training_data: bool = True

    def initialize_model(self) -> Sequential:
        """
        Initializes model layers.
        """
        print("Initializing model")
        alexnet_model = Sequential()

        # Layer 1
        print("Building layer 1...")
        alexnet_model.add(
            Conv2D(
                filters=self.l1_num_kernel,
                kernel_size=self.l1_kernel_size,
                strides=self.l1_strides,
                activation=self.activation_function,
                input_shape=self.input_img_shape,
                padding=self.l1_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l1_pool_size,
                strides=self.l1_pool_strides,
                padding=self.l1_pool_pad,
            )
        )

        # Layer 2
        print("Building layer 2...")
        alexnet_model.add(
            Conv2D(
                filters=self.l2_num_kernel,
                kernel_size=self.l2_kernel_size,
                strides=self.l2_strides,
                activation=self.activation_function,
                padding=self.l2_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l2_pool_size,
                strides=self.l2_pool_strides,
                padding=self.l2_pool_pad,
            )
        )

        # Layer 3
        print("Building layer 3...")
        alexnet_model.add(
            Conv2D(
                filters=self.l3_num_kernel,
                kernel_size=self.l3_kernel_size,
                strides=self.l3_strides,
                activation=self.activation_function,
                padding=self.l3_padding,
            )
        )
        alexnet_model.add(BatchNormalization())

        # Layer 4
        print("Skipping layer 4...")

        # Layer 5
        print("Building layer 5...")
        alexnet_model.add(
            Conv2D(
                filters=self.l5_num_kernel,
                kernel_size=self.l5_kernel_size,
                strides=self.l5_strides,
                activation=self.activation_function,
                padding=self.l5_padding,
            )
        )
        alexnet_model.add(BatchNormalization())
        alexnet_model.add(
            MaxPool2D(
                pool_size=self.l5_pool_size,
                strides=self.l5_pool_strides,
                padding=self.l5_pool_pad,
            )
        )

        # FC layer 6
        print("Building layer 6...")
        alexnet_model.add(Flatten())
        alexnet_model.add(
            Dense(self.l6_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # FC layer 7
        print("Building layer 7...")
        alexnet_model.add(
            Dense(self.l7_num_neurons, activation=self.activation_function)
        )
        alexnet_model.add(Dropout(self.dropout))
        alexnet_model.add(BatchNormalization())

        # Output softmax layer
        print("Building output layer...")
        alexnet_model.add(
            Dense(self.output_neurons, activation=self.output_activation_function)
        )

        alexnet_model.compile(
            loss=self.loss_function, optimizer=self.optimizer, metrics=["accuracy"]
        )

        # Check model layers
        print("Model architecture:")
        print(alexnet_model.summary())

        return alexnet_model

    def train_model(self) -> None:
        """
        Trains model using the CIFAR-10 dataset, and caches training history.
        """
        print("Training model")
        training_history = self.model.fit(
            self.dataset.training_dataset,
            self.dataset.training_labels,
            epochs=self.num_epochs,
            batch_size=self.batch_size,
            shuffle=self.shuffle_training_data,
            validation_data=(
                self.dataset.validation_dataset,
                self.dataset.validation_labels,
            ),
        )

        self.train_status = True
        self.training_history = training_history

    def eval_model(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Returns categorical cross entropy results, predicted and true labels for testing dataset.
        """
        print("Evaluating model")
        if not self.train_status:
            raise ValueError("Model has not been trained.")

        # loss function results
        loss, accuracy = self.model.evaluate(
            self.dataset.testing_dataset, self.dataset.testing_labels
        )

        # Classification results, convert prob to class labels
        predictions = self.model.predict(self.dataset.testing_dataset)
        pred_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.dataset.testing_labels, axis=1)

        return loss, accuracy, pred_labels, true_labels

    def generate_confusion_matrix(
        self, pred_labels: np.ndarray, true_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Generates a confusion matrix.
        """
        print("Generating confusion matrix")
        matrix = confusion_matrix(true_labels, pred_labels)

        # Categories are indexed in sequential order
        classification_categories = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

        matrix_df = pd.DataFrame(
            matrix, index=classification_categories, columns=classification_categories
        )

        return matrix_df
