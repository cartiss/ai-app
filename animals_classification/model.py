"""Train, save and evaluate Animals Classification Model."""
import cv2
import os

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import models, layers, losses
from typing import Tuple, List
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import load_model


class Predictor:
    """Predictor"""

    def __init__(self, model_path: str, image_size: Tuple[int, int, int]):
        """
        Predictor initialization.

        :param model_path: Path to model to load it
        :param image_size: Image size
        """
        self.INPUT_SIZE = image_size
        self.model = load_model(model_path, custom_objects={"BatchNormalization": BatchNormalization,
                                                            'Sequential': models.Sequential,
                                                            'Inception_V3': InceptionV3})
        self.image_processor = ImageProcessor()

    @staticmethod
    def _get_label(idx: int) -> str:
        """
        Get label from text file by index.

        :param idx: Label's index
        :return: String of label
        """
        dataset_loader = ImageDatasetLoader()

        return dataset_loader.get_labels_list()[idx]

    def predict(self, image_path: str) -> str:  # TODO: check in which form image will come from website
        """
        Predict an animal on an example.

        :param image_path: Path to image
        :return: Predicted label
        """
        image_array = self.image_processor.prepare_image(image_path, self.INPUT_SIZE)
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 3)

        predict = self.model.predict(image_array)
        idx = np.argmax(predict)

        return self._get_label(idx)


class AnimalsClassificationModel:
    """Animals classification model."""

    def __init__(self, input_size: Tuple[int, int, int]):
        """
        Animals classification model initialization.

        :param input_size: Input size
        """
        self.INPUT_SIZE = input_size
        self.dataset_loader = ImageDatasetLoader('dataset')  # TODO: change path
        self.model = None

    def save_model(self, save_path: str) -> None:
        """
        Save model to file.

        :param save_path: Path where to save a model
        """
        self.model.save(save_path)

    def train(self, epochs: int) -> None:
        """
        Train animals classification model.

        :param epochs: Quantity of epochs
        """
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=self.INPUT_SIZE)

        self.dataset_loader.parse_dataset()
        train_x, train_y = self.dataset_loader.get_prepared_dataset()

        self.model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(5)
        ])

        self.model.compile(
            optimizer='adam',
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.fit(train_x, train_y, epochs=epochs, validation_split=0.2, batch_size=8, shuffle=True)

        self.save_model('animals_model')

    def evaluate(self) -> None:
        """Evaluate the model."""
        if self.model:
            test_x, test_y = self.dataset_loader.get_prepared_dataset(test_set=True)
            self.model.evaluate(test_x, test_y, batch_size=8)
        else:
            raise ValueError('Evaluating is failed. To evaluate the model, you have to train it.')


class ImageProcessor:
    """Image processor."""

    @staticmethod
    def image_to_array(image_path: str) -> np.array:
        """
        Transform and normalize image to numpy array.

        :param image_path: Image path
        :return: Image as numpy array
        """
        return cv2.imread(image_path)

    @staticmethod
    def resize_image(image, image_size: Tuple[int, int, int]) -> np.array:
        """
        Resize image.

        :param image: np.array image
        :param image_size: Image properties to resize an image
        :return: Resized image
        """
        return cv2.resize(image, (image_size[0], image_size[1]))

    def prepare_image(self, image_path, image_size: Tuple[int, int, int]) -> np.array:
        """
        Prepare image for using: transforming, resizing.

        :param image_path: Image path
        :param image_size: Image properties to resize an image
        :return: Prepared image for using
        """
        image_array = self.image_to_array(image_path)
        resized_image = self.resize_image(image_array, image_size)

        return resized_image


class ImageDatasetLoader:
    """Image dataset loader."""

    def __init__(self, dataset_path: str = 'dataset', test_distribution: float = 0.1):
        """
        Image dataset loader initialization.

        :param dataset_path: Path to dataset folder
        :param test_distribution: Test set distribution
        """
        self.BASE_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)
        self.TEST_DISTRIBUTION = test_distribution

    def parse_dataset(self):
        

    def get_labels_list(self) -> List[str]:
        """
        Get list of labels from txt file.

        :return: List of labels
        """
        try:
            with open(os.path.join(self.BASE_DATASET_PATH, 'name of the animals.txt')) as labels_file:
                labels = labels_file.read().splitlines()
        except FileNotFoundError:
            raise FileNotFoundError('Incorrect path to "name of the animals.txt" file!')

        return labels


def main() -> None:
    """Train Animals classification model."""
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    image_size = (299, 299, 3)
    model = AnimalsClassificationModel(image_size)
    model.train(5)
    model.save_model('animals_classification/animals_model')

    predictor = Predictor('animals_model', image_size)
    test_prediction = predictor.predict('animals_classification/butterfly.jpg')
    print(test_prediction)


if __name__ == '__main__':
    main()
