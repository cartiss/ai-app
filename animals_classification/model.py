"""Train, save and evaluate Animals Classification Model."""
import os
from typing import List, Tuple

import cv2

import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras import models, layers, losses
from keras.layers import BatchNormalization
from keras.utils import to_categorical
from tensorflow.python.keras.models import load_model
from sklearn.model_selection import train_test_split


class AnimalDatasetParser:
    """Animal Dataset Parser."""

    def __init__(self, dataset_path: str, test_distribution: float, image_width: int, image_height: int) -> None:
        """
        Animal dataset parser initialization.

        :param dataset_path: Path to dataset folder
        :param test_distribution: Train/test split percent (0-1)
        :param image_width: Input image width
        :param image_height: Input image height
        """
        self.BASE_DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset_path)
        self.IMAGE_WIDTH = image_width
        self.IMAGE_HEIGHT = image_height
        self.TEST_DISTRIBUTION = test_distribution

    def get_labels_list(self) -> List[str]:
        """Get list of labels from txt file."""
        try:
            with open(os.path.join(self.BASE_DATASET_PATH, 'name of the animals.txt')) as labels_file:
                labels = labels_file.read().splitlines()
        except FileNotFoundError:
            raise FileNotFoundError('Incorrect path to "name of the animals.txt" file!')

        return labels

    def image_to_array(self, image_path: str) -> np.array:
        """
        Transform image to numpy array with required standards.

        :param image_path: Image path
        :return: Image as numpy array
        """
        return cv2.resize(cv2.imread(image_path), (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)) / 255.

    def _parse_folders(self) -> Tuple[np.array, np.array]:
        """
        Parse dataset images from folders.

        :return: Tuple: (Array of images, array of labels)
        """
        labels = self.get_labels_list()

        full_dataset_x = []
        full_dataset_y = []

        for label_i, label in enumerate(labels):
            label_folder = os.path.join(self.BASE_DATASET_PATH, 'animals', label)

            if not os.path.exists(label_folder):
                raise FileNotFoundError(f'Dataset doesn\'t have {label}\'s folder!')

            for filename in os.listdir(label_folder):
                example_path = os.path.join(label_folder, filename)

                if os.path.isfile(example_path):
                    image = self.image_to_array(example_path)

                    full_dataset_x.append(image)
                    full_dataset_y.append(label_i)

        return np.array(full_dataset_x), np.array(full_dataset_y)

    def parse_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Parse images, split to train/test, prepare dataset.

        :return: Tuple: (train examples, test examples, train labels, test labels)
        """
        full_dataset_x, full_dataset_y = self._parse_folders()

        train_x, test_x, train_y, test_y = train_test_split(full_dataset_x, full_dataset_y,
                                                            test_size=self.TEST_DISTRIBUTION,
                                                            random_state=42, shuffle=True)

        num_classes = len(self.get_labels_list())

        train_y = to_categorical(train_y, num_classes=num_classes)
        test_y = to_categorical(test_y, num_classes=num_classes)

        return train_x, test_x, train_y, test_y


class AnimalsClassificationModel:
    """Animals classification model."""

    def __init__(self):
        """Animals classification model initialization."""
        self.INPUT_SIZE = (299, 299, 3)
        self.train_x, self.test_x, self.train_y, self.test_y = None, None, None, None
        self.dataset_parser = AnimalDatasetParser('dataset', 0.2, self.INPUT_SIZE[0], self.INPUT_SIZE[1])
        self.model = None

    def prepare_data(self) -> None:
        """Prepare and split data from dataset folder."""
        self.train_x, self.test_x, self.train_y, self.test_y = self.dataset_parser.parse_dataset()

    def _get_label(self, label_id) -> str:
        """
        Get label by label ID.

        :param label_id: Label ID
        :return: Label
        """
        return self.dataset_parser.get_labels_list()[label_id]

    def train(self, epochs: int) -> models.Sequential:
        """
        Train animals classification model.

        :param epochs: Quantity of epochs.
        :return: Trained model
        """
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=self.INPUT_SIZE)

        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(3)
        ])

        model.compile(
            optimizer='adam',
            loss=losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.fit(self.train_x, self.train_y, epochs=epochs, validation_split=0.3, batch_size=32, shuffle=True)

        model.evaluate(self.test_x, self.test_y, batch_size=32)

        return model

    @staticmethod
    def save_model(model: models.Sequential, save_path: str) -> None:
        """
        Save model to file.

        :param model: Model to save
        :param save_path: Path where to save a model
        """
        model.save(save_path)

    def import_model(self, model_path: str) -> None:
        """
        Import pretrained model.

        :param model_path: Path to model
        """
        self.model = load_model(model_path, custom_objects={"BatchNormalization": BatchNormalization,
                                                            'Sequential': models.Sequential,
                                                            'Inception_V3': InceptionV3})

    def predict(self, image_path: str) -> str:  # TODO: check in which form image will come from website
        """
        Predict an animal on an example.

        :param image_path: Path to image
        :return: Predicted label
        """
        image_array = self.dataset_parser.image_to_array(image_path)
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 3)

        predict = self.model.predict(image_array)
        idx = np.argmax(predict)

        return self._get_label(idx)


def main() -> None:
    """Train Animals classification model."""
    model = AnimalsClassificationModel()
    model.prepare_data()
    trained_model = model.train(1)
    model.save_model(trained_model, 'animals_model')
    model.import_model('animals_model')
    test_prediction = model.predict('butterfly.jpg')
    print(test_prediction)


if __name__ == '__main__':
    main()
