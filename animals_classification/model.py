import logging
import os
import sys
from typing import List, Tuple

import cv2

import numpy as np

from keras.applications.vgg16 import VGG16
from keras import models, layers, losses
from keras.utils import to_categorical


class AbstractDatasetParser:

    def __init__(self, dataset_path: str) -> None:
        self.BASE_DATASET_PATH = os.path.abspath(os.path.join('animals_classification', dataset_path))

    def parse_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        raise NotImplementedError()


class AnimalDatasetParser(AbstractDatasetParser):

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path)
        self.IMAGE_WIDTH = 150
        self.IMAGE_HEIGHT = 150
        self.EXAMPLES_QUANTITY = 5400
        self.CATEGORIES_QUANTITY = 90
        self.TEST_DISTRIBUTION = 20

    def _get_labels_list(self) -> List[str]:
        try:
            with open(os.path.join(self.BASE_DATASET_PATH, 'name of the animals.txt')) as labels_file:
                labels = labels_file.read().splitlines()
        except FileNotFoundError:
            logging.error('Incorrect path to "name of the animals.txt" file!')
            sys.exit(1)

        return labels

    def parse_dataset(self) -> Tuple[np.array, np.array, np.array, np.array]:
        labels = self._get_labels_list()
        train_x = np.zeros((4320, self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        train_y = np.zeros(4320)
        test_x = np.zeros((1080, self.IMAGE_WIDTH, self.IMAGE_HEIGHT))
        test_y = np.zeros(1080)

        train_examples_counter = 0
        test_examples_counter = 0

        for label_i, label in enumerate(labels):
            label_folder = os.path.join(self.BASE_DATASET_PATH, 'animals', label)

            if os.path.exists(label_folder):
                for file_i, filename in enumerate(os.listdir(label_folder)):
                    example_path = os.path.join(label_folder, filename)

                    if os.path.isfile(example_path):
                        image = cv2.resize(
                            cv2.imread(example_path, cv2.IMREAD_GRAYSCALE),
                            (self.IMAGE_WIDTH, self.IMAGE_HEIGHT)) / 255.

                        if file_i % 5 == 0:
                            test_x[test_examples_counter] = image
                            test_y[test_examples_counter] = label_i
                            test_examples_counter += 1
                        else:
                            train_x[train_examples_counter] = image
                            train_y[train_examples_counter] = label_i
                            train_examples_counter += 1

            else:
                logging.error('Dataset folder is incorrect!')
                sys.exit(1)

        # if train_examples_counter != self.EXAMPLES_QUANTITY:
        #     logging.error('Dataset is incorrect!')
        #     sys.exit(1)

        train_y = to_categorical(train_y, num_classes=self.CATEGORIES_QUANTITY)
        test_y = to_categorical(test_y, num_classes=self.CATEGORIES_QUANTITY)
        print(test_y)

        return train_x, test_x, train_y, test_y


class AnimalsClassificationModel:
    """Animals classification model."""

    def __init__(self, dataset_parser: AbstractDatasetParser = AnimalDatasetParser('dataset')):
        self.dataset_parser = dataset_parser

    # def _get_label(self, label_id) -> str:
    #     return self.dataset_parser.get_labels_list()[label_id]

    def train(self):
        train_x, test_x, train_y, test_y = self.dataset_parser.parse_dataset()

        print(train_x.shape)

        base_model = VGG16(weights="imagenet", include_top=False, input_shape=train_x[0].shape)
        base_model.trainable = False

        model = models.Sequential([
            base_model,
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(90)
        ])

        model.compile(
            optimizer='adam',
            loss=losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        model.fit(train_x, train_y, epochs=20, validation_split=0.2, batch_size=32)

        model.evaluate(test_x, test_y, batch_size=32)

    def predict(self):
        pass

    def evaluate(self):
        pass


def main() -> None:
    dataset_parser = AnimalDatasetParser('dataset')
    model = AnimalsClassificationModel(dataset_parser)
    model.train()


if __name__ == '__main__':
    main()