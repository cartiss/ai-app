"""Train, save and evaluate Animals Classification Model."""
import os

import cv2

import numpy as np
import tensorflow as tf

from typing import Tuple, Union
from pathlib import Path

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import models, layers, losses
from tensorflow.keras.models import load_model

from utils.data_processing.downloader import DataDownloader
from utils.data_processing.preprocessor import ImageDataPreprocessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'

BASE_DATASET_PATH = Path('models/animals_classification/dataset')


class Predictor:
    """Predictor"""

    def __init__(self, model_path: str):
        """
        Predictor initialization.

        :param model_path: Path to model to load it
        """
        self.model = load_model(model_path)
        self.dataset_loader = ImageDatasetLoader()

    def predict(self, image: Union[str, np.ndarray], input_size: Tuple[int, int, int]) -> str:
        """
        Predict an animal on an example.

        :param image_path: Path to image
        :param image_size: Image size
        :return: Predicted label
        """
        if type(image) == np.ndarray:
            image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        else:
            image = cv2.imread(image)

        image_array = cv2.resize(image, (input_size[0], input_size[1]))
        image_array = image_array.reshape(1, image_array.shape[0], image_array.shape[1], 3)

        predict = self.model.predict(image_array)
        idx = np.argmax(predict)

        return self.dataset_loader.get_label(int(idx))


class AnimalsClassificationModel:
    """Animals classification model."""

    def __init__(self, input_size: Tuple[int, int, int]) -> None:
        """
        Animals classification model initialization.

        :param input_size: Input size
        """
        self.INPUT_SIZE = input_size
        self.BATCH_SIZE = 32

        self.dataset_loader = ImageDatasetLoader()  # TODO: change path
        self.image_data_preprocessor = ImageDataPreprocessor()

        self.model = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def train(self, epochs: int) -> None:
        """
        Train animals classification model.

        :param epochs: Quantity of epochs
        """
        self.train_ds, self.val_ds = self.dataset_loader.parse_dataset(
            batch_size=self.BATCH_SIZE,
            image_width=self.INPUT_SIZE[0],
            image_height=self.INPUT_SIZE[1],
            validation_split=0.3
        )

        self.train_ds, self.test_ds = tf.keras.utils.split_dataset(self.train_ds, left_size=0.9, shuffle=True)

        self.model = models.Sequential([
            DenseNet121(weights="imagenet", include_top=False, input_shape=self.INPUT_SIZE),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(10)
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
            loss=losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )

        self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds)

        self.model.save(Path('trained_models') / 'animals_classification' / 'model.h5')

        self._evaluate()

    def _evaluate(self) -> None:
        """Evaluate the model."""
        if self.model and self.test_ds:
            self.model.evaluate(self.test_ds, batch_size=self.BATCH_SIZE)
        else:
            raise ValueError('Evaluating is failed. To evaluate the model, you have to train it.')


class ImageDatasetLoader:
    """Image dataset loader."""

    @staticmethod
    def get_label(idx: int) -> str:
        """
        Get label from text file by index.

        :param idx: Label's index
        :return: String of label
        """
        labels = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']

        return labels[idx]

    @staticmethod
    def parse_dataset(
        batch_size: int,
        image_height: int,
        image_width: int,
        validation_split: float
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Download and parse dataset folders to tf.data.Dataset

        :param batch_size: Batch size
        :param image_height: Image height
        :param image_width: Image width
        :param validation_split: Validation split (0-1)
        :return: Tuple with training and validation sets
        """
        if not BASE_DATASET_PATH.exists():
            data_handler = DataDownloader()
            data_handler.download_kaggle_dataset(kaggle_id='alessiocorrado99/animals10',
                                                 archive_name='animals10.zip',
                                                 extract_path=BASE_DATASET_PATH)

        ds_train = tf.keras.preprocessing.image_dataset_from_directory(
            BASE_DATASET_PATH / 'raw-img',
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=(image_height, image_width),
            shuffle=True,
            validation_split=0.3,
            seed=42,
            subset='training'
        )

        ds_val = tf.keras.preprocessing.image_dataset_from_directory(
            BASE_DATASET_PATH / 'raw-img',
            labels='inferred',
            label_mode='int',
            batch_size=batch_size,
            image_size=(image_height, image_width),
            shuffle=True,
            validation_split=validation_split,
            seed=42,
            subset='validation'
        )

        return ds_train, ds_val


def main() -> None:
    """Train Animals classification model."""
    image_size = (224, 224, 3)
    model = AnimalsClassificationModel(image_size)
    model.train(20)


if __name__ == '__main__':
    main()
