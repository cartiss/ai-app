import tensorflow as tf
from tensorflow.data import Dataset

from typing import Tuple


class ImageDataPreprocessor:
    """Image data preprocessor."""

    @staticmethod
    def split_dataset(
        data: Dataset,
        train_split: float,
        val_split: float = None,
        test_split: float = None,
        shuffle: bool = True,
    ) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Splits a tf.data.Dataset into train set, validation set and test set if specified. If you don't desire to
        form a test set, don't specify the test_split parameter and discard the third element in the returned tuple

        :param data: tf.data.Dataset to split
        :param train_split: size of training set. Must be a float number in range [0; 1]
        :param val_split: Optional. Size of the validation set. Must be a float number in range [0; 1]
        :param test_split: Optional. Size of the test set if needed.
        :param shuffle: whether to shuffle the dataset before splitting. Defaults to True
        :return: Tuple of three Dataset objects containing train set, validation set and test set in this order.
        If the test_split parameter was not specified, discard the third element in the tuple
        """
        if not test_split:
            test_split = 0.0

        if not val_split:
            val_split = 1.0 - train_split - test_split

        for i in (train_split, val_split, test_split):
            if i < 0 or i > 1:
                raise ValueError('All split sizes must be in range [0; 1] if specified')

        if test_split + val_split + train_split != 1:
            raise ValueError('The total size of splits must add up to 1.')

        if shuffle:
            data = data.shuffle(1000)

        train_ds, val_test_split = tf.keras.utils.split_dataset(data, left_size=train_split)
        val_proportional_split_size = val_split / (1 - test_split)
        val_ds, test_ds = tf.keras.utils.split_dataset(val_test_split, left_size=val_proportional_split_size)

        return train_ds, val_ds, test_ds

    # TODO: augment_underrepresented_classes()
    # @staticmethod
    # def augment_underrepresented_classes(data: Dataset):
    #     pass
