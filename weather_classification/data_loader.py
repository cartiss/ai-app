import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import image_dataset_from_directory

from pathlib import Path
from typing import Tuple

from utils.data_downloader import DataHandler


class DataLoader:
    """
    Contains methods for downloading image datasets from Kaggle and converting them to tensorflow and pytorch formats.
    """

    @staticmethod
    def download_img_dataset(
            data_storage_path: str,
            dataset_id: str,
    ):
        """
        Downloads and unpacks a Kaggle dataset into the specified directory.

        :param data_storage_path: path to store the dataset
        :param dataset_id: kaggle dataset id. Defaults to the weather dataset
        """
        path = Path(data_storage_path)
        api = DataHandler.kaggle_authenticate()
        DataHandler.download_kaggle_dataset(api, dataset_id=dataset_id)
        if not path.is_dir():
            Path.mkdir(path)
        DataHandler.extract_dataset(archive_name='weather_dataset', dataset_path=path)

    @staticmethod
    def load_dataset_to_tf_format(
            data_storage_path: str = 'data',
            subset: str = None,
            label_mode: str = 'inferred',
            color_mode: str = 'rgb',
            download: bool = False,
            kaggle_dataset_id: str = None,
    ) -> Dataset:
        """
        Returns the dataset in the specified folder in tensorflow.data.Dataset format.
        Expects the following directory structure if the subset parameter is not specified:
        \---data
            +---class_1
            +---class_2
            +---...
            \---class_n

        Expects the following directory structure if the subset parameter is specified:
        \---data
            \---subset_name
                +---class_1
                +---class_2
                +---...
                \---class_n

        :param data_storage_path: path to the dataset
        :param subset: expected values: 'train', 'validation', 'test'. If specified, the dataset is generated from
        the files in the subset directory. Defaults to None
        :param label_mode: if 'inferred', the labels are generated to be used with 'categorical_crossentropy' loss.
        If 'int', the labels are generated to be used with 'sparse_categorical_crossentropy' loss.
        If 'binary', the labels are generated to be used with 'binary_crossentropy' loss
        Defaults to 'inferred'
        :param color_mode: One of "grayscale", "rgb", "rgba". Whether the images will be converted to have
        1, 3, or 4 channels. Defaults to 'rgb'
        :param download: whether to download the dataset before converting to Dataset format. Defaults to False
        :param kaggle_dataset_id: if download is True, the Kaggle dataset id. Defaults to None
        :return: tensorflow.data.Dataset object with images from the specified directory
        """
        if download:
            DataLoader.download_img_dataset(data_storage_path=data_storage_path, kaggle_dataset_id=kaggle_dataset_id)

        if not subset:
            dataset = image_dataset_from_directory(
                directory=data_storage_path,
                labels=label_mode,
                color_mode=color_mode
            )
            return dataset
        else:
            dataset = image_dataset_from_directory(
                diretory=str(Path(data_storage_path) / subset),
                labels=label_mode,
                color_mode=color_mode
            )
            return dataset

    # TODO: def load_dataset_to_pytorch_format():


if __name__ == '__main__':
    DataLoader.load_dataset()
