import torch
import os

from naïve_bayes.data_downloader import DataHandler


class LicensePlateDetectionModel:
    def __init__(self):
        pass

    @staticmethod
    def download_dataset():
        handler = DataHandler()
        kaggle_api = handler.kaggle_authenticate()
        handler.download_kaggle_dataset(kaggle_api, 'andrewmvd/car-plate-detection')
        handler.extract_dataset('car-plate-detection.zip',
                                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset'))

    def train(self):
        self.download_dataset()

    def evaluate(self):
        pass


if __name__ == '__main__':
    model = LicensePlateDetectionModel()
    model.train()