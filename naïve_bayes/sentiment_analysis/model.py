"""Train tweet sentiment analysis model."""
import json
import logging
import os
import sys
from zipfile import ZipFile

from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd

from typing import Tuple, Dict
from naïve_bayes.text_formatter import TweetTextFormatter


class SentimentAnalysisModel:
    """Naïve bayes sentiment analysis model"""

    def __init__(self, file_path):
        """
        Model initialization and preprocess dataset.

        :param file_path: Path to JSON parameters file
        """
        self.word_freq = {}
        self.class_freq = {}
        self.text_formatter = TweetTextFormatter()
        self.file_path = file_path

        self.train_data, self.test_data = self._preprocess_data()

    @staticmethod
    def _download_dataset():
        dataset_path = 'naïve_bayes/datasets'

        if not os.path.exists(dataset_path):
            os.mkdir(dataset_path)

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('kazanova/sentiment140')
        zf = ZipFile('sentiment140.zip')
        zf.extractall(path='naïve_bayes/datasets/')
        zf.close()

    def _read_dataset(self) -> pd.DataFrame:
        try:
            data = pd.read_csv(os.path.normpath('naïve_bayes/datasets/training.1600000.processed.noemoticon.csv'),
                               encoding='latin-1', names=['sentiment', 'id', 'date', 'flag', 'user', 'text'])
        except FileNotFoundError:
            self._download_dataset()
            data = self._read_dataset()

        return data

    def _preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess dataset before training."""
        data = self._read_dataset()

        data = data[['sentiment', 'text']]

        data.loc[data['sentiment'] == 4, 'sentiment'] = 1

        data['text'] = self.text_formatter.process_text(data['text'])

        test_data = data[:200000]
        train_data = data[200000:]

        return train_data, test_data

    @staticmethod
    def count_tweets(data_frame):
        """Count word frequency per each class."""
        freqs_counts = {}

        for _, row in data_frame.iterrows():
            for word in row['text']:
                if word not in freqs_counts.keys():
                    freqs_counts[word] = {0: 0, 1: 0}
                freqs_counts[word][row['sentiment']] += 1

        return freqs_counts

    def train(self) -> None:
        """Train model and export parameters to JSON file."""
        word_freq = self.count_tweets(self.train_data)

        class_freq = {0: len(self.train_data['sentiment'].loc[self.train_data['sentiment'] == 0]),
                      1: len(self.train_data['sentiment'].loc[self.train_data['sentiment'] == 1])}

        self._save(word_freq, class_freq)

    def _save(self, word_freq: Dict[str, Dict[int, int]], class_freq: Dict[int, int]) -> None:
        """
        Save parameters to JSON file.

        :param word_freq: Dict of words frequency per class
        :param class_freq: Dict of quantity of tweets per class
        :return:
        """
        general_dict = {'word_freq': word_freq, 'class_freq': class_freq}

        try:
            with open(self.file_path, 'w') as file:
                file.write(json.dumps(general_dict))
        except FileNotFoundError:
            logging.error('File model')

    def import_params(self) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        """
        Import parameters from JSON file.

        :param file_path: Path to file with parameters
        :return: Tuple with 2 dicts:
                - word_freq (words frequency per class)
                - class_freq (quantity of tweets per class)
        """
        try:
            trained_model = open(self.file_path)
        except FileNotFoundError:
            logging.error('File with model parameters has not been found, please train model!')
            sys.exit(1)
        except OSError as error:
            logging.error(f'OS Error: {error}')
            sys.exit(1)

        parameters = json.load(trained_model)

        try:
            return parameters['word_freq'], parameters['class_freq']
        except KeyError:
            logging.error('Incorrect content in parameters file')
            sys.exit()

    @staticmethod
    def predict(tweets: pd.Series, model_params: Tuple[Dict[str, Dict[str, int]], Dict[str, int]]) -> pd.Series:
        """
        Predict is tweet positive/negative.

        :param tweets: pd.Series of tweets with text
        :param model_params: Imported parameters from JSON file
        :return: pd.Series of predictions
        """
        predictions = []

        word_freq, class_freq = model_params

        for text in tweets:
            prob_dict = {0: 1.0, 1: 1.0}

            for word in text:
                if word in word_freq.keys():
                    for i in range(2):
                        prob_dict[i] *= word_freq[word][str(i)] / class_freq[str(i)]

            for i in range(2):
                prob_dict[i] *= class_freq[str(i)]

            predictions.append(max(prob_dict, key=prob_dict.get))

        return pd.Series(predictions, index=tweets.index)

    def evaluate(self) -> float:
        """Evaluate model accuracy."""
        predictions = self.predict(self.test_data['text'], self.import_params())
        miss_count = 0
        correct_count = 0

        for predict, sentiment in zip(predictions, self.test_data['sentiment']):
            if predict != sentiment:
                miss_count += 1
            else:
                correct_count += 1

        accuracy = correct_count * 100 / len(self.test_data)

        return accuracy


def main() -> None:
    model = SentimentAnalysisModel(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), 'model.json')))
    model.train()
    print(model.evaluate())


if __name__ == '__main__':
    main()
