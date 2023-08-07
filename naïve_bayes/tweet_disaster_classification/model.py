"""Naive Bayes model which predicts whether the tweet is about disastrous events based on its text."""
import json
import os
import sys

import pandas as pd

from naïve_bayes.data_downloader import DataHandler
from naïve_bayes.text_formatter import TweetTextFormatter


class NaiveBayesModel:
    """Class for Naive Bayes model determining whether a tweet is about a disaster based on its text."""

    def __init__(self) -> None:
        """Initialize class fields."""
        self.freq_dict = {}
        self.class_freq = {0: 0, 1: 0}

    def train(self, text_samples: pd.Series, labels: pd.Series) -> None:
        """
        Update the model's parameters with values from the training data.

        :param text_samples: pd.Series containing processed text samples.
        :param labels: pd.Series containing labels for the text samples with the corresponding index.
        """
        # calculate class frequencies
        self.class_freq[0] = len(labels.loc[labels == 0])
        self.class_freq[1] = len(labels.loc[labels == 1])

        # calculate occurrences of each word in the training data
        for _, (text, label) in pd.concat([text_samples, labels], axis=1).iterrows():
            for word in text:
                if word not in self.freq_dict:
                    self.freq_dict[word] = {0: 0, 1: 0}

                match label:
                    case 0:
                        self.freq_dict[word][0] += 1
                    case 1:
                        self.freq_dict[word][1] += 1

    def predict(self, text_samples: pd.Series) -> pd.Series:
        """
        Predict class labels for each example in the given processed text samples and returns
        a pd.Series with predictions for text samples with the corresponding index.
        :param text_samples: pd.Series with tweet texts for prediction.
        """
        if not isinstance(text_samples, pd.Series):
            text_samples = pd.Series(text_samples)

        predictions = []

        for text in text_samples:
            prob_dict = {'0': 1.0, '1': 1.0}
            for word in text.split(' '):
                if word in self.freq_dict:
                    for i in ['0', '1']:
                        prob_dict[i] *= self.freq_dict[word][i] / self.class_freq[i]

            for i in ['0', '1']:
                prob_dict[i] *= self.class_freq[i]

            predictions.append(max(prob_dict, key=prob_dict.get))

        return pd.Series(predictions, index=text_samples.index)

    @staticmethod
    def read_dataset() -> pd.DataFrame:
        """
        Read dataset, if not exists - download it.

        :return: Dataset dataframe
        """
        dataset_path = os.path.normpath('naïve_bayes/datasets/tweet_disaster/')

        try:
            data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), index_col='id')
        except FileNotFoundError:
            data_handler = DataHandler()
            api = data_handler.kaggle_authenticate()
            data_handler.download_kaggle_competition(api, 'nlp-getting-started')
            data_handler.extract_dataset('nlp-getting-started.zip', dataset_path)
            data = pd.read_csv(os.path.join(dataset_path, 'train.csv'), index_col='id')

        return data

    def export_parameters_to_json(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Export current model parameters to .json files.

        :param class_freq_path: the path to the .json file to store the dictionary with class frequencies.
        :param words_freq_path: the path to the .json file to store the dictionary with frequencies for each word.
        """
        json_freq_dict = json.dumps(self.freq_dict)
        json_class_freq = json.dumps(self.class_freq)

        try:
            with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), class_freq_path)),
                      mode='w', encoding='utf-8') as file:
                file.write(json_class_freq)
            with open(os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__), words_freq_path)),
                      mode='w', encoding='utf-8') as file:
                file.write(json_freq_dict)
        except FileNotFoundError:
            print('File not found. Export terminated.')
            sys.exit(1)
        except OSError:
            print('Error opening the file. Export terminated.')
            sys.exit(1)

    def import_parameters(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Import model parameters from .json files.

        :param class_freq_path: the path to the .json file with the dictionary with class frequencies.
        :param words_freq_path: the path to the .json file with the dictionary with frequencies for each word.
        """
        try:
            with open(os.path.join('naïve_bayes', 'tweet_disaster_classification', class_freq_path),
                      mode='r', encoding='utf-8') as file:
                self.class_freq = json.loads(file.read())
            with open(os.path.join('naïve_bayes', 'tweet_disaster_classification', words_freq_path),
                      mode='r', encoding='utf-8') as file:
                self.freq_dict = json.loads(file.read())
        except FileNotFoundError:
            print('File not found. Import terminated.')
            sys.exit(1)
        except OSError:
            print('Error opening the file. Import terminated.')
            sys.exit(1)


def main() -> None:
    """Train Tweet Disaster Classification model and export parameters to JSON."""
    model = NaiveBayesModel()
    data = model.read_dataset()

    formatted_text_samples = TweetTextFormatter().process_text(text=data['text'])

    model.train(text_samples=formatted_text_samples, labels=data['target'])
    model.export_parameters_to_json(class_freq_path='class_freq.json', words_freq_path='words_freq.json')


if __name__ == '__main__':
    main()
