"""Naive Bayes model which predicts whether the tweet is about disastrous events based on its text."""
import json

import pandas as pd


class NaiveBayesModel:
    """Class for Naive Bayes model determining whether a tweet is about a disaster based on its text."""

    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initializes empty parameters and the dataset.
        :param data: pd.DataFrame with the training data. Must contain the following columns:
            - stemmed: contains a dictionary of stemmed words of the clean text of the tweet;
            - target: contains the label for each example;
        """
        self.data = data
        self.freq_dict = {}
        self.class_freq = {}

    def calculate_class_freq(self) -> None:
        """Calculates the amount of examples with each class label in the training data."""
        self.class_freq[0] = len(self.data.loc[self.data.target == 0])
        self.class_freq[1] = len(self.data.loc[self.data.target == 1])

    def generate_freq_dict(self) -> None:
        """Calculates the amount of occurrences of each word for each class label in the training data."""
        for _, row in self.data.iterrows():
            for word in row['processed_text']:
                if word not in self.freq_dict:
                    self.freq_dict[word] = {0: 0, 1: 0}

                match row['target']:
                    case 0:
                        self.freq_dict[word][0] += 1
                    case 1:
                        self.freq_dict[word][1] += 1

    def train(self) -> None:
        """Updates the param dictionaries with values from the training data."""
        self.calculate_class_freq()
        self.generate_freq_dict()

    def predict(self, data_frame: pd.DataFrame) -> None:
        """
        Predicts class labels for each example in the given data x and appends the 'predictions' column
        with labels to it.
        :param data_frame: pd.DataFrame with tweet texts for prediction. Must contain the following columns:
            - stemmed: contains a dictionary of stemmed words of the clean text of the tweet.
        """
        predictions = []

        for _, row in data_frame.iterrows():
            prob_dict = {0: 1.0, 1: 1.0}
            for word in row['processed_text']:
                if word in self.freq_dict:
                    for i in [0, 1]:
                        prob_dict[i] *= self.freq_dict[word][i] / self.class_freq[i]

            for i in [0, 1]:
                prob_dict[i] *= self.class_freq[i]

            predictions.append(max(prob_dict, key=prob_dict.get))

        data_frame['predictions'] = predictions

    def export_parameters_to_json(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Exports current parameters to .json files.
        :param class_freq_path: the path to the .json file to store the dictionary with class frequencies.
        :param words_freq_path: the path to the .json file to store the dictionary with frequencies for each word.
        """
        json_freq_dict = json.dumps(self.freq_dict)
        json_class_freq = json.dumps(self.class_freq)

        try:
            with open(class_freq_path, mode='w', encoding='utf-8') as file:
                file.write(json_class_freq)
        except FileNotFoundError:
            print('File not found. Export terminated.')
            return
        except OSError:
            print('Error opening the file. Export terminated.')

        # TODO: exception handling
        with open(words_freq_path, mode='w', encoding='utf-8') as file:
            file.write(json_freq_dict)

    def import_parameters(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Imports current parameters from .json files.
        :param class_freq_path: the path to the .json file with the dictionary with class frequencies.
        :param words_freq_path: the path to the .json file with the dictionary with frequencies for each word.
        """

        with json.open(class_freq_path, mode='r', encoding='utf-8') as file:
            self.class_freq = json.loads(file.read())

        with json.open(words_freq_path, mode='r', encoding='utf-8') as file:
            self.freq_dict = json.loads(file.read())
