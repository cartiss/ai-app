"""Naive Bayes model which predicts whether the tweet is about disastrous events based on its text."""
import json

import pandas as pd


class NaiveBayesModel:
    """Class for Naive Bayes model determining whether a tweet is about a disaster based on its text."""

    def __init__(self) -> None:
        """Initialize class fields."""
        self.text_samples = None
        self.labels = None
        self.freq_dict = None
        self.class_freq = None

    def train(self, text_samples: pd.Series, labels: pd.Series) -> None:
        """
        Update the model's parameters with values from the training data.

        :param text_samples: pd.Series containing processed text samples.
        :param labels: pd.Series containing labels for the text samples with the corresponding index.
        """
        # calculate class frequencies
        self.class_freq[0] = len(self.labels.loc[self.labels == 0])
        self.class_freq[1] = len(self.labels.loc[self.labels == 1])

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

        for _, text in text_samples:
            prob_dict = {0: 1.0, 1: 1.0}
            for word in text:
                if word in self.freq_dict:
                    for i in [0, 1]:
                        prob_dict[i] *= self.freq_dict[word][i] / self.class_freq[i]

            for i in [0, 1]:
                prob_dict[i] *= self.class_freq[i]

            predictions.append(max(prob_dict, key=prob_dict.get))

        return pd.Series(predictions, index=text_samples.index)

    def export_parameters_to_json(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Export current model parameters to .json files.

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

        try:
            with open(words_freq_path, mode='w', encoding='utf-8') as file:
                file.write(json_freq_dict)
        except FileNotFoundError:
            print('File not found. Export terminated.')
            return
        except OSError:
            print('Error opening the file. Export terminated.')

    def import_parameters(self, class_freq_path: str, words_freq_path: str) -> None:
        """
        Import model parameters from .json files.
        :param class_freq_path: the path to the .json file with the dictionary with class frequencies.
        :param words_freq_path: the path to the .json file with the dictionary with frequencies for each word.
        """
        try:
            with json.open(class_freq_path, mode='r', encoding='utf-8') as file:
                self.class_freq = json.loads(file.read())
        except FileNotFoundError:
            print('File not found. Import terminated.')
            return
        except OSError:
            print('Error opening the file. Import terminated.')

        try:
            with json.open(words_freq_path, mode='r', encoding='utf-8') as file:
                self.freq_dict = json.loads(file.read())
        except FileNotFoundError:
            print('File not found. Import terminated.')
            return
        except OSError:
            print('Error opening the file. Import terminated.')
