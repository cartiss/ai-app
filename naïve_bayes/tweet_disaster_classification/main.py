"""
Functions necessary for setting up and running the Naive Bayes model.
Call these from the server application
"""

import os

import pandas as pd

from naïve_bayes.tweet_disaster_classification.text_formatter import TweetTextFormatter
from naïve_bayes.tweet_disaster_classification.model import NaiveBayesModel


def train_model(filepath: str, json_class_path: str, json_words_path: str, index_col: str = 'id') -> None:
    """
    Generates model parameters from the Kaggle 'Natural Language Processing with Disaster Tweets' dataset
    and exports them in .json format.
    !WARNING: do not call this from a website view. This function is supposed to be run once.
    :param filepath: path to the .csv file with training data.
    :param json_class_path: path to the .json file to export class frequencies into.
    :param json_words_path: path to the .json file to export words frequencies into.
    :param index_col: the name of the index column in the dataset.
    """
    data_train = pd.read_csv(filepath, index_col=index_col)
    formatted_text_samples = TweetTextFormatter.process_text(data_train['text'])
    labels = data_train['target']

    model = NaiveBayesModel()
    model.train(text_samples=formatted_text_samples, labels=labels)
    model.export_parameters_to_json(class_freq_path=json_class_path, words_freq_path=json_words_path)


def import_model(json_class_path: str, json_words_path: str) -> NaiveBayesModel:
    """
    Sets up and an instance of the NaiveBayesModel class with model parameters specified in the .json files.
    !WARNING: do not call this from a website view. This function is supposed to be run once.
    :param json_class_path: path to the .json file to import class frequencies from.
    :param json_words_path: path to the .json file to import words frequencies from.
    :return: NaiveBayesModel instance with imported parameters.
    """
    model = NaiveBayesModel()
    if os.path.exists(json_class_path) and os.path.exists(json_words_path):
        model.import_parameters(class_freq_path=json_class_path, words_freq_path=json_words_path)
    else:
        # TODO: exception handling
        pass

    return model
