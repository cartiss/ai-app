from typing import List

import pandas as pd
from naive_bayes.sentiment_analysis.model import SentimentAnalysisModel
from naive_bayes.tweet_disaster_classification.model import NaiveBayesModel


def make_sentiment_prediction(text) -> pd.Series:
    """
    Predict the positivity/negativity of a tweet.

    :param text: tweet entered by the user
    :return: pd.Series of predictions
    """
    sentiment_analysis_model = SentimentAnalysisModel('trained_models/sentiment_analysis/model.json')
    parameters = sentiment_analysis_model.import_params()
    text = pd.Series(text)
    return sentiment_analysis_model.predict(text, parameters)


def make_disaster_prediction(text) -> pd.Series:
    """
    Predict whether a tweet is about a disaster or not.

    :param text: tweet entered by the user
    :return: pd.Series of predictions
    """
    disaster_classification_model = NaiveBayesModel()
    disaster_classification_model.import_parameters('class_freq.json', 'words_freq.json')
    return disaster_classification_model.predict(text)


def compile_results(prediction_sentiment, prediction_disaster) -> List[str]:
    """
    Compile result phrases to display to the user.

    :param prediction_sentiment: pd.Series of predictions
    :param prediction_disaster: pd.Series of predictions
    :return: List with phrases that are displayed to the user
    """
    results = []

    if prediction_sentiment[0] == 1:
        results.append("This tweet is positive.")
    else:
        results.append("This tweet is negative.")

    if prediction_disaster[0] == 1:
        results.append("This tweet is about disaster.")
    else:
        results.append("This tweet is not about disaster.")

    return results
