import logging
from typing import List

import pandas as pd
from naïve_bayes.sentiment_analysis.model import SentimentAnalysisModel
from naïve_bayes.tweet_disaster_classification.model import NaiveBayesModel
from models import db, Project, Category


def make_sentiment_prediction(text) -> pd.Series:
    """
    Predict the positivity/negativity of a tweet.

    :param text: tweet entered by the user
    :return: pd.Series of predictions
    """
    sentiment_analysis_model = SentimentAnalysisModel('naïve_bayes/sentiment_analysis/model.json')
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

    logging.error(results)

    return results


def make_homepage_queryset() -> dict:
    """
        Make queryset of categories and their projects.

        :return: dict with categories and their projects
    """
    categories_and_projects = db.session.query(Category, Project).join(Project,
                                                                       Category.id == Project.category_id).all()
    category_project_dict = {}

    for category, project in categories_and_projects:
        if category not in category_project_dict:
            category_project_dict[category] = []
        category_project_dict[category].append(project)

    return category_project_dict


def make_category_projects_queryset(category_name):
    """
        Make queryset of category and projects of this category.

        :param category_name: str with name of category
        :return: Category object and list of projects
    """
    prepared_category_name = prepare_category_name(category_name)
    category = Category.query.filter_by(name=str(prepared_category_name)).first()
    if category:
        projects = Project.query.filter_by(category_id=category.id).all()
        return category, projects
    else:
        return None, []


def prepare_category_name(category_name) -> str:
    """
        Format the name of category.

        :param category_name: str with name of category
        :return: formatted str
    """
    if len(category_name) == 3:
        return category_name.upper()
    else:
        return category_name.capitalize()
