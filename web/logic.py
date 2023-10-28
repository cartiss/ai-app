import logging
from typing import List, Tuple, Union

import pandas as pd
from web.models import db, Project, Category


def make_sentiment_prediction(text, sentiment_analysis_model, parameters) -> pd.Series:
    """
    Predict the positivity/negativity of a tweet.

    :param text: tweet entered by the user
    :param sentiment_analysis_model: object of SentimentAnalysisModel
    :param parameters: model's parameters
    :return: pd.Series of predictions
    """
    text = pd.Series(text)
    return sentiment_analysis_model.predict(text, parameters)


def make_disaster_prediction(text, disaster_classification_model) -> pd.Series:
    """
    Predict whether a tweet is about a disaster or not.

    :param text: tweet entered by the user
    :param disaster_classification_model: object of NaiveBayesModel
    :return: pd.Series of predictions
    """
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


def make_category_projects_queryset(category_name) -> Tuple[Union[Category, None], List[Project]]:
    """
    Make queryset of category and projects of this category.

    :param category_name: str with name of category
    :return: Category object and list of projects
    """
    category = Category.query.filter_by(name=str(category_name)).first()
    if category:
        projects = Project.query.filter_by(category_id=category.id).all()
        return category, projects

    return None, []
