"""Run Flask application."""
import pathlib
import sys
from decouple import config
import numpy as np

from flask import render_template, request



PARENT_FOLDER = str(pathlib.Path(__file__).parent.parent.absolute())
sys.path.insert(0, PARENT_FOLDER)
from models.animals_classification.model import Predictor

# flake8: noqa
from logic import make_sentiment_prediction, make_disaster_prediction, compile_results, make_homepage_queryset, \
    make_category_projects_queryset
# flake8: noqa
from web.models import db, app
from models.naive_bayes.sentiment_analysis.model import SentimentAnalysisModel
from models.naive_bayes.tweet_disaster_classification.model import NaiveBayesModel

sentiment_analysis_model = SentimentAnalysisModel('trained_models/sentiment_analysis/model.json')
parameters = sentiment_analysis_model.import_params()
predictor = Predictor(pathlib.Path('trained_models') / 'animals_classification' / 'animals_model.h5')

disaster_classification_model = NaiveBayesModel()
disaster_classification_model.import_parameters('class_freq.json', 'words_freq.json')

app.config['SECRET_KEY'] = config('SECRET_KEY')
app.config['UPLOAD_FOLDER'] = 'static'


@app.route("/")
def index() -> str:
    """Render home page."""
    category_project_dict = make_homepage_queryset()

    return render_template('index.html', category_project_dict=category_project_dict)


@app.route("/<category_name>")
def category_projects(category_name) -> str:
    """Render page with specific category of project."""
    category, projects = make_category_projects_queryset(category_name)

    return render_template('filtered_projects.html', category=category, projects=projects)


@app.route("/tweet-model/", methods=["GET", "POST"])
def tweet_model() -> str:
    """Render tweet model page."""
    if request.method == 'POST':
        text = request.form['tweet']
        prediction_sentiment = make_sentiment_prediction(text, sentiment_analysis_model, parameters)
        prediction_disaster = make_disaster_prediction(text, disaster_classification_model)

        results = compile_results(prediction_sentiment, prediction_disaster)

        return render_template('tweet_model.html', title="Tweet Model", results=results)

    return render_template('tweet_model.html', title="Tweet Model")


@app.route("/animals-model/", methods=["GET", "POST"])
def animals_model() -> str:
    """Render tweet model page."""
    if request.method == 'POST':
        image_bytes = request.files['image'].read()
        image_numpy = np.frombuffer(image_bytes, np.uint8)

        animal_prediction = predictor.predict(image_numpy, (224, 224, 3))

        return render_template('animals_model.html', title="Animals Model", result=animal_prediction)

    return render_template('animals_model.html', title="Animals Model")


def main() -> None:
    """Run server."""
    with app.app_context():
        db.create_all()

    app.run(debug=True, host='0.0.0.0')


if __name__ == "__main__":
    main()
