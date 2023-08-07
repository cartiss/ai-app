"""Run Flask application."""

from flask import Flask, render_template, request

import pandas as pd

from naïve_bayes.sentiment_analysis.model import SentimentAnalysisModel
from naïve_bayes.tweet_disaster_classification.model import NaiveBayesModel

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'


@app.route("/")
def index() -> None:
    """Render home page."""
    return render_template('index.html')


@app.route("/tweet-model", methods=["GET", "POST", ])
def tweet_model() -> None:
    """Render tweet model page."""
    if request.method == 'POST':
        text = request.form['tweet']
        results = []

        sentiment_analysis_model = SentimentAnalysisModel('naïve_bayes/sentiment_analysis/model.json')
        parameters = sentiment_analysis_model.import_params()
        text = pd.Series(text)
        prediction_sentiment = sentiment_analysis_model.predict(text, parameters)


        disaster_classification_model = NaiveBayesModel()
        disaster_classification_model.import_parameters('class_freq.json', 'words_freq.json')
        prediction_disaster = disaster_classification_model.predict(text)

        if prediction_sentiment[0] == 1:
            results.append("This tweet is positive.")
        else:
            results.append("This tweet is negative.")

        if prediction_disaster[0] == 1:
            results.append("This tweet is about disaster.")
        else:
            results.append("This tweet is not about disaster.")

        return render_template('tweet_model.html', title="Tweet Model", results=results)

    return render_template('tweet_model.html', title="Tweet Model")


def main() -> None:
    """Run server."""
    app.run(debug=True, host='0.0.0.0')


if __name__ == "__main__":
    main()
