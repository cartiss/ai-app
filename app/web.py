"""Run Flask application."""

from flask import Flask, render_template, request
from models import db
from models import Project, Tag, ProjectTag
from utils import make_sentiment_prediction, make_disaster_prediction, compile_results

app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost/aiprojectsdb'
db.init_app(app)


@app.route("/")
def index() -> None:
    """Render home page."""
    return render_template('index.html')


@app.route("/tweet-model/", methods=["GET", "POST"])
def tweet_model() -> str:
    """Render tweet model page."""
    if request.method == 'POST':
        text = request.form['tweet']
        prediction_sentiment = make_sentiment_prediction(text)
        prediction_disaster = make_disaster_prediction(text)

        results = compile_results(prediction_sentiment, prediction_disaster)

        return render_template('tweet_model.html', title="Tweet Model", results=results)

    return render_template('tweet_model.html', title="Tweet Model")


def main() -> None:
    """Run server."""
    app.run(debug=True, host='0.0.0.0')


if __name__ == "__main__":
    main()
