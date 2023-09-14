"""Run Flask application."""
from flask import render_template, request

from utils import make_sentiment_prediction, make_disaster_prediction, compile_results, make_homepage_queryset, \
    make_category_projects_queryset
from models import db, app

app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'
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
        prediction_sentiment = make_sentiment_prediction(text)
        prediction_disaster = make_disaster_prediction(text)

        results = compile_results(prediction_sentiment, prediction_disaster)

        return render_template('tweet_model.html', title="Tweet Model", results=results)

    return render_template('tweet_model.html', title="Tweet Model")


def main() -> None:
    """Run server."""
    with app.app_context():
        db.create_all()

    app.run(debug=True, host='0.0.0.0')


if __name__ == "__main__":
    main()
