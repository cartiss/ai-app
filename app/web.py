"""Run Flask application."""
import os

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc

from utils import make_sentiment_prediction, make_disaster_prediction, compile_results, make_homepage_queryset, make_category_projects_queryset
from models import db, app, Project, Tag, ProjectTag, Category

postgres_name = os.getenv('SQL_DB')
# app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'
# app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ai_db_user:Hgyfshbjs8482@localhost/aiprojectsdb'
app.config['UPLOAD_FOLDER'] = 'static'


# db = SQLAlchemy(app)


# class Project(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(255))
#     description = db.Column(db.Text)
#     category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
#     category = db.relationship('Category', backref='projects')
#     url = db.Column(db.String(255), unique=True)
#     image_url = db.Column(db.String(255))
#     tags = db.relationship('Tag', secondary='project_tag', backref='projects')
#
#
# class Category(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(255))
#
#
# class Tag(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(255))
#
#
# class ProjectTag(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     project_id = db.Column(db.Integer, db.ForeignKey('project.id'))
#     tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'))


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
