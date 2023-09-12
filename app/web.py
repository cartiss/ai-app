"""Run Flask application."""
import os

from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import desc


from utils import make_sentiment_prediction, make_disaster_prediction, compile_results

postgres_name = os.getenv('SQL_DB')
app = Flask(__name__)
app.config['SECRET_KEY'] = 'iejowda32msflsn3dkf7jnad9mk1lpd'
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://ai_db_user:Hgyfshbjs8482@localhost/aiprojectsdb'
app.config['UPLOAD_FOLDER'] = 'static'
db = SQLAlchemy(app)



class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    description = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    tags = db.relationship('Tag', secondary='project_tag', backref='projects')
    url = db.Column(db.String(255), unique=True)


class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))


class ProjectTag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'))
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'))


@app.route("/")
def index() -> None:
    """Render home page."""
    projects = Project.query.order_by(Project.id).all()
    return render_template('index.html', projects=projects)


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
