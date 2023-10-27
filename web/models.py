from flask_sqlalchemy import SQLAlchemy
from flask import Flask
from decouple import config

db_user = config('SQL_USER')
db_password = config('SQL_PASSWORD')
db_name = config('SQL_DB')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://{}:{}@localhost/{}'.format(db_user, db_password, db_name)
db = SQLAlchemy(app)


class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    description = db.Column(db.Text)
    category_id = db.Column(db.Integer, db.ForeignKey('category.id'))
    category = db.relationship('Category', backref='projects')
    url = db.Column(db.String(255), unique=True)
    image_url = db.Column(db.String(255))
    tags = db.relationship('Tag', secondary='project_tag', backref='projects')


class Category(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))


class Tag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))


class ProjectTag(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'))
    tag_id = db.Column(db.Integer, db.ForeignKey('tag.id'))
