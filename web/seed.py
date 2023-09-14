from web import app, db
from web import Project, Tag, ProjectTag, Category

category1 = Category(name='Vision')
category2 = Category(name='NLP')
category3 = Category(name='Tabular')

project1 = Project(name='Tweet Model', description='Project descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject description', image_url='images/example.jpg', url='/tweet-model', category=category2)
project2 = Project(name='Second Project', description='Project descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject descriptionProject description', image_url='images/example.jpg', url='/project2url', category=category1)
project3 = Project(name='Third Project', description='DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION DESCRIPTION ', image_url='images/example.jpg', url='/project3url', category=category3)

tag1 = Tag(name='Python')
tag2 = Tag(name='TensorFlow')
tag3 = Tag(name='CNN')

project_tag1 = ProjectTag(project_id=project1, tag_id=tag1)
project_tag2 = ProjectTag(project_id=project2, tag_id=tag2)

with app.app_context():
    db.session.add(project1)
    db.session.add(project2)
    db.session.add(project3)
    db.session.add(tag1)
    db.session.add(tag2)
    db.session.commit()

    # Потім встановіть відношення між Project і Tag через таблицю project_tag
    project1.tags.append(tag1)
    project2.tags.append(tag2)

    # Знову збережіть зміни в базу даних
    db.session.commit()