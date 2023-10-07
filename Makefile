install_dependencies:
	poetry install --no-root

install_dev_dependencies:
	poetry install --no-root --dev

run_web:
	poetry run python web/web.py

run_docker:
	docker-compose up --build

train_sentiment:
	poetry run python naïve_bayes/sentiment_analysis/model.py --param_path=trained_models/sentiment_analysis

train_disaster:
	poetry run python naïve_bayes/tweet_disaster_classification/model.py

pylint:
	poetry run pylint .

flake8:
	poetry run flake8 .
