<h1>AI Web Playground</h1>

<p>This is a Flask-based website for hosting our AI models and implementing the full model life cycle for them.</p>

<p>We've created different models from NLP (Natural Language Processing) and CV (Computer Vision) fields. You can see the list of them below.</p>

<p>Each model has a detail page where user can interact with the model using their own data. </p>

### List of models

#### NLP:

- [*Tweet sentiment analysis*](/models/naive_bayes/sentiment_analysis)
- [*Tweet disaster classification*](/models/naive_bayes/tweet_disaster_classification)


#### CV:

- [*Animals classification*](/models/animals_classification)
- [*Weather classification*](/models/weather_classification) `IN PROGRESS`
- [*Military aircraft detection*](/models/military_aircraft_detection) `IN PROGRESS`
- [*Car license plate detection*](/models/license_plate_detection) `IN PROGRESS`

### Train models
#### Prepare environment
```
git clone https://github.com/cartiss/ai_playground.git 
cd ai-app
poetry install
```
#### Choose model to train
- [*Animals classification*](/models/animals_classification) 

```
make train_animal
```

- [*Tweet sentiment analysis*](/models/naive_bayes/sentiment_analysis) 

```
make train_sentiment
```

- [*Tweet disaster classification*](/models/naive_bayes/tweet_disaster_classification)

```
make train_disaster
```