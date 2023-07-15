from naïve_bayes.tweet_disaster_classification.data_parser import DisasterTweetDataFrame
from naïve_bayes.tweet_disaster_classification.model import NaiveBayesModel

train = DisasterTweetDataFrame()
train.read_from_csv('data/train.csv', index_col='id')
train.process_text()

# test = DisasterTweetDataFrame()
# test.read_from_csv('data/test.csv', index_col='id')
# test.process_text()

model = NaiveBayesModel(train.data_frame)
model.train()
# model.predict(test.data_frame)
#
# submission = test.data_frame.loc[:, 'predictions']
# submission.rename('target', inplace=True)
#
# submission.to_csv('data/submission.csv')
