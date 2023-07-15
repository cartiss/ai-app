"""Text formatter."""
import re
import string

import pandas as pd
import contractions
import nltk


class TweetTextFormatter:
    """Contains methods for formatting and cleaning sets of text samples."""

    @staticmethod
    def clean_text(text: pd.Series) -> pd.Series:
        """
        Cleans set of text samples.
        Cleaning the text involves:
            - converting it to lower case;
            - expanding contractions;
            - removing URLs and HTML tags;
            - removing any non-ASCII characters;
            - removing punctuations;
            - removing numbers.
        :param text: pd.Series containing the set of tweet text samples.
        :return: pd.Series containing the set of cleaned tweet text samples.
        """
        # to lower case
        clean_text = text.apply(lambda text_field: text_field.lower())

        # expand contractions
        clean_text = [contractions.fix(word) for word in clean_text]

        # remove urls
        clean_text = [re.sub(r'https?://\S+|www\.\S+', '', word) for word in clean_text]

        # remove html tags
        html_regex = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        clean_text = [re.sub(html_regex, '', word) for word in clean_text]

        # remove non-ascii characters
        clean_text = [re.sub(r'[^\x00-\x7f]', '', word) for word in clean_text]

        # remove punctuations
        clean_text = [word.translate(str.maketrans('', '', string.punctuation)) for word in clean_text]

        # remove numbers
        clean_text = [re.sub(r'\b\d+\b', '', word) for word in clean_text]

        return pd.Series(clean_text, index=text.index)

    @staticmethod
    def tokenize(clean_text: pd.Series) -> pd.Series:
        """
        Tokenizes set of text samples by making each separate word a token.
        It's strongly suggested to have the text cleaned before passing
        it to this method as this class provides no way of cleaning the text after it has been tokenized.
        :param clean_text: pd.Series containing the set of cleaned text samples.
        :return: pd.Series containing the set of sets of tokenized text samples.
        """
        tokenized = clean_text.apply(nltk.tokenize.word_tokenize)
        return pd.Series(tokenized, index=clean_text.index)

    @staticmethod
    def remove_stopwords(tokenized_text: pd.Series) -> pd.Series:
        """
        Removes stopwords from a set of tokenized text samples.
        :param tokenized_text: pd.Series containing the set of tokenized text samples.
        :return: pd.Series containing the set of the text samples with stopwords removed.
        """

        nltk.download('punkt')
        nltk.download('stopwords')
        stop = set(nltk.corpus.stopwords.words('English'))
        removed_stopwords = tokenized_text.apply(
            lambda text: [word for word in text if word not in stop]
        )
        return pd.Series(removed_stopwords, index=tokenized_text.index)

    @staticmethod
    def stem(tokenized_text: pd.Series) -> pd.Series:
        """
        Stems the words in the set of tokenized text samples.
        :param tokenized_text: pd.Series containing the set of tokenized text samples.
        :return: pd.Series containing the set of the text samples with stemmed words.
        """
        stemmer = nltk.PorterStemmer()
        stemmed = tokenized_text.apply(lambda text: [stemmer.stem(word) for word in text])
        return pd.Series(stemmed, index=tokenized_text.index)

    def process_text(self, text: pd.Series) -> pd.Series:
        """Processes the text samples in the data and appends the 'processed_text' column to the object's dataframe."""
        cleaned = self.clean_text(text)
        tokenized = self.tokenize(cleaned)
        removed_stopwords = self.remove_stopwords(tokenized)
        stemmed = self.stem(removed_stopwords)
        return pd.Series(stemmed, index=text.index)
