from __future__ import division
from os import path
import multiprocessing
import numpy as np

from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec
import random
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')
import pandas as pd
from utils import *

KEEP_COLS = [
    'title',
    'url',
    'text',
    'time',
    'score',
]


def most_common_element(arr):
    """Returns the most common element in arr, breaking ties randomly."""

    if len(arr) == 0:
        return np.nan

    list_of_tuples = Counter(arr).most_common()
    max_count = list_of_tuples[0][1]
    max_results = [value for value, count in list_of_tuples if count == max_count]

    return random.choice(max_results)

def text_tokenizer(text):
    """Return two array of tokenized lower case words with and without
    stopwords removed.
    Parameters
    ----------
    text: string
    """

    tokenizer = RegexpTokenizer(r'\w+')
    text = text.lower().replace("'", '')

    all_tokens = tokenizer.tokenize(text)
    without_stopwords = [word for word in all_tokens if not word in STOPWORDS]

    return np.array(all_tokens), np.array(without_stopwords)

class LabeledLineSentence(object):
    """A class which permutes the order of documents / labels. Used as a iterator
    that returns a LabeledSentence.
    Parameters
    ----------
    doc_list: a list of documnet strings
    labels_list: a list of label strings
    References
    ----------
    https://rare-technologies.com/doc2vec-tutorial/
    """

    def __init__(self, doc_list, labels_list):
       self.doc_list = doc_list
       self.labels_list = labels_list

    def __iter__(self):
        """Return a LabeledSentence after permuting the order of the documents."""

        perms = np.random.permutation( zip(self.doc_list, self.labels_list) )
        for text, label in perms:
            all_words, without_stop = text_tokenizer(text)
            yield LabeledSentence(all_words, [label])

class Doc2VecClassifier(object):

    def fit(self, text_array, labels):
        """Train a Doc2Vec model. In order to use the model for classification, we need all
        documents to be included.
        Parameters
        ----------
        text_array: list of strings, all documents in the corpus
        labels: list, containing unique identifier for every document in test_array
        """

        if gensim.models.doc2vec.FAST_VERSION == -1:
            raise Exception('Need to add c compiler to increase Doc2Vec speed')

        # initialize Doc2Vec object
        cores = multiprocessing.cpu_count()
        self.model = Doc2Vec(min_count=1, window=10, size=50, sample=1e-4, negative=5, workers=cores)

        # build model vocabulary
        itt = LabeledLineSentence(text_array, labels)
        self.model.build_vocab(itt)

        # train the model
        for epoch in range(10):
            print 'epoch', epoch
            self.model.train(itt, total_examples=self.model.corpus_count, epochs=self.model.iter)
            self.model.alpha -= 0.002
            self.model.min_alpha = self.model.alpha

    def predict(self, train_dict, test_dict, k):
        """Return arrays for true values and predicted values.
        Parameters
        ----------
        train_dict: dictionary, training data mapping labels to stars
        test_dict: dictionary, test data mapping labels to stars
        k: number of nearest vectors to use for classification
        """

        y_pred = []
        for label in test_dict.iterkeys():
            # since the model contains both training and test data we get the k * 3
            # nearest vectors and keep the k nearest vectors in the training data
            most_sim = self.model.docvecs.most_similar(str(label), topn=k*3)
            import bpdb; bpdb.set_trace()
            votes = []
            for lab, sim in most_sim:
                # only keep vectors in the training data
                if lab in train_dict:
                    votes.append(train_dict[lab])
                    # keep k nearest vectors
                    if len(votes) == k:
                        break

            # the most common value from the k nearest vectors is the prediction
            y_pred.append(most_common_element(votes))

        return y_pred

def doc2vec_model(df, train_dict, test_dict, k):
    """Fit, predict, and show classification results for the Doc2Vec model.
    Parameters
    ----------
    df: DataFrame, full dataset
    train_dict: dictionary, training data mapping labels to stars
    test_dict: dictionary, test data mapping labels to stars
    k: number of nearest vectors to use for classification
    Returns
    -------
    trained Doc2Vec model
    """

    print 'Fitting the Doc2Vec model...'
    d2v = Doc2VecClassifier()
    d2v.fit(df['text'].values, df['id'].values)
    y_pred = d2v.predict(train_dict, test_dict, k)
    y_test = test_dict.values()
     
    classification_errors(y_test, y_pred, 'Doc2Vec with k = {0}'.format(k))

    return d2v.model

def doc2vec_examples(yelp_model):
    """Print examples using the Doc2Vec model trained with yelp reviews data."""

    print '\nDoc2Vec examples:'
    print yelp_model.most_similar('pizza')
    print yelp_model.doesnt_match('burrito taco nachos pasta'.split())
    print yelp_model.doesnt_match('waiter server bartender napkin'.split())
    print yelp_model.most_similar(positive=['bar', 'food'], negative=['alcohol'])
    print yelp_model.most_similar(positive=['drink', 'hot', 'caffeine'])


if __name__ == '__main__':

    # load and preprocess the yelp reviews data
    df = pd.read_csv("new.csv")[0:1000]

    # exploratory data analysis plots
    #show_all_eda(df)

    # split data into training, test, and validation sets
    df_train, df_test, df_val = df_train_test_val_split(df[KEEP_COLS])
    X_train, y_train, X_test, y_test, X_val, y_val = train_test_val_split(df[KEEP_COLS], 'score')
    train_dict, test_dict, val_dict = train_test_val_dicts(df, 'id', 'score')

    # use grid search to optimize stage 1 & 2 model parameters
    #grid_search_naive_bayes_with_tfidf(df_train['text'].values, df_train['stars'].values)
    #grid_search_stage_two(X_train, y_train)

    # fit and print classification results for the two stage tfidf model
    #two_stage_model(X_train, y_train, X_test, y_test)

    # fit and print classification results for the Doc2Vec model
    model = doc2vec_model(df, train_dict, test_dict, k=25)

    # use the Doc2Vec model
    doc2vec_examples(model)
