from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+').tokenize


STOPWORDS = stopwords.words('english')

import re
from pandas import DataFrame, read_csv
import pandas as pd

#Todo read introduction to statistical learning
#doc2vec / assumes similaries
#svm  / draws boundry lines  
#latent semantic indexing
#topic modeling <--investigate later on
TEST_WORDS = 'a tester fishes sentences doesn\'t require what\'s not here, It\'s there'
class NLP:    
    def __init__(self):
        #df = pd.read_csv("new.csv")
        self.stem  = SnowballStemmer('english').stem


    def select_training_subset(self):
        #ensure data is valid top level data
        #reorder file contents randomly
        pass

    def preprocess(self, text):
        #TODO replace TEST_WORDS with real data
        text = TEST_WORDS
        text = text.lower()
        words = tokenizer(text)
        wordsFiltered = []
        for w in words:
            w = self.stem(w)
            if w not in STOPWORDS:
                wordsFiltered.append(w)
        print wordsFiltered
        #remove non ascii words #Anvesh
        """
        Converting Numerics to words(1990 to one nine nine zero)
        """
        
    def process(self):
        #try using  TFIDF
        #also try doc2vec
        #TODO think about synthetic features
        pass

nlp = NLP()
nlp.preprocess()
