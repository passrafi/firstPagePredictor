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
        self.df = pd.read_csv("hacker_news_sample.csv", sep = ",", encoding= 'utf8')
        self.stem  = SnowballStemmer('english').stem
	self.selectedRows = None

    def select_training_subset(self):
        #ensure data is valid top level data
        #reorder file contents randomly
	score = self.df["score"]
	title = self.df["title"]
	self.selectedRows = self.df.loc[title.notnull() & score.notnull() & score.apply(lambda val: val > 50)] 

		
        pass

    
    def preprocess(self, text):
	
        #TODO replace TEST_WORDS with real data
        #remove non ascii words #Anvesh
	self.select_training_subset()
	self.selectedRows["title"] = self.selectedRows["title"].str.replace(r'[^\x00-\x7F]+', ' ')
	self.selectedRows["title"] = self.selectedRows["title"].str.lower()
	print self.selectedRows.head()			
        
	#words = tokenizer(text)
        #wordsFiltered = []
        #for w in words:
        #    w = self.stem(w)
        #    if w not in STOPWORDS:
        #        wordsFiltered.append(w)
        #print wordsFiltered
        #
        """
        Converting Numerics to words(1990 to one nine nine zero)
        """
        
    def process(self):
        #try using  TFIDF
        #also try doc2vec
        #TODO think about synthetic features
        pass

nlp = NLP()
nlp.preprocess("supdog")
