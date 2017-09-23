from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
STOPWORDS = stopwords.words('english')

import re
from pandas import DataFrame, read_csv
import pandas as pd

#Todo read introduction to statistical learning
#doc2vec / assumes similaries
#svm  / draws boundry lines  
#latent semantic indexing
#topic modeling <--investigate later on
class NLP:    
    #order might matter
    df = pd.read_csv("new.csv", names=names, header=None, delim_whitespace=True)
    self.stem  = SnowballStemmer('english').stem


    def select_training_subset():
        #ensure data is valid top level data
        #reorder file contents randomly

    def preprocess_word(word):
        #stemming/lemmatization
        word = self.lemmatize(word)
        #TODO stopword/punctuation removal

        

    def preprocess():
        #do lemmatization first #Anvesh
        #then do stemming #Rafi
        #remove stop words and punctuation, remove non ascii words #Anvesh
        #handle word boundries #Rafi
        #lowercase everything #Rafi


        """
        Stemming/Lemmatizing
        Converting all words to lower case
        Punctuation removal
        Stop words removal
        Converting Numerics to words(1990 to one nine nine zero)
        """
        pass
        
    def process()
        #try using  TFIDF
        #also try doc2vec
        #TODO think about synthetic features
        pass

