import nltk
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
    df = pd.read_csv("20k.csv", names=names, header=None, delim_whitespace=True)
    def preprocess():
        #reorder file contents randomly
        #ensure data is valid top level data
        #do lemmatization first
        #then do stemming
        #remove stop words and punctuation 
        #handle word boundries
        #stemming / lemmatization 
        #lowercase everything
        pass
        
    def process()
        #try using  TFIDF
        #also try doc2vec
        pass

