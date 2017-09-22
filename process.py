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

    def select_training_subset():
        #ensure data is valid top level data
        #reorder file contents randomly
    def preprocess():
        #do lemmatization first #Anvesh
        #then do stemming #Rafi
        #remove stop words and punctuation, remove non ascii words #Anvesh
        #handle word boundries #Rafi
        #lowercase everything #Rafi
        pass
        
    def process()
        #try using  TFIDF
        #also try doc2vec
        pass

