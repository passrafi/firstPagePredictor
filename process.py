from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+').tokenize
from doc2vec import do_doc2vec

STOPWORDS = {}
for word in stopwords.words('english'):
    STOPWORDS[word] = 0



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
        self.df = pd.read_csv("new.csv", sep = ",", encoding= 'utf8')
        self.stem  = SnowballStemmer('english').stem
        self.preprocess()
        self.selectedRows = None

    def select_training_subset(self):
        #ensure data is valid top level data
        #reorder file contents randomly
	score = self.df["score"]
	title = self.df["title"]
	self.selectedRows = self.df.loc[title.notnull() & score.notnull() & score.apply(lambda val: val > 50)] 



    def removeStops(self, stuff):
        return ' '.join([word for word in stuff.split() if word not in STOPWORDS])

    def preprocess_line(self, text):
        #TODO replace TEST_WORDS with real data
        #remove non ascii words #Anvesh
        words = tokenizer(text)
        wordsFiltered = []
        self.df
        for w in words:
            w = self.stem(w)
            if not STOPWORDS.has_key(w):
                wordsFiltered.append(w)
        return ' '.join(wordsFiltered)
        #self.select_training_subset()
        #self.selectedRows["title"] = self.selectedRows["title"].str.replace(r'[^\x00-\x7F]+', ' ')
        #self.selectedRows["title"] = self.selectedRows["title"].str.lower()
        #self.selectedRows["title"] = self.selectedRows["title"].str.replace(r'[^\w\s]', ' ')
        #self.selectedRows["title"] = self.selectedRows["title"].map(lambda val: self.removeStops(val))
        #print self.selectedRows.head()
        

    def preprocess(self):
        print 'starting preprocess'
        for i in xrange(0, len(self.df['title'])):
            self.df['title'][i] = self.preprocess_line(self.df['title'][i])
            if i%100 ==0:
                print i
        print 'done with preprocess'



    def process(self):
        do_doc2vec(self.df[0:10000])
        #try using  TFIDF
        #also try doc2vec
        #TODO think about synthetic features
        pass

nlp = NLP()
nlp.process()
