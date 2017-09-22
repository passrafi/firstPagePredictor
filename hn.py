import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
#from textblob.classifiers
import numpy as np
from textblob import TextBlob as tb
import math

def preproc_data(hnadf):
	score = hnadf["score"]
	title = hnadf["title"]
	x = hnadf.loc[title.notnull() & score.notnull() & score.apply(lambda val: val > 50)] 
	return x	

def produce_histogram(hnadf):
	x = preproc_data(hnadf)	
	y = x["score"].value_counts(bins=2000)
	z = y.loc[y.apply(lambda val: val > 0)]

	zz =  pd.Series(z.values, index = z.index.map(lambda val: val.right))

	ax = np.log(zz).plot(style='+')
	fig = ax.get_figure()

	fig.savefig('/home/anvesh/Downloads/test.pdf')
	

def tf(word, blob):
	return blob.words.count(word) / (1.0*len(blob.words))
def n_containing(word, bloblist):
	return sum(1 for blob in bloblist if word in blob.words)
def idf(word, bloblist):
	return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))
def tfidf(word, blob, bloblist):
	x = tf(word, blob) * idf(word, bloblist)
	print word, x 
	return x

def calc_tf_idf(hnadf):
	
	stop_words = set(['the', 'and', 'a'])
	
	x = preproc_data(hnadf)	
	z = x["title"].apply(lambda val: tb(val))
	print type(z)	
	#remove stop words sorta like this...
	#y = x["title"].apply(lambda val: set(tb(val).lower().words) - stop_words)
	print z.head()	
	scores = z.apply(lambda blob: {word: tfidf(word, blob, z.tolist()) for word in blob.words})
	scores.to_csv("/home/anvesh/Desktop/tfidf.log")
	print "Done!"




hnadf = pd.read_csv("hacker_news_sample.csv", sep=",", encoding = 'utf8')

#produce_histogram(hnadf)
calc_tf_idf(hnadf)

