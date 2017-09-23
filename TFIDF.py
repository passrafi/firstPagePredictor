import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
#from textblob.classifiers
import numpy as np
from textblob import TextBlob as tb
import math
from multiprocessing import Process, Pool, Queue
import datetime
import time
import json
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
	
class tfidf:

	def __init__(self, dataSet, numProcs = 4):
		self.numProcs = numProcs #don't really need this data saved but whatever
		self.dataArr = np.array_split(dataSet, numProcs)
		self.tfidfArr = []
		self.Q = Queue()	
		self.n_containing = {}
		
		self.dataSet = dataSet
		
		# move this shit elsewhere
		x = preproc_data(dataSet)	
		#print type(z)	
		#remove stop words sorta like this...
		#y = x["title"].apply(lambda val: set(tb(val).lower().words) - stop_words)

		z = x["title"].apply(lambda val: tb(val))
		print "number of records: ", len(z.tolist())
		
		i = 0
		for d in self.dataArr:
			xx = preproc_data(d)
			zz = xx["title"].apply(lambda val: tb(val))
			self.tfidfArr.append(Process(target = self.calc_tf_idf, args = (zz,i,)))
			i = i + 1

		print "Populating hashtable of 'word in corpus' frequencies"
		self.populate_n_containing(x["title"].tolist())
		self.lenList = len(z.tolist())
		print len(self.n_containing.keys())
		print "Done! Ready to calculate tfidf!"
		self.scores = None
		
	def tf(self, word, blob):
		return blob.words.count(word) / (1.0*len(blob.words))
	def idf(self, word):
		return math.log(self.lenList / (1 + self.n_containing[word]))
	def exec_tfidf(self, word, blob):
		x = self.tf(word, blob) * self.idf(word)
		return x
	
	#long for loop but this seems to not take too long...
	def populate_n_containing(self, bloblist):
		#pPool = Pool()
		i = 0
		for blob in bloblist:
			i = i + 1
			#print i
			for word in blob.split():
				if(word not in self.n_containing):
					self.n_containing[word] = -1
					self.bottleNeck(word, bloblist) #synchronously called for now but could be optimized
					#pPool.apply_async(self.bottleNeck, (word, bloblist,), callback=self.update_Dict)
			if(i % 3000 == 0):
				print i, len(self.n_containing.keys())
		wordFreqs = pd.from_dict(self.n_containing)
		wordFreqs.to_csv("/home/anvesh/Desktop/tfidf")			
		#pPool.close()
		#pPool.join()				
	def update_Dict(self, retVal):
		D =  json.loads(retVal)	
		for r in D:
			self.n_containing[r] = 1.0 * retVal[r]	
		
	def bottleNeck(self, word, bloblist):
		x = sum(1 for blob1 in bloblist if word in blob1.split())
		self.n_containing[word] = x
	def calc_tf_idf(self, titleSeries, procIdx):	
		scores = titleSeries.apply(lambda blob: {word: self.exec_tfidf(word, blob) for word in blob.words})
		scores.to_csv("/home/anvesh/Desktop/tfidf-" + str(procIdx) + "-.log")


	def runAll(self):
		print "starting " + str(self.numProcs) + " workers..."	
		for p in self.tfidfArr:
			p.start()
		for p in self.tfidfArr:
			p.join()
		print "Done!"
	


hnadf = pd.read_csv("hacker_news_sample.csv", sep=",", encoding = 'utf8')
ts = time.time()
st1 = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')



#produce_histogram(hnadf)
#calc_tf_idf(hnadf)
print "start time: ", st1
TF = tfidf(hnadf, numProcs = 4)
TF.runAll()
ts = time.time()
st2 = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')

print "end time: ", st2
