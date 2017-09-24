#title,url,text,dead,by,score,time,type,id,parent,descendants,ranking,deleted,timestamp
#helper file that cleans corrupt data or incorrect data types
import pandas as pd
from datetime import datetime
import time


def str_count (str):
  str_len = len(str)
  return str_len

def word_count (words):
    return len(words.split(" "))


data = pd.read_csv('new.csv')

# maxes = data.groupby('url', group_keys=False).apply(lambda x: x.ix[x.score.idxmax()])

# sort values based on score and dropping duplicates on url, possible dups if there multiple articles with the same score
data = data.sort_values('score', ascending=False).drop_duplicates('url').sort_index()
# only keep story articles and remove deleted content
data = data.loc[(data['type'] == 'story') & data["url"].notnull()]
# adding word count for title
data["title_word_count"] = data["title"].apply(word_count)
data["title_str_count"] = data["title"].apply(str_count)

print data.sort_values('score', ascending=False).head(10)


# data = data.query('deleted != "TRUE"')
data.to_csv('cleansed_hacker_news_sample.csv', index=False)
