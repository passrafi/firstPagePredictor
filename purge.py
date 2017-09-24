#title,url,text,dead,by,score,time,type,id,parent,descendants,ranking,deleted,timestamp
#helper file that cleans corrupt data or incorrect data types
import pandas as pd
from datetime import datetime
import rfc3339      
import iso8601  
import time


def str_count (str):
    str_len = len(str)
    return str_len

def word_count (words):
    return len(words.split(" "))


def get_date_object(date_string):
    return iso8601.parse_date(date_string)

def get_hour (time_object):
    the_time = get_date_object(time_object)
    current_hour =  the_time.hour - (the_time.hour % 3)
    return current_hour

def get_day (time_object):
    the_time = get_date_object(time_object)
    current_day = the_time.weekday()
    return current_day


def get_year (time_object):
    the_time = get_date_object(time_object)
    current_day = the_time.year
    return current_day


data = pd.read_csv('new.csv')

# maxes = data.groupby('url', group_keys=False).apply(lambda x: x.ix[x.score.idxmax()])

# sort values based on score and dropping duplicates on url, possible dups if there multiple articles with the same score
data = data.sort_values('score', ascending=False).drop_duplicates('url').sort_index()
# only keep story articles and remove deleted content
data = data.loc[(data['type'] == 'story') & data["url"].notnull()]
# adding word count for title
data["user"] = data["by"]

data["title_word_count"] = data["title"].apply(word_count)
data["title_str_count"] = data["title"].apply(str_count)
data["hour"] = data["timestamp"].apply(get_hour)
data["day_of_week"] = data["timestamp"].apply(get_day)
data["year"] = data["timestamp"].apply(get_year)


# print data.dtypes

print data.sort_values('score', ascending=False).head(10)

# data = data.query('deleted != "TRUE"')
data.to_csv('more-features.csv', index=False)

