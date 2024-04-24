#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


# In[4]:


movies=pd.read_csv('dataset/tmdb_5000_movies.csv')
credits=pd.read_csv('dataset/tmdb_5000_credits.csv')


# In[5]:


movies.head(2)


# In[6]:


credits.head(2)


# In[7]:


movies.shape


# In[8]:


credits.shape


# In[9]:


#merging datset
movies=movies.merge(credits,on="title")


# In[10]:


movies.head(2)


# In[11]:


movies.shape


# In[12]:


# node need of budget,homepage,original language,popularity,
movies.columns


# In[13]:


movies=movies[['movie_id','title','overview','genres',"keywords","cast","crew"]];


# In[14]:


movies.head(2)


# In[15]:


movies.shape


# In[16]:


movies.isnull().sum()


# In[17]:


movies.dropna(inplace=True)


# In[18]:


movies.isnull().sum()


# In[19]:


movies.shape


# In[20]:


movies.duplicated().sum()


# In[21]:


movies.iloc[0]['genres']


# In[22]:


import ast

def convert(text):
  l=[]
  for i in ast.literal_eval(text):
    l.append(i['name'])
  return l


# In[23]:


movies['genres']=movies['genres'].apply(convert)


# In[24]:


movies.iloc[0]['keywords']


# In[25]:


movies['keywords']=movies['keywords'].apply(convert)


# In[26]:


movies.head(2)


# In[27]:


movies.iloc[0]['cast']


# In[28]:


# movies['cast']=movies['cast'].apply(convert)
import ast

def convert_cast(text):
  l = []
  counter=0;
  for i in ast.literal_eval(text):
    if counter < 3:
      l.append(i['name'])
    counter +=1

  return l


# In[29]:


movies['cast'] = movies['cast'].apply(convert_cast)


# In[30]:


movies.head(2)


# In[31]:


movies.iloc[0]['crew']


# In[32]:


import ast

def fetch_director(text):
  l=[]
  for i in ast.literal_eval(text):
    if i['job'] == "Director":
      l.append(i['name'])
      break;
  return l


# In[33]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[34]:


movies.head(2)


# In[35]:


movies.iloc[0]['overview']


# In[36]:


movies['overview'] =movies['overview'].apply(lambda x:x.split())


# In[37]:


movies.head()


# In[38]:


movies.iloc[0]['overview']


# In[39]:


def removespace(word):
  l = []
  for i in word:
    l.append(i.replace(" ",""))
  return l;


# In[40]:


movies['cast']=movies['cast'].apply(removespace)
movies['crew']=movies['crew'].apply(removespace)
movies['genres']=movies['genres'].apply(removespace)
movies['keywords']=movies['keywords'].apply(removespace)
# movies['cast']=movies['cast'].apply(removespace)


# In[41]:


movies.head()


# In[42]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[43]:


movies.head()


# In[44]:


movies.iloc[0]['tags']


# In[45]:


new_df=movies[["movie_id","title","tags"]]


# In[46]:


new_df.head()


# In[47]:


new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))


# In[48]:


new_df.head()


# In[49]:


new_df['tags']=new_df['tags'].apply(lambda x: x.lower())


# In[50]:


new_df.head()


# In[54]:


# pip install nltk
import nltk
from nltk.stem import PorterStemmer


# In[55]:


ps =PorterStemmer()


# In[56]:


def stems(text):
  l=[]
  for i in text.split():
    l.append(ps.stem(i))
  return " ".join(l)


# In[57]:


new_df['tags']=new_df['tags'].apply(stems)


# In[58]:


new_df.iloc[0]['tags']


# In[59]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words='english')


# In[60]:


vector=cv.fit_transform(new_df['tags']).toarray()


# In[61]:


vector


# In[62]:


vector.shape


# In[63]:


from sklearn.metrics.pairwise import cosine_similarity


# In[64]:


similarity=cosine_similarity(vector)


# In[65]:


similarity


# In[66]:


similarity.shape


# In[67]:


new_df[new_df['title'] == 'Spider-Man'].index[0]


# In[68]:


def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:10]:
        print(new_df.iloc[i[0]].title)


# In[69]:


recommend('Spider-Man')


# In[70]:


recommend("The Dark Knight Rises")


# In[74]:


import pickle
pickle.dump(new_df, open('artificates/movie_list.pkl' , "wb"))
pickle.dump(new_df, open('artificates/similarity.pkl' , "wb"))


# In[ ]:




