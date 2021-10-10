#!/usr/bin/env python
# coding: utf-8

# # XSS\-Detection
# 
# Data: **dmzo\_nomal\.csv** as positive samples and **xssed.csv** as XSS samples\.
# 
# Process:
# 1. Normalization
# 2. Segmentation
# 3. Feature extraction (word2vec, doc2vec or statistics)
# 4. SVM
# 

# ## 1\. Load Data

# In[1]:


get_ipython().system('pip install gensim==4.1.2')
get_ipython().system('pip install numpy==1.19.5')


# In[2]:


import os 
import pandas as pd 

# data_path = './'
# work_path = './'
work_path = '/home/aistudio/work/'
data_path = '/home/aistudio/data/data52101/'


# In[3]:


normal = pd.read_csv(os.path.join(data_path, 'dmzo_nomal.csv'), header=None, names=['raw'])
mal = pd.read_csv(os.path.join(data_path, 'xssed.csv'), header=None, names=['raw'])
normal_df = pd.DataFrame(normal)
mal_df = pd.DataFrame(mal)

# mal_df
# normal_df


# In[4]:


normal_df.head()


# In[5]:


mal_df.head()


# ## 2\. Normalization and Segmentation
# 

# In[6]:


import nltk 
import re 
from urllib.parse import unquote 

def GeneSeg(payload):
    # digits -> '0'
    payload = payload.lower()
    payload = unquote(unquote(payload))
    payload, num = re.subn(r'\d+', '0', payload)
    # url -> 'http://u'
    payload, num = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    # segmentation 
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
        '''
    return nltk.regexp_tokenize(payload, r)


# In[7]:


# add a new column 
normal_df["parsed"] = normal_df['raw'].apply(GeneSeg)
mal_df['parsed'] = mal_df['raw'].apply(GeneSeg)
print(len(normal_df))
print(len(mal_df))


# In[8]:


normal_df.head()


# In[9]:


mal_df.head()


# ## 3\. Word Table 
# 
# Generate a new list (data frame) using the existing lists\.
# 
# DO NOT add this new list to the existing data frame `mal_df` or `normal_df`, for the new list is much longer\.
# 
# Text deduplication has to be done before all the words are sorted by quantity\.

# In[10]:


mal_word = list()
mal['parsed'].apply(lambda x: [mal_word.append(i) for i in x])
print(len(mal_word))
tmp = pd.DataFrame(mal_word, columns=['words'])
# add count 
mal_word_df = pd.DataFrame(tmp['words'].value_counts().to_frame().reset_index())
mal_word_df.columns = ['words', 'counts']
mal_word_df


# In[11]:


# select the first 300 words as word table
# save as mal_word_table.csv
wordtable_len = 300 
mal_word_df = mal_word_df[:wordtable_len]
mal_word_df.to_csv(os.path.join(work_path, 'mal_word_table.csv'))
mal_word_df.head()


# ## 4\. Word Vector
# 
# Now it's time to go back to `mal_df`\. We add a new column `words` to the data frame\. 
# 
# If a word is not in the list generated previously (`mal_word_df`, or the mal\_word\_table\.csv), it will be replaced with 'WORD'\.

# In[12]:


mal_word_list = mal_word_df['words'].tolist()
def wv_filter(lst):
    new_lst = list()
    for item in lst:
        # print(item)
        if item not in mal_word_list:
            new_lst.append('WORD')
        else:
            new_lst.append(item)
    return new_lst

mal_df['words'] = mal_df["parsed"].apply(wv_filter)
mal_df.head()


# In[13]:


# the same for normal_df
normal_df["words"] = normal_df['parsed'].apply(wv_filter)
normal_df.head()


# ### 4\.1 word2vec
# 

# In[14]:


from gensim.models.word2vec import Word2Vec 
import numpy


# In[15]:


# embedding_size: dimension of feature vector
embedding_size = 128 
#skip_window: indicates the maximum distance between the current word and the predicted word in a sentence
skip_window = 5 
# num_sampled: sets the number of noise words (if > 0)
num_sampled = 64 
# iteration time 
num_iter = 100 
data_set = mal_df['words']
data_set[:10]


# In[16]:


# do pay attention to parameter names
model = Word2Vec(data_set, vector_size=embedding_size, window=skip_window, negative=num_sampled, epochs=num_iter)


# ### 4\.2 Save/Load Model

# In[29]:


model.save(os.path.join(work_path, 'model_word2vec_auto'))


# In[30]:


model_new = Word2Vec.load(os.path.join(work_path, 'model_word2vec_auto'))


# ### 4\.3 Test

# In[31]:


# find the most similar words based on keywords
embeddings = model_new.wv
embeddings.similar_by_word("</script>", 5)


# In[32]:


embeddings.similar_by_word("alert(", 5)


# ## 5\. SVM
# 

# In[18]:


import numpy as np
from sklearn import model_selection
from sklearn.svm import SVC


tags = list()
y = list()
x = list()
# xssed.csv
for item in mal_df['raw'].tolist():
    tags.append(item.strip())
    y.append(1)
# dmzo_nomal.csv 
for item in normal_df['raw'].tolist():
    tags.append(item.strip())
    y.append(0)

def count_str(line, str):
    return line.lower().count(str)


def get_feats(line):
    return [
        count_str(line, 'script'),
        count_str(line, 'java'),
        count_str(line, 'iframe'),
        count_str(line, '<'),
        count_str(line, '>'),
        count_str(line, '\"'),
        count_str(line, '\''),
        count_str(line, '%'),
        count_str(line, '('),
        count_str(line, ')'),
    ]

for tag in tags:
    x.append(get_feats(tag))

x = np.array(x)
y = np.array(y)

# print(x.size)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, random_state=0, test_size=0.3)


# SVM classifier
clf = SVC(C=0.5, kernel='linear')

# train
clf.fit(x_train, y_train)


# accuracy
def show_accuracy(a, b, tip):
    acc = a == b
    print("%s Accuracy:%.3f" % (tip, np.mean(acc)))


# print accuracy
show_accuracy(clf.predict(x_train), y_train, 'training data')
show_accuracy(clf.predict(x_test), y_test, 'testing data')


# callback (TP/P, TP/(TP+FN))
def show_recall(y, y_hat, tip):
    cnt = 0
    length = len(y)
    for i in range(length):
        if y[i] == 1 and y_hat[i] == 1:
            cnt += 1
    print('%s Recall: %.3f' % (tip, cnt/np.sum(y)))


show_recall(y_train, clf.predict(x_train), 'training data')
show_recall(y_test, clf.predict(x_test), 'testing data')


def predict(tag):
    x = []
    x.append(get_feats(tag))
    return clf.predict(x)[0]


def result(string):
    tag = string 
    y = predict(tag.strip())
    if y == 1:
        print("XSS")
    else:
        print("no-mal")


# In[20]:


result('form.search_text=Dell%22%3E%3Cscript%3Ealert(/xss-Bulgari<br/>a/.source)%3C/script%3E&form.hardware_category=LAPTOP')
result('"c=26zzzzzzzzzz""><script>alert(1)</script>"')
result('search_query=%3CSCRIPT%3Ealert%28/XSS%20by%2003<br/>storic/%29%3C/SCRIPT%3E')
result('id=224%22%3E%3Cscript%3Ealert%28document.cookie%29%3C/script%3E')
result('ring=slashrecs&amp;id=26&amp;hub')
result("ring=thenewstartreksl")
result('ring=communiqueerotiq&amp;id=11&amp;hub')
result("x=0")

