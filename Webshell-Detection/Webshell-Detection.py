#!/usr/bin/env python
# coding: utf-8

# # Webshell\-Detection 
# 
# Steps:
# 

# In[1]:


# !pip install pandas==1.3.4


# In[2]:


import os 


data_path = "./data/"
php_benign_path = os.path.join(data_path, "php-benign/")
php_webshell_path = os.path.join(data_path, "php-webshell/")
save_path = os.path.join(data_path, "save/")
opcode_blacklist_path = os.path.join(save_path, "opcode_blacklist.csv")
opcode_whitelist_path = os.path.join(save_path, "opcode_whitelist.csv")


# ## 1\. Opcode 

# In[3]:


from utils import load_php_opcode_from_dir_with_file


# In[4]:


# blacklist 
load_php_opcode_from_dir_with_file(php_webshell_path, opcode_blacklist_path)


# In[5]:


# whitelist 
load_php_opcode_from_dir_with_file(php_benign_path, opcode_whitelist_path)


# ## 2\. Load 

# In[6]:


import os 
import pandas as pd 
import numpy as np 


# In[7]:


blacklist_df = pd.read_csv(opcode_blacklist_path, header=None)
whitelist_df = pd.read_csv(opcode_whitelist_path, header=None)
# whitelist_df
blacklist_df.head()


# In[8]:


X_blacklist = np.array(blacklist_df)
X_whitelist = np.array(whitelist_df)
y_blacklist = [1] * len(X_blacklist)
y_whitelist = [0] * len(X_whitelist)
X = list()
for item in X_blacklist:
    X.append(item[0])
for item in X_whitelist:
    X.append(item[0])
y = y_blacklist + y_whitelist


# In[9]:


print(len(X))
print(len(y))
X[0]


# ## 3\. Naive Bayes 

# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import joblib
from utils import get_feature_by_tfidf
import os 


model_path2 = os.path.join(save_path, "gnb.pkl")


# In[11]:


X_nb = get_feature_by_tfidf(input_X=X)
y_nb = y 

X_nb_train, X_nb_test, y_nb_train, y_nb_test = train_test_split(X_nb, y_nb, test_size=0.4, random_state=0)
clf = GaussianNB()
clf.fit(X_nb_train, y_nb_train)
joblib.dump(clf, model_path2)
y_pred = clf.predict(X_nb_test)

print("[*] accuracy: {}".format(metrics.accuracy_score(y_nb_test, y_pred)))
print("[*] precision: {}".format(metrics.precision_score(y_nb_test, y_pred)))
print("[*] recall: {}".format(metrics.recall_score(y_nb_test, y_pred)))
print("[*] loss: {}".format(metrics.brier_score_loss(y_nb_test, y_pred)))
print("[*] confusion matrix:\n{}".format(metrics.confusion_matrix(y_nb_test, y_pred)))


# ## 4\. Random Forest 

# In[12]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
from utils import get_feature_by_tfidf
import os 


model_path = os.path.join(save_path, "rf.pkl")


# In[13]:


X_rf = get_feature_by_tfidf(input_X=X)
y_rf = y 

X_rf_train, X_rf_test, y_rf_train, y_rf_test = train_test_split(X_rf, y_rf, test_size=0.4, random_state=0)
clf2 = RandomForestClassifier(n_estimators=50)
clf2.fit(X_rf_train, y_rf_train)
joblib.dump(clf2, model_path)
y_pred2 = clf2.predict(X_rf_test)

print("[*] accuracy: {}".format(metrics.accuracy_score(y_rf_test, y_pred2)))
print("[*] precision: {}".format(metrics.precision_score(y_rf_test, y_pred2)))
print("[*] recall: {}".format(metrics.recall_score(y_rf_test, y_pred2)))
print("[*] loss: {}".format(metrics.brier_score_loss(y_rf_test, y_pred2)))


# ## 5\. Test 

# In[14]:


import os 
import joblib 
from utils import load_php_opcode, get_feature_by_tfidf


model_path = os.path.join(save_path, "rf.pkl")


def do_test(test_file):
    blacklist_df = pd.read_csv(opcode_blacklist_path, header=None)
    whitelist_df = pd.read_csv(opcode_whitelist_path, header=None)
    X_blacklist = np.array(blacklist_df)
    X_whitelist = np.array(whitelist_df)
    all_file = list()
    for item in X_blacklist:
        all_file.append(item[0])
    for item in X_whitelist:
        all_file.append(item[0])
    
    opcode = load_php_opcode(test_file)
    all_file.append(opcode)
    x = get_feature_by_tfidf(all_file)
    rf = joblib.load(model_path)
    y_p = rf.predict(x[-1:])
    if y_p == [0]:
        print("benign")
    else:
        print("webshell!!!")


# In[15]:


do_test("./test/0d48acd8559ceed891f997f7204b34cbf411292b.php")
do_test("./test/2.php")
do_test("./test/1.php")

