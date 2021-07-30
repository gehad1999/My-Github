#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the necessary libraries
import pandas as pd
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statistics


# In[4]:


df = pd.read_csv('data.csv')
#Separating features and labels
features=df.iloc[:, :-1]
label=df.iloc[:,-1]


# In[5]:


svc = SVC(kernel='linear')
accu=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=1+i)
    # Default Linear kernel
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    accu.append(acc)


# In[6]:


print(accu)


# In[8]:


mea=statistics.mean(accu)
print("mean = ",mea)


# In[9]:


# After scaling the data
z=(features.min()-features.max())
X=(features-features.mean(axis=0))/z
print("Scaled \n",X)


# In[10]:


accu2=[]
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.4, random_state=1+i)
    # Default Linear kernel
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    acc = metrics.accuracy_score(y_test,y_pred)
    accu2.append(acc)


# In[11]:


print(accu2)


# In[12]:


mea2=statistics.mean(accu2)
print("mean = ",mea2)


# In[13]:


# The difference between the data before scaling and after scaling doesn't make a big difference on the accuracy.
# (the range of features values not wide) Features not have large difference on the range of their values.


# In[ ]:




