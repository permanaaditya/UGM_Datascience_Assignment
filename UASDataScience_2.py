#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer


# In[25]:


#Proses import dataset
data = pd.read_csv('./gojek_twitter_dataset.csv')
data


# In[26]:


#Proses data preprosessing dan feature extraction
count_vectorizer  = CountVectorizer(stop_words = "english", max_features = 5000 )
feature_vector    = count_vectorizer.fit(data.tweet)
train_ds_features = count_vectorizer.transform(data.tweet)
features        = feature_vector.get_feature_names()
features_counts = np.sum(train_ds_features.toarray(), axis = 0)
feature_counts  = pd.DataFrame(dict( features = features,counts = features_counts ))
feature_counts.sort_values("counts", ascending = False)[0:20]


# In[12]:


train_x, test_x, train_y, test_y = train_test_split(train_ds_features, data.sentimen, test_size = 0.3, random_state = 42 )
print(train_x)


# In[21]:


# Proses pembentukan model klasifikasi menggunakan SVM 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(train_x.toarray(), train_y)


# In[22]:


# Proses prediksi dan evaluasi dari model yang telah dibentuk
y_pred         = svm_classifier.predict(test_x.toarray())
confusion_mat  = confusion_matrix(test_y,y_pred)
class_report   = classification_report(test_y,y_pred)
fig, ax        = plt.subplots(figsize=(30,30)) 
sn.heatmap(confusion_mat, annot=True, linewidths=.5, ax=ax)
plt.show()
print(confusion_mat)
print(class_report)

