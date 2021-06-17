#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest


# In[3]:


data = pd.read_csv('./data.csv')
dataF = pd.DataFrame(data.iloc[:,0:32])
#dataF = dataF.drop(dataF.iloc[:,32], axis = 1)
Y = dataF.iloc[0:,1]
X = dataF.iloc[0:,2:32]
#print(dataF.iloc[0])


# In[4]:


# Feature extraction menggunakan function chi
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(X, Y)
# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)


# In[5]:


# Feature selection menggunakan correlation matrix
corrMatrix = dataF.corr()
fig, ax = plt.subplots(figsize=(30,30)) 
sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
plt.show()


# In[6]:


#pd.set_option('display.max_rows', None)
corr_pairs = corrMatrix.unstack()
#print(corr_pairs) 
sorted_pairs = corr_pairs.sort_values(kind="quicksort")
sorted_pairs = sorted_pairs[(sorted_pairs)>0.8]
sorted_pairs = sorted_pairs[(sorted_pairs)<1]
print(sorted_pairs)

# Sesudah melakukan proses feature extraction dan feature correlation,
# kita dapat melihat feature-feature yang mempunyai nilai tinggi yang mendekati 1. 
# Dalam process ini dipilih nilai diatas 0.8, maka didapatkan feature yang akan digunakan sebagai berikut :
# ========================================================================================================
#   area_worst  
#   area_mean   
#   area_se
#   perimeter_worst 
#   perimeter_mean  
#   radius_worst 
#   radius_mean
#   perimeter_se
#   texture_worst 
#   texture_mean  
#   concavity_worst 
#   radius_se 
#   concavity_mean     
#   compactness_worst 
#   concave points_worst 
#   concave points_mean
#   compactness_mean 
#   symmetry_worst
#   concavity_se
# =========================================================================================================


# In[416]:


# Create X dan Y berdasarkan feature yang telah dipilih pada proses sebelumnya, selanjutnya proses splitting data training 
# dan data testing sebesar 70 % untuk training dan 30% untuk testing.

Y_new = dataF.iloc[0:,1]
X_new = dataF.iloc[0:,2:32]
X_new_1 = X_new["area_worst"]
X_new_2 = X_new["area_mean"]
X_new_3 = X_new["area_se"]
X_new_4 = X_new["perimeter_worst"]
X_new_5 = X_new["perimeter_mean"]
X_new_6 = X_new["radius_worst"]
X_new_7 = X_new["radius_mean"]
X_new_8 = X_new["perimeter_se"]
X_new_9 = X_new["texture_worst"]
X_new_10 = X_new["texture_mean"]
X_new_11 = X_new["concavity_worst"]
X_new_12 = X_new["radius_se"]
X_new_13 = X_new["concavity_mean"]
X_new_14 = X_new["compactness_worst"]
X_new_15 = X_new["concave points_worst"]
X_new_16 = X_new["concave points_mean"]
X_new_17 = X_new["compactness_mean"]
X_new_18 = X_new["symmetry_worst"]
X_new_19 = X_new["concavity_se"]
X_new_all = pd.concat([X_new_1, X_new_2, X_new_3, X_new_4, X_new_5, X_new_6, X_new_7, X_new_8, X_new_9, X_new_10, X_new_11, X_new_12, X_new_13, X_new_14, X_new_15, X_new_16, X_new_17, X_new_18, X_new_19], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_new_all, Y_new, test_size=0.3) # 70% training and 30% test


# In[413]:


# Proses fitting data menggunakan SVM dan kernel Linear

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)


# In[414]:


# Proses prediksi menggunakan data testing yang telah dibentuk pada proses sebelumnya

y_pred = svm_classifier.predict(X_test)


# In[415]:


# Proses evaluasi dari model yang telah dibentuk
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Dari hasil evaluasi menunjukan akurasi sebesar 0.94 atau 94% akurasi

