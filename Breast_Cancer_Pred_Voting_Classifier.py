#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
cancer_data["feature_names"]


# In[2]:


import pandas as pd
df = pd.DataFrame(cancer_data.data, columns=cancer_data['feature_names'])
df['cancer'] = cancer_data.target
df.sample(5)


# In[3]:


faltantes = df.isnull().sum()
faltantes


# In[4]:


df.dtypes


# In[5]:


cols_to_scale =cancer_data['feature_names']
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() 
data_scaled = scaler.fit_transform(df[cols_to_scale])
df_scaled = pd.DataFrame(data_scaled,columns=cancer_data['feature_names'] )
df_scaled['cancer'] = df['cancer'].copy()
df_scaled.sample(5)


# In[6]:


X = data_scaled
y = df['cancer']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
tree_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()


voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf),('dt', tree_clf), ('knn', knn_clf) ],
    voting='hard')
voting_clf.fit(X_train, y_train)


# In[7]:


from sklearn.metrics import accuracy_score

for clf in (log_clf, rnd_clf, svm_clf, tree_clf, knn_clf,voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# In[ ]:




