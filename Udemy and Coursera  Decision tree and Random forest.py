#!/usr/bin/env python
# coding: utf-8

# ## Decision tree and Random forest

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Import the data

# In[5]:


df = pd.read_csv('kyphosis.csv')


# ## Structure and Summary of the data

# In[7]:


df.info()  ## Structure


# In[8]:


df.head()   ## The data


# In[9]:


df.describe()  ## Summary


# In[11]:


df.shape   ## size/shape


# In[20]:


## data Visulazation

sns.pairplot(df, hue = 'Kyphosis')


# In[23]:


### Train test split

from sklearn.model_selection import train_test_split
X = df.drop('Kyphosis', axis = 1)
y = df['Kyphosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[24]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()


# In[25]:


## fit the decision tree model
dtree.fit(X_train, y_train)


# In[27]:


### prediction
pred = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))


# In[30]:


from sklearn import tree
tree.plot_tree(dtree)
plt.show()


# ## Random Forest: 

# In[32]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[33]:


rfc_pred = rfc.predict(X_test)


# In[35]:


print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




