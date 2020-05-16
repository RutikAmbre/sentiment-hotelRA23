#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
Reviewdata=pd.read_csv('train.csv')


# In[3]:


Reviewdata.shape


# In[4]:


Reviewdata.head()


# In[5]:


Reviewdata.info()


# In[6]:


Reviewdata.describe().transpose()


# In[7]:


count=Reviewdata.isnull().sum().sort_values(ascending=False)
percentage=((Reviewdata.isnull().sum()/len(Reviewdata)*100)).sort_values(ascending=False)
missing_data=pd.concat([count,percentage],axis=1,keys=['Count','Percentage'])
print('Count and Percentage of missing values in columns:')
missing_data


# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print('Percentage for default\n')
print(round(Reviewdata.Is_Response.value_counts(normalize=True)*100,2))
round(Reviewdata.Is_Response.value_counts(normalize=True)*100,2).plot(kind='bar')
plt.title('Percentage Distribution by review type')
plt.show()


# In[9]:


Reviewdata.drop(columns=['User_ID','Browser_Used','Device_Used'],inplace=True)


# In[10]:


import re
import string
#converts to lowercase ,removes square bracket ,punctuation and numbers.
#re.sub has format regular expression,new string,processed string
def text_clean_1(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text) 
    text=re.sub('\w*\d\w*','',text)
    return text

cleaned1=lambda x:text_clean_1(x)


# In[11]:


Reviewdata['Cleaned_Description']=pd.DataFrame(Reviewdata.Description.apply(cleaned1))
Reviewdata.head()


# In[12]:


def text_clean_2(text):
    text=re.sub('[''""...]','',text)
    text=re.sub('\n','',text)
    return text

cleaned2=lambda x:text_clean_2(x)


# In[13]:


Reviewdata['Cleaned_Description_New']=pd.DataFrame(Reviewdata['Cleaned_Description'].apply(cleaned2))
Reviewdata.head()


# In[15]:


from sklearn.model_selection import train_test_split
X=Reviewdata.Cleaned_Description_New
Y=Reviewdata.Is_Response
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
print("X_train : ",len(X_train))
print("X_test : ",len(X_test))
print("Y_train : ",len(Y_train))
print("Y_test : ",len(Y_test))


# In[18]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec=TfidfVectorizer()
clf=LogisticRegression(solver="lbfgs")
from sklearn.pipeline import Pipeline #executes one by one


# In[19]:


model=Pipeline([('vectorizer',tvec),('classifier',clf)])
model.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix
y_pred=model.predict(X_test)
confusion_matrix(y_pred,Y_test)


# In[20]:


from sklearn.metrics import accuracy_score,precision_score,recall_score

print("Accuracy : ",accuracy_score(y_pred,Y_test))
print("Precision : ",precision_score(y_pred,Y_test,average='weighted'))
print("Recall : ",recall_score(y_pred,Y_test,average='weighted'))


# In[22]:


example=['I am satisfied']
result=model.predict(example)
print(result)


# In[28]:


example1=['I am frustated']
result1=model.predict(example1)
print(result1)


# In[ ]:




