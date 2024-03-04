#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler


# ##### ### Key Name	Description
# 1. 
# Loan_ID	Unique Lo
# a    n ID2. 
# Gender	Male/ F    
#     emal3. e
# Married	Applicant married    
#      (Y/4. N)
# Dependents	Number of dep    
#     ende5. nts
# Education	Applicant Education (Graduate/ Under G    
#     radu6. ate)
# Self_Employed	Self-emplo    
#     yed 7. (Y/N)
# ApplicantIncome	Applic    
#     ant 8. income
# CoapplicantIncome	Coappli    
#     cant9.  income
# LoanAmount	Loan amount     
#     in t10. housands
# Loan_Amount_Term	Term of a l    
#     oan 11. in months
# Credit_History	credit history me    
#     ets 12. guidelines
# Property_Area	Urban/ Se    
#     mi-U13. rban/ Rural
# Loan_Status	Loan approved (Y/N)
# 

# In[2]:


df = pd.read_csv('lt.csv.csv')
df.head()


# Remove loan id column (irrelevant)

# In[3]:


df.drop(["Loan_ID"], axis="columns", inplace=True)
df.dropna(inplace=True)


# In[4]:


df


# edit and convert number on string data to numeric

# In[ ]:





# convert y/n and male/female data to 1/0

# In[5]:


#df['Self_Employed'] = df['Self_Employed'].replace({'Yes': 1, 'No': 0})
#df['Dependents'] = pd.factorize(df['Dependents'])[0] + 1
df['Property_Area'] = pd.factorize(df['Property_Area'])[0] + 1
df['Loan_Status'] = df['Loan_Status'].replace({'Y': 1, 'N': 0})
df['Education'] = df['Education'].replace({'Graduate': 1, "Not Graduate": 0})
df["Credit_History"] = pd.to_numeric(df['Credit_History'], errors='coerce').astype(int)
df["LoanAmount"] = pd.to_numeric(df['LoanAmount'], errors='coerce').astype(int)


# base on dataset source, loan amount are written in thousands so we will use the real number

# In[6]:


df["LoanAmount"] = df.LoanAmount*1000


# look at features correlation and remove unperformed features

# In[7]:


#df.drop([ "Loan_Amount_Term", "Self_Employed"], axis="columns", inplace=True)


# In[8]:


df


# In[9]:


scaler = StandardScaler()
df[["ApplicantIncome", "LoanAmount"]] = scaler.fit_transform(df[["ApplicantIncome", "LoanAmount"]])


# each column impact on loan status visualization

# In[10]:


pd.crosstab(df['Dependents'], df.Loan_Status).plot(kind="bar")
pd.crosstab(df['Education'], df.Loan_Status).plot(kind="bar")
pd.crosstab(df['Credit_History'], df.Loan_Status).plot(kind="bar")
pd.crosstab(df['Property_Area'], df.Loan_Status).plot(kind="bar")


# In[11]:


df[df.Loan_Status == 1].shape


# In[12]:


df[df.Loan_Status == 0].shape


# <b> 4. Data Preparation

# In[13]:


#Extracting Independent and dependent Variable  
X = df.drop(["Loan_Status"], axis=1)
y = df['Loan_Status']


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=64)


# <b> 5. Create LOGISTIC Regression model

# In[15]:


#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression  
#classifier= LogisticRegression(random_state=0)  
classifier.fit(X_train, y_train)  


# In[ ]:


regression = LogisticRegression(C=1, penalty='l1', solver='liblinear')
regression.fit(X_train, y_train)
regression.score(X_test, y_test)


# In[ ]:


#Predicting the test set result  
y_pred= classifier.predict(X_test)  


# In[ ]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


# In[16]:


cm


# In[17]:


import seaborn as sns
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status',data=df)


# In[18]:


#with open('logistic_loan.pkl', 'wb') as f:
    #pickle.dump(regression, f)


# In[19]:


#scaler.transform([[4500,3000]])


# In[20]:


#regression.predict([0,1,1,scaler.transform([[4500,3000]]),1,3])


# In[21]:


df


# In[22]:


pickle.dump(regression, open('logisticloan.pkl','wb'))


# In[23]:


pickled_model = pickle.load(open('logisticloan.pkl', 'rb'))
pickled_model.predict(X_test)


# In[ ]:




