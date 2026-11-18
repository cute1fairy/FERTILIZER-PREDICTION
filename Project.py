#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score,auc,roc_curve,RocCurveDisplay,classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize,StandardScaler
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')


# In[3]:


data = pd.read_csv(r'data_core.csv')


# In[4]:


data.head(5)


# In[5]:


data.shape


# In[6]:


data.dtypes


# In[7]:


data.rename(columns={'Soil Type':'Soil_Type','Crop Type':'Crop_Type',
                     'Fertilizer Name':'Fertilizer_Name','Temparature':'Temperature'},inplace=True)


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum()


# In[10]:


data['Soil_Type'] = data['Soil_Type'].astype('category')
data['Crop_Type'] = data['Crop_Type'].astype('category')
data['Fertilizer_Name'] = data['Fertilizer_Name'].astype('category')


# In[11]:


data.info()


# In[12]:


data['Fertilizer_Name'].unique()


# In[13]:


data.Crop_Type.unique()


# In[14]:


data.Soil_Type.unique()


# In[15]:


data['Temperature'].describe()


# In[16]:


data.info()


# In[17]:


data['Crop_Type'].value_counts()


# In[18]:


data.groupby('Fertilizer_Name')[['Nitrogen', 'Phosphorous', 'Potassium']].mean()


# In[19]:


sns.histplot(data.Nitrogen)
plt.show()


# #### Inference: The Nitrogen histogram shows the distribution of values.
# ### - Lowest observed value: 0
# ### - Highest observed value: 46
# ### - Average value: 18.43
# ### This helps understand how the data is spread and whether the values are concentrated towards
# ### lower or higher ranges.

# In[20]:


sns.histplot(data.Phosphorous)
plt.show()


# #### Inference: The Phosphorous histogram shows the distribution of values.
# #### - Lowest observed value: 0
# #### - Highest observed value: 46
# #### - Average value: 18.51
# #### This helps understand how the data is spread and whether the values are concentrated towards
# #### lower or higher ranges

# In[21]:


sns.histplot(data.Potassium)
plt.show()


# #### Inference: The Potassium histogram shows the distribution of values.
# #### - Lowest observed value: 0
# #### - Highest observed value: 23
# #### - Average value: 3.92
# #### This helps understand how the data is spread and whether the values are concentrated towards
# #### lower or higher ranges.

# In[22]:


sns.histplot(data.Moisture)
plt.show()


# #### Inference: The Moisture histogram shows the distribution of values.
# #### - Lowest observed value: 20.0
# #### - Highest observed value: 70.0
# #### - Average value: 43.58
# #### This helps understand how the data is spread and whether the values are concentrated towards
# #### lower or higher ranges.

# In[23]:


sns.histplot(data.Temperature)
plt.show()


# #### Inference: The Temparature histogram shows the distribution of values.
# #### - Lowest observed value: 20.0
# #### - Highest observed value: 40.0
# #### - Average value: 30.34
# #### This helps understand how the data is spread and whether the values are concentrated towards
# #### lower or higher ranges.

# In[24]:


sns.histplot(data.Humidity)
plt.show()


# #### Inference: The Humidity histogram shows the distribution of values.
# #### - Lowest observed value: 40.02
# #### - Highest observed value: 80.0
# #### - Average value: 59.21
# #### This helps understand how the data is spread and whether the values are concentrated towards
# #### lower or higher ranges.

# In[25]:


sns.histplot(data.Fertilizer_Name)


# In[26]:


le = LabelEncoder()
data['Crop_Type'] = le.fit_transform(data['Crop_Type'])
data['Soil_Type'] = le.fit_transform(data['Soil_Type'])
data['Fertilizer_Name'] = le.fit_transform(data['Fertilizer_Name'])


# In[27]:


sns.heatmap(data.corr()[['Fertilizer_Name']].sort_values(by='Fertilizer_Name',ascending=False),annot=True)


# In[28]:


X = data.drop('Fertilizer_Name', axis=1)


# In[29]:


y = data['Fertilizer_Name']


# In[30]:


y


# In[31]:


Y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5,6])
Y


# In[32]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)


# In[33]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[34]:


lr =LogisticRegression()


# In[35]:


cls = OneVsRestClassifier(lr)


# In[36]:


cls.fit(X_train,Y_train)


# In[37]:


y_pred = cls.predict(X_test)
y_pred


# In[38]:


fpr = dict()
tpr = dict()
roc_auc = dict()


# In[39]:


y_pred.shape


# In[40]:


Y_test.shape


# In[41]:


n_classes=7


# In[42]:


for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])


# In[43]:


for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],label="Roc curve area:{}".format(roc_auc[i]))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='upper right')


# In[44]:


accuracy_score(Y_test,y_pred)


# In[45]:


print(classification_report(Y_test, y_pred))


# In[46]:


rf = RandomForestClassifier()


# In[47]:


cls = OneVsRestClassifier(rf)


# In[48]:


cls.fit(X_train,Y_train)


# In[49]:


y1_pred = cls.predict(X_test)


# In[50]:


fpr = dict()
tpr = dict()
roc_auc = dict()


# In[51]:


n_classes=7


# In[52]:


for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y1_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])


# In[53]:


for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],label="Roc curve area:{}".format(roc_auc[i]))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='upper right')


# In[54]:


accuracy_score(Y_test,y1_pred)


# In[55]:


print(classification_report(Y_test, y1_pred))


# In[56]:


Ab = AdaBoostClassifier()


# In[57]:


cls = OneVsRestClassifier(Ab)


# In[58]:


cls.fit(X_train,Y_train)


# In[59]:


y2_pred = cls.predict(X_test)


# In[60]:


fpr = dict()
tpr = dict()
roc_auc = dict()


# In[61]:


n_classes=7


# In[62]:


for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y2_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])


# In[63]:


for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],label="Roc curve area:{}".format(roc_auc[i]))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='upper right')


# In[64]:


accuracy_score(Y_test,y2_pred)


# In[65]:


print(classification_report(Y_test, y2_pred))


# In[66]:


Gb = GradientBoostingClassifier()


# In[67]:


cls = OneVsRestClassifier(Gb)


# In[68]:


cls.fit(X_train,Y_train)


# In[69]:


y3_pred = cls.predict(X_test)
y3_pred


# In[70]:


fpr = dict()
tpr = dict()
roc_auc = dict()


# In[71]:


n_classes=7


# In[72]:


for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y3_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])


# In[73]:


for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],label="Roc curve area:{}".format(roc_auc[i]))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='upper right')


# In[74]:


accuracy_score(Y_test,y3_pred)


# In[75]:


print(classification_report(Y_test, y3_pred))


# In[76]:


Dt = DecisionTreeClassifier()


# In[77]:


Dt.fit(X_train,Y_train)


# In[78]:


y4_pred = Dt.predict(X_test)


# In[79]:


fpr = dict()
tpr = dict()
roc_auc = dict()


# In[80]:


n_classes=7


# In[81]:


for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],y4_pred[:,i])
    roc_auc[i] =auc(fpr[i],tpr[i])


# In[82]:


for i in range(n_classes):
    plt.plot(fpr[i],tpr[i],label="Roc curve area:{}".format(roc_auc[i]))
    plt.plot([0,1],[0,1],'k--')
    plt.xlim(0,1.05)
    plt.ylim(0,1.05)
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.legend(loc='upper right')


# In[83]:


accuracy_score(Y_test,y4_pred)


# In[84]:


import pickle
pickle.dump(Dt, open("model.pkl", "wb"))


# In[85]:


pickle.dump(scaler, open("scaler.pkl", "wb"))


# In[86]:


import os
os.getcwd()


# In[87]:


os.listdir()


# In[88]:


import numpy as np
print(np.unique(y_pred))


# In[89]:


print(np.unique(Y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




