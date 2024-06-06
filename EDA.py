#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.ticker as mtick  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
os.chdir("C:/Users/DELL/PycharmProjects/DataScience")
from sklearn import preprocessing


# In[36]:


df = pd.read_csv('Churn_Modelling.csv')


# In[32]:


df.head(5)


# In[8]:


# NUMERICAL ANALYSIS


# In[9]:


df.corr()


# In[10]:


plt.figure(figsize=(20,8))
df.corr()['Exited'].sort_values(ascending=False).plot(kind='bar')


# In[13]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr(), cmap='Paired')


# In[14]:


df['Age'].value_counts().sort_index(ascending=True).plot()


# In[16]:


df1 = df.loc[df["Exited"]==1]
df1['Age'].value_counts().sort_index(ascending=True).plot()


# In[ ]:


# FEATURE BINNING


# In[18]:


df.info()


# In[33]:


df.drop(columns=['CustomerId','RowNumber','Surname'], axis=0, inplace=True)


# In[34]:


df.head(5)


# In[37]:


df.Age.min()


# In[38]:


df.Age.max()


# In[39]:


labels = ['0-20', '21-40', '41-60', 'Above 61']
bins = [0, 20, 40, 60, 100 ]

df['Age_bins'] = pd.cut(df.Age, bins, labels = labels, include_lowest=True)


# In[40]:


df.head(5)


# In[41]:


df.Age_bins.value_counts()


# In[42]:


plt.bar(labels, df.Age_bins.value_counts())
plt.title('Age Count')
plt.xlabel('Age Bins')
plt.ylabel('Age Count')


# In[ ]:


######### HANDLING MISSING VALUES #############


# In[43]:


df.Gender.mode()


# In[44]:


df['Gender'] = df['Gender'].fillna('Male')


# In[45]:


df.Gender.value_counts()


# In[47]:


#### LABEL ENCODING


# In[48]:


le = preprocessing.LabelEncoder()
df["Gender_label"] = le.fit_transform(df.Gender.values)


# In[49]:


df.Gender_label.value_counts()


# # ONE HOT CODING

# In[51]:


one_hot = pd.get_dummies(df['Geography'])


# In[52]:


one_hot


# In[53]:


df_dumm = pd.get_dummies(df)
df_dumm.head(5)


# # Target Encoding

# In[54]:


pip install category encoders


# In[60]:


pip install category_encoders


# In[61]:


from category_encoders import TargetEncoder


# In[63]:


encoder = TargetEncoder()

df2 = pd.read_csv('Churn_Modelling.csv')


# In[64]:


df2.drop(columns=['CustomerId','RowNumber','Surname'], axis=0, inplace=True)
df2['Gender'] = df2['Gender'].fillna('Male')


# In[65]:


df2.head(5)


# In[66]:


encoder = TargetEncoder()

df2['Gender_Encoded'] = encoder.fit_transform(df2['Gender'], df2['Exited'])


# In[67]:


df2.head(5)


# In[68]:


df2['Gender_Encoded'].value_counts()


# # Hash Encoding

# In[69]:


from category_encoders import HashingEncoder


# In[70]:


x = df.Gender
y = df.Exited


# In[72]:


ce_hash = HashingEncoder(cols = ['Gender'])
ce_hash.fit_transform(x,y)


# In[73]:


cc = pd.read_csv('CustomerChurn.csv')


# In[74]:


cc.head(5)


# In[76]:


cc.shape


# In[77]:


cc.columns.values


# In[78]:


cc.dtypes


# In[79]:


cc.describe()


# In[81]:


cc['Churn'].value_counts()


# In[82]:


cc['Churn'].value_counts()/len(cc)*100


# In[83]:


cc['Churn'].value_counts().plot(kind='barh', figsize=(8, 6))
plt.xlabel("Count", labelpad=14)
plt.ylabel("Target Variable", labelpad=14)
plt.title("Count of TARGET Variable per category", y=1.02)


# In[87]:


missing = pd.DataFrame((cc.isnull().sum())*100/cc.shape[0]).reset_index()
plt.figure(figsize=(16,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =7)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show()


# Data Cleaning

# In[88]:


c = cc.copy()


# Total charges should be of numeric type. Let's convert it to a numeric data type

# In[89]:


c.TotalCharges = pd.to_numeric(c.TotalCharges, errors='coerce')
c.isnull().sum()


# Since the % of these records compared to total dataset is very low ie 0.15%, it is safe to ignore them from further processing.

# In[90]:


c.dropna(how = 'any', inplace = True)


# Divide customers into bins based on tenure e.g. for tenure < 12 months: assign a tenure group if 1-12, for tenure between 1 to 2 Yrs, tenure group of 13-24; so on...

# In[91]:


c['tenure'].max()


# In[92]:


lab = ["{0}-{1}".format(i, i+11) for i in range(1, 72, 12)]
c['tenure_group'] = pd.cut(c.tenure, range(1, 80, 12), right=False, labels = lab)


# In[93]:


c['tenure_group'].value_counts()


# Remove columns not required for processing

# In[94]:


c.drop(columns=['customerID', 'tenure'], axis = 1, inplace = True)


# In[95]:


c.head(5)


# **Data Exploration**

# 1. **Univariate Analysis**

# In[96]:


for i, predictor in enumerate(c.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=c, x=predictor, hue='Churn')


# **Numerical Analysis**

# In[98]:


c.gender.value_counts()


# In[99]:


c_tgt0 = c[c['Churn']=='No']
c_tgt1 = c[c['Churn']=='Yes']


# In[100]:


c_tgt1.gender.value_counts()


# In[101]:


pd.crosstab(c.PaymentMethod, c.Churn)


# 2. Conver the target variable Churn in a binary numeric valkue i.e. Yes=1, No=0

# In[102]:


c['Churn'] = np.where(c.Churn == 'Yes', 1, 0)


# In[103]:


c.head(5)


# 3. Convert all categorical values into dummy variables

# In[104]:


c_dum = pd.get_dummies(c)
c_dum.head()


# Relation between monthly and toatal charges

# In[108]:


sns.lmplot(data=c_dum, x='MonthlyCharges', y='TotalCharges', fit_reg=False)


# In[109]:


c_dum['MonthlyCharges'].corr(c_dum['TotalCharges'])


# Churn by monthly charges and total charges

# In[110]:


Mth = sns.kdeplot(c_dum.MonthlyCharges[(c_dum["Churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(c_dum.MonthlyCharges[(c_dum["Churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')


# **Insight** : Churn is high when the monthly charges are high

# In[111]:


Tot = sns.kdeplot(c_dum.TotalCharges[(c_dum["Churn"] == 0) ],
                color="Red", shade = True)
Tot = sns.kdeplot(c_dum.TotalCharges[(c_dum["Churn"] == 1) ],
                ax =Tot, color="Blue", shade= True)
Tot.legend(["No Churn","Churn"],loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Total Charges')
Tot.set_title('Total charges by churn')


# **Surprising insight** :  as higher Churn at lower Total Charges

#  **Build a corelation of all predictors with 'Churn'**

# In[112]:


plt.figure(figsize=(20,8))
c_dum.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# **Derived Insight:**
# 
# HIGH Churn seen in case of Month to month contracts, No online security, No Tech support, First year of subscription and Fibre Optics Internet
# 
# LOW Churn is seens in case of Long term contracts, Subscriptions without internet service and The customers engaged for 5+ years
# 
# Factors like Gender, Availability of PhoneService and # of multiple lines have alomost NO impact on Churn.

# **This is also evident from the Heatmap below**

# In[113]:


plt.figure(figsize=(12,12))
sns.heatmap(c_dum.corr(), cmap="Paired")


# **Bivariate Analysis**

# In[114]:


c_tgt0 = c.loc[c["Churn"]==0]
c_tgt1 = c.loc[c["Churn"]==1]


# In[115]:


len(c_tgt0)


# In[116]:


len(c_tgt1)


# In[117]:


def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()


# In[118]:


uniplot(c_tgt1,col='Partner',title='Distribution of Gender for Churned Customers',hue='gender')


# In[120]:


uniplot(c_tgt0,col='Partner',title='Distribution of Gender for Non Churned Customers',hue='gender')


# In[121]:


uniplot(c_tgt1,col='PaymentMethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')


# In[122]:


uniplot(c_tgt1,col='Contract',title='Distribution of Contract for Churned Customers',hue='gender')


# In[123]:


uniplot(c_tgt1,col='TechSupport',title='Distribution of TechSupport for Churned Customers',hue='gender')


# In[ ]:




