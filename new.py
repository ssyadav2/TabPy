#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
from sklearn.utils import resample
from xgboost import XGBClassifier
import tabpy_client


# In[98]:


df=pd.read_csv("dummy_data.csv")


# In[99]:


df = df[(df["max_age"]>0) & (df['max_age']<110) & (df["min_age"]>0) & (df["min_age"]<110)]


# In[100]:


features_drop=["hshld_id", "cal_dt", "cal_month", "randn"]


# In[101]:


df.drop(features_drop, axis=1, inplace=True)


# In[102]:


# Removing unvarying variables
df.drop(["crophailcount", "act_ann_prem_ind", "farm_ind", "Unnamed: 0.1", "Unnamed: 0"], axis=1, inplace=True)


# In[103]:


# performing over-sampling

y1=df[df["umbrella_xsell_3q"]==0]
y2=df[df["umbrella_xsell_3q"]==1]

y2_mod=resample(y2, replace=True, n_samples=len(y1))
df_bal=pd.concat([y1, y2_mod])


# In[104]:


# Separating features and target class
Y=df_bal["umbrella_xsell_3q"]
X=df_bal.drop(["umbrella_xsell_3q"], axis=1)


# In[59]:


model = XGBClassifier()


# In[60]:


model.fit(X.values, Y.values)


# In[75]:


def PCS(homecount, prim_amount, autocount, vehcount, max_age):
    renterscount = 0
    farmcount = 0
    life10count = 0
    life20count = 1
    life30count = 0
    termlifecount = 0
    wholelifecount = 3
    fixedulcount = 0
    individualhealthcount = 0
    longtermcarecount = 0
    medsupcount = 0
    disabilityincomecount = 0
    commercialcount = 0
    financialcount = 1
    federalcropcount = 0
    married_ind = 1
    min_age = 35
    hh_tenure = 10
    months_since_last_purch = 16
    num_persons = 3
    stip_ann_prem_ind = 0
    home_owner_0_ind = 0
    home_owner_1_ind = 1
    drvr_cnt = 2
    max_bi_limit = 300000
    nbr_of_sq_feet = 1700
    clm_cnt_lst_12mo = 0
    clm_cnt_lst_36mo = 1
    clm_cnt_lst_60mo = 1
    clm_cnt_lst_84mo = 1
    X = np.column_stack([int(autocount), int(homecount), int(renterscount), int(farmcount),
       life10count, life20count, life30count, termlifecount,
       wholelifecount, fixedulcount, individualhealthcount,
       longtermcarecount, medsupcount, disabilityincomecount,
       commercialcount, financialcount, federalcropcount, int(max_age),
       min_age, hh_tenure, months_since_last_purch, married_ind,
       num_persons, stip_ann_prem_ind, home_owner_0_ind,
       home_owner_1_ind, drvr_cnt, int(vehcount), max_bi_limit,
       int(prim_amount), nbr_of_sq_feet, clm_cnt_lst_12mo,
       clm_cnt_lst_36mo, clm_cnt_lst_60mo, clm_cnt_lst_84mo])
    a = model.predict(X).tolist()
    b = model.predict_proba(X).tolist()
    c = max(b[0][0],b[0][1])
    a.append(c)    
    return a


# In[73]:

time.sleep(240)
connection = tabpy_client.Client('http://tp-server-cfd.herokuapp.com/')
connection.deploy('PCS',PCS,
                  'Predicting probability of cross-sell from given data', override=True)

# In[ ]:




