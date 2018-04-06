
# coding: utf-8

# In[2]:


import math
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn import ensemble
from sklearn import metrics
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.options.display.max_columns = None


# # Load Preprocessed Data
# This is the training data preprocessed as explained in the individual report

# In[3]:


df_train = pd.read_csv("encoded_train.csv")


# In[5]:


df_train = df_train.drop("Unnamed: 0", axis=1)
df_train.head()


# ## SAMPLING

# In[359]:


df_train_clicked = df_train[df_train["click"] == 1]
df_train_noclick = df_train[df_train["click"] == 0]
df_train_noclick_sample = df_train_noclick.sample(frac=1, random_state=3)
df_train = pd.concat([df_train_clicked, df_train_noclick_sample])
df_train.head()


# In[360]:


X = df_train.drop(["click"], axis=1)
y = df_train["click"]
X.tail()


# ## Training

# In[363]:


# clf = linear_model.LogisticRegression()
# clf = ensemble.GradientBoostingClassifier()
clf = xgb.XGBClassifier(n_jobs=3)


# In[364]:


clf.fit(X, y)


# In[365]:


fig, ax = plt.subplots(figsize=(20,10)) 
xgb.plot_importance(clf,ax=ax)


# # Testing with Validation Set

# In[368]:


df_valid = pd.read_csv("validation.csv")
df_valid.head(10)


# In[369]:


df_valid = df_valid.drop(["bidid","IP","userid","domain","url","urlid","slotid","slotformat",
             "slotvisibility","creative","bidprice","payprice","keypage","region","city"], axis=1)
df_valid['os'], df_valid['browser'] = df_valid['useragent'].str.split('_', 1).str
df_valid = df_valid.drop(["useragent"], axis=1)
df_ad = pd.get_dummies(df_valid['adexchange'])
df_ad.columns = [str(col) + '_a' for col in df_ad.columns]
df_hour = pd.get_dummies(df_valid['hour'])
df_hour.columns = [str(col) + '_h' for col in df_hour.columns]
df_os = pd.get_dummies(df_valid['os'])
df_os.columns = [str(col) + '_o' for col in df_os.columns]
df_tag = df_valid["usertag"].str.get_dummies(sep=",")
df_valid = pd.concat([df_valid, 
                      df_os,
                      pd.get_dummies(df_valid['browser']),
                      pd.get_dummies(df_valid['weekday']),
                      df_hour,
                     df_ad,
                     pd.get_dummies(df_valid['advertiser']),
                     df_tag], axis=1)
df_valid = df_valid.drop(["weekday","adexchange","advertiser","hour","os","browser","usertag"], axis=1)
df_valid["slotarea"] = df_valid["slotheight"] * df_valid["slotwidth"]
df_valid["slotratio"] = df_valid["slotheight"] / df_valid["slotwidth"]
df_valid = df_valid.replace('null',np.nan).dropna()


# In[370]:


pd.DataFrame([X.columns, df_valid.columns[1:]])


# In[371]:


df_valid.head()


# In[372]:


res = clf.predict_proba(df_valid.drop("click", axis=1))
res


# In[373]:


res_df = pd.DataFrame(res)
res_df.head()


# In[374]:


res_df["pred"] = res_df[1].apply(lambda x: 1 if x > 0.01 else 0)
res_df[res_df["pred"] == 1].shape


# In[375]:


metrics.confusion_matrix(df_valid["click"], res_df["pred"])


# In[376]:


print(metrics.classification_report(df_valid["click"], res_df["pred"]))


# In[377]:


avg_ctr = res_df[1].mean()
avg_ctr


# In[380]:


xdf = pd.DataFrame()
xdf["click"] = df_valid["click"]
xdf["pred"] = res_df[1]
xdf[xdf["click"] == 1].boxplot(column="pred")


# In[381]:


xdf[xdf["click"] == 0].boxplot(column="pred")


# In[382]:


xdf[xdf["click"] == 1].median()


# In[383]:


xdf[xdf["click"] == 0].median()


# In[393]:


metrics.roc_auc_score(xdf["click"], xdf["pred"])


# # BIDDING

# In[386]:


base_bid = 90
df = pd.read_csv("validation.csv")
df.drop(["bidid","IP","userid","domain","url","urlid","slotid","slotformat",
             "slotvisibility","creative","bidprice","keypage","usertag"], axis=1)
df["newbid"] = res_df[1].apply(lambda x: base_bid*(x/avg_ctr))

df['win'] = (df['newbid'] >= df['payprice']).astype(int)
df['paid'] = df['win'] * df['payprice']
df['cum_paid'] = df['paid'].cumsum()
df = df[df['cum_paid'] <= 6250000]

wins = df["win"].astype(int).sum()
clicks = df[df["win"] == 1]["click"].astype(int).sum()
cost = df[(df["newbid"] >= df["payprice"])]["payprice"].sum()/1000
win_rate = wins / df.shape[0]
ctr = (clicks / wins)
cpm = (cost / wins)
cpc = (cost / clicks) if clicks > 0 else -1
bids_processed = df.shape[0]
new_row = [wins, clicks, cost, win_rate, ctr, cpm, cpc, bids_processed]
new_row


# # Generating Bid Values for Testing Set

# In[ ]:


df_test = pd.read_csv("test.csv")
df_test = df_test.drop(["bidid","IP","userid","domain","url","urlid","slotid","slotformat",
             "slotvisibility","creative","keypage","region","city"], axis=1)
df_test['os'], df_test['browser'] = df_test['useragent'].str.split('_', 1).str
df_test = df_test.drop(["useragent"], axis=1)
df_ad = pd.get_dummies(df_test['adexchange'])
df_ad.columns = [str(col) + '_a' for col in df_ad.columns]
df_hour = pd.get_dummies(df_test['hour'])
df_hour.columns = [str(col) + '_h' for col in df_hour.columns]
df_os = pd.get_dummies(df_test['os'])
df_os.columns = [str(col) + '_o' for col in df_os.columns]
df_tag = df_test["usertag"].str.get_dummies(sep=",")
df_test = pd.concat([df_test, 
                      df_os,
                      pd.get_dummies(df_test['browser']),
                      pd.get_dummies(df_test['weekday']),
                      df_hour,
                     df_ad,
                     pd.get_dummies(df_test['advertiser']),
                     df_tag], axis=1)
df_test = df_test.drop(["weekday","adexchange","advertiser","hour","os","browser","usertag"], axis=1)
df_test["slotarea"] = df_test["slotheight"] * df_test["slotwidth"]
df_test["slotratio"] = df_test["slotheight"] / df_test["slotwidth"]
df_test = df_test.replace('null',np.nan).dropna()

res = clf.predict_proba(df_test)
res_df = pd.DataFrame(res)
base_bid = 89
bids = res_df[1].apply(lambda x: base_bid*(x/avg_ctr))
df = pd.read_csv("test.csv")
df["bidprice"] = bids
df[["bidid","bidprice"]].head()

