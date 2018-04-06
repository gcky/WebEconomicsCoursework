
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[101]:


df = pd.read_csv("train.csv")


# In[102]:


df.columns


# In[103]:


pd.options.display.max_columns = None
df.head()


# In[104]:


df2 = df[df["click"] == 1]
df.shape


# In[105]:


useragent_df = df["useragent"]


# fig, ax = plt.subplots()
# useragent_count = df["useragent"].value_counts()
# useragent_count.plot(figsize=(18, 16), ax=ax, kind='bar')

# useragent_count

# os_df = useragent_df
# os_df = os_df.apply(lambda x: x.split("_")[0])
# os_count = os_df.value_counts()
# fig, ax = plt.subplots()
# os_count.plot(ax=ax, kind='bar')

# In[110]:


dfx = pd.DataFrame(df["advertiser"].value_counts()).sort_index()
dfx


# In[107]:


df_ad = df[["payprice","advertiser"]]
df_ad.groupby("advertiser").sum()


# In[9]:


df_clicked = df[df["click"]==1]


# In[10]:


df_clicked[["click", "advertiser"]].groupby("advertiser").count()


# In[11]:


df[(df["weekday"]==0) & (df["advertiser"] == 1458)].shape[0]


# In[12]:


df[df["advertiser"]==1458]["weekday"].value_counts()


# # WEEKDAY VS CTR

# In[13]:


advertiser1 = 3427
ad1_weekday_ctr = pd.DataFrame()
ad1_weekday_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("weekday").sum()["click"]
ad1_weekday_ctr["count"] = df[df["advertiser"]==advertiser1]["weekday"].value_counts()
ad1_weekday_ctr["CTR"] = ad1_weekday_ctr["click"] / ad1_weekday_ctr["count"]
ad1_weekday_ctr["weekday"] = ad1_weekday_ctr.index


# In[14]:


ad1_weekday_ctr


# In[15]:


advertiser2 = 3476
ad2_weekday_ctr = pd.DataFrame()
ad2_weekday_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("weekday").sum()["click"]
ad2_weekday_ctr["count"] = df[df["advertiser"]==advertiser2]["weekday"].value_counts()
ad2_weekday_ctr["CTR"] = ad2_weekday_ctr["click"] / ad2_weekday_ctr["count"]
ad2_weekday_ctr["weekday"] = ad2_weekday_ctr.index


# In[16]:


ad2_weekday_ctr


# In[17]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Weekday')
plt.ylabel('CTR')
plt.plot(ad1_weekday_ctr["weekday"], ad1_weekday_ctr["CTR"])
plt.plot(ad2_weekday_ctr["weekday"], ad2_weekday_ctr["CTR"])


# # HOUR VS CTR

# In[18]:


ad1_hour_ctr = pd.DataFrame()
ad1_hour_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("hour").sum()["click"]
ad1_hour_ctr["count"] = df[df["advertiser"]==advertiser1]["hour"].value_counts()
ad1_hour_ctr["CTR"] = ad1_hour_ctr["click"] / ad1_hour_ctr["count"]
ad1_hour_ctr["hour"] = ad1_hour_ctr.index
ad1_hour_ctr


# In[19]:


ad2_hour_ctr = pd.DataFrame()
ad2_hour_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("hour").sum()["click"]
ad2_hour_ctr["count"] = df[df["advertiser"]==advertiser2]["hour"].value_counts()
ad2_hour_ctr["CTR"] = ad2_hour_ctr["click"] / ad2_hour_ctr["count"]
ad2_hour_ctr["hour"] = ad2_hour_ctr.index
ad2_hour_ctr


# In[20]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Hour')
plt.ylabel('CTR')
plt.plot(ad1_hour_ctr["hour"], ad1_hour_ctr["CTR"])
plt.plot(ad2_hour_ctr["hour"], ad2_hour_ctr["CTR"])


# # OS VS CTR

# In[21]:


os_df = useragent_df 
os_df = os_df.apply(lambda x: x.split("_")[0]) 
os_count = os_df.value_counts() 
fig, ax = plt.subplots() 
os_count.plot(ax=ax, kind='bar')


# In[22]:


df["os"] = df["useragent"].apply(lambda x: x.split("_")[0])
df["browser"] = df["useragent"].apply(lambda x: x.split("_")[1])


# In[23]:


ad1_os_ctr = pd.DataFrame()
ad1_os_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("os").sum()["click"]
ad1_os_ctr["count"] = df[df["advertiser"]==advertiser1]["os"].value_counts()
ad1_os_ctr["CTR"] = ad1_os_ctr["click"] / ad1_os_ctr["count"]
ad1_os_ctr["os"] = ad1_os_ctr.index
ad1_os_ctr


# In[24]:


ad2_os_ctr = pd.DataFrame()
ad2_os_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("os").sum()["click"]
ad2_os_ctr["count"] = df[df["advertiser"]==advertiser2]["os"].value_counts()
ad2_os_ctr["CTR"] = ad2_os_ctr["click"] / ad2_os_ctr["count"]
ad2_os_ctr["os"] = ad2_os_ctr.index
ad2_os_ctr


# In[25]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('OS')
plt.ylabel('CTR')
plt.plot(ad1_os_ctr["os"], ad1_os_ctr["CTR"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_os_ctr["os"], ad2_os_ctr["CTR"], marker='x', linestyle='None', markersize = 10.0)


# # BROWSER VS CTR

# In[26]:


ad1_browser_ctr = pd.DataFrame()
ad1_browser_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("browser").sum()["click"]
ad1_browser_ctr["count"] = df[df["advertiser"]==advertiser1]["browser"].value_counts()
ad1_browser_ctr["CTR"] = ad1_browser_ctr["click"] / ad1_browser_ctr["count"]
ad1_browser_ctr["browser"] = ad1_browser_ctr.index
ad1_browser_ctr


# In[27]:


ad2_browser_ctr = pd.DataFrame()
ad2_browser_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("browser").sum()["click"]
ad2_browser_ctr["count"] = df[df["advertiser"]==advertiser2]["browser"].value_counts()
ad2_browser_ctr["CTR"] = ad2_browser_ctr["click"] / ad2_browser_ctr["count"]
ad2_browser_ctr["browser"] = ad2_browser_ctr.index
ad2_browser_ctr


# In[28]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Browser')
plt.ylabel('CTR')
plt.plot(ad1_browser_ctr["browser"], ad1_browser_ctr["CTR"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_browser_ctr["browser"], ad2_browser_ctr["CTR"], marker='x', linestyle='None', markersize = 10.0)


# # REGION VS CTR

# In[29]:


ad1_region_ctr = pd.DataFrame()
ad1_region_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("region").sum()["click"]
ad1_region_ctr["count"] = df[df["advertiser"]==advertiser1]["region"].value_counts()
ad1_region_ctr["CTR"] = ad1_region_ctr["click"] / ad1_region_ctr["count"]
ad1_region_ctr["region"] = ad1_region_ctr.index
ad1_region_ctr.sort_values(['region'])
ad1_region_ctr["region"] = ad1_region_ctr["region"].apply(str)
ad1_region_ctr


# In[30]:


ad2_region_ctr = pd.DataFrame()
ad2_region_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("region").sum()["click"]
ad2_region_ctr["count"] = df[df["advertiser"]==advertiser2]["region"].value_counts()
ad2_region_ctr["CTR"] = ad2_region_ctr["click"] / ad2_region_ctr["count"]
ad2_region_ctr["region"] = ad2_region_ctr.index
ad2_region_ctr.sort_values(['region'])
ad2_region_ctr["region"] = ad2_region_ctr["region"].apply(str)
ad2_region_ctr


# In[31]:


fig, ax = plt.subplots(figsize=(20,10)) 
plt.grid(True)
plt.xlabel('Region')
plt.ylabel('CTR')
plt.plot(ad1_region_ctr["region"], ad1_region_ctr["CTR"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_region_ctr["region"], ad2_region_ctr["CTR"], marker='x', linestyle='None', markersize = 10.0)


# # AD EXCHANGE VS CTR

# In[33]:


ad1_exchange_ctr = pd.DataFrame()
ad1_exchange_ctr["click"] = df[df["advertiser"]==advertiser1].groupby("adexchange").sum()["click"]
ad1_exchange_ctr["count"] = df[df["advertiser"]==advertiser1]["adexchange"].value_counts()
ad1_exchange_ctr["CTR"] = ad1_exchange_ctr["click"] / ad1_exchange_ctr["count"]
ad1_exchange_ctr["adexchange"] = ad1_exchange_ctr.index
ad1_exchange_ctr.sort_values(['adexchange'])
ad1_exchange_ctr["adexchange"] = ad1_exchange_ctr["adexchange"].apply(str)
ad1_exchange_ctr


# In[34]:


ad2_exchange_ctr = pd.DataFrame()
ad2_exchange_ctr["click"] = df[df["advertiser"]==advertiser2].groupby("adexchange").sum()["click"]
ad2_exchange_ctr["count"] = df[df["advertiser"]==advertiser2]["adexchange"].value_counts()
ad2_exchange_ctr["CTR"] = ad2_exchange_ctr["click"] / ad2_exchange_ctr["count"]
ad2_exchange_ctr["adexchange"] = ad2_exchange_ctr.index
ad2_exchange_ctr.sort_values(['adexchange'])
ad2_exchange_ctr["adexchange"] = ad2_exchange_ctr["adexchange"].apply(str)
ad2_exchange_ctr


# In[37]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Ad Exchange')
plt.ylabel('CTR')
plt.plot(ad1_exchange_ctr["adexchange"], ad1_exchange_ctr["CTR"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_exchange_ctr["adexchange"], ad2_exchange_ctr["CTR"], marker='x', linestyle='None', markersize = 10.0)


# # PRICE VS WEEKDAY

# In[38]:


ad1_weekday_price = pd.DataFrame()
ad1_weekday_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("weekday").mean()["payprice"]
ad1_weekday_price["weekday"] = ad1_weekday_price.index
ad1_weekday_price


# In[39]:


ad2_weekday_price = pd.DataFrame()
ad2_weekday_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("weekday").mean()["payprice"]
ad2_weekday_price["weekday"] = ad2_weekday_price.index
ad2_weekday_price


# In[40]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Weekday')
plt.ylabel('Price')
plt.plot(ad1_weekday_price["weekday"], ad1_weekday_price["payprice"])
plt.plot(ad2_weekday_price["weekday"], ad2_weekday_price["payprice"])


# # HOUR VS PRICE

# In[42]:


ad1_hour_price = pd.DataFrame()
ad1_hour_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("hour").mean()["payprice"]
ad1_hour_price["hour"] = ad1_hour_price.index
ad1_hour_price.head()


# In[43]:


ad2_hour_price = pd.DataFrame()
ad2_hour_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("hour").mean()["payprice"]
ad2_hour_price["hour"] = ad2_hour_price.index
ad2_hour_price.head()


# In[45]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Hour')
plt.ylabel('Price')
plt.plot(ad1_hour_price["hour"], ad1_hour_price["payprice"])
plt.plot(ad2_hour_price["hour"], ad2_hour_price["payprice"])


# # OS VS PRICE

# In[46]:


ad1_os_price = pd.DataFrame()
ad1_os_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("os").mean()["payprice"]
ad1_os_price["os"] = ad1_os_price.index
ad1_os_price.head()


# In[50]:


ad2_os_price = pd.DataFrame()
ad2_os_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("os").mean()["payprice"]
ad2_os_price["os"] = ad2_os_price.index
ad2_os_price.head()


# In[49]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('OS')
plt.ylabel('Price')
plt.plot(ad1_os_price["os"], ad1_os_price["payprice"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_os_price["os"], ad2_os_price["payprice"], marker='x', linestyle='None', markersize = 10.0)


# # BROWSER VS PRICE

# In[51]:


ad1_browser_price = pd.DataFrame()
ad1_browser_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("browser").mean()["payprice"]
ad1_browser_price["browser"] = ad1_browser_price.index
ad1_browser_price.head()


# In[52]:


ad2_browser_price = pd.DataFrame()
ad2_browser_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("browser").mean()["payprice"]
ad2_browser_price["browser"] = ad2_browser_price.index
ad2_browser_price.head()


# In[53]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Browser')
plt.ylabel('Price')
plt.plot(ad1_browser_price["browser"], ad1_browser_price["payprice"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_browser_price["browser"], ad2_browser_price["payprice"], marker='x', linestyle='None', markersize = 10.0)


# # REGION VS PRICE

# In[59]:


ad1_region_price = pd.DataFrame()
ad1_region_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("region").mean()["payprice"]
ad1_region_price["region"] = ad1_region_price.index
ad1_region_price.sort_values(['region'])
ad1_region_price["region"] = ad1_region_price["region"].apply(str)
ad1_region_price.head()


# In[60]:


ad2_region_price = pd.DataFrame()
ad2_region_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("region").mean()["payprice"]
ad2_region_price["region"] = ad2_region_price.index
ad2_region_price.sort_values(['region'])
ad2_region_price["region"] = ad2_region_price["region"].apply(str)
ad2_region_price.head()


# In[61]:


fig, ax = plt.subplots(figsize=(20,10)) 
plt.grid(True)
plt.xlabel('Region')
plt.ylabel('Price')
plt.plot(ad1_region_price["region"], ad1_region_price["payprice"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_region_price["region"], ad2_region_price["payprice"], marker='x', linestyle='None', markersize = 10.0)


# # AD EXCHANGE VS PRICE

# In[62]:


ad1_exchange_price = pd.DataFrame()
ad1_exchange_price["payprice"] = df[df["advertiser"]==advertiser1].groupby("adexchange").mean()["payprice"]
ad1_exchange_price["exchange"] = ad1_exchange_price.index
ad1_exchange_price.sort_values(['exchange'])
ad1_exchange_price["exchange"] = ad1_exchange_price["exchange"].apply(str)
ad1_exchange_price.head()


# In[63]:


ad2_exchange_price = pd.DataFrame()
ad2_exchange_price["payprice"] = df[df["advertiser"]==advertiser2].groupby("adexchange").mean()["payprice"]
ad2_exchange_price["exchange"] = ad2_exchange_price.index
ad2_exchange_price.sort_values(['exchange'])
ad2_exchange_price["exchange"] = ad2_exchange_price["exchange"].apply(str)
ad2_exchange_price.head()


# In[65]:


fig, ax = plt.subplots() 
plt.grid(True)
plt.xlabel('Ad Exchange')
plt.ylabel('Price')
plt.plot(ad1_exchange_price["exchange"], ad1_exchange_price["payprice"], marker='.', linestyle='None', markersize = 10.0)
plt.plot(ad2_exchange_price["exchange"], ad2_exchange_price["payprice"], marker='x', linestyle='None', markersize = 10.0)


# # WEEKDAY VS ECPC

# In[74]:


ad1_weekday_ecpc = pd.DataFrame()
ad1_weekday_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("weekday").sum()["payprice"]
ad1_weekday_ecpc["clicks"] = df[df["advertiser"]==advertiser2].groupby("weekday").sum()["click"]
ad1_weekday_ecpc["ecpc"] = (ad1_weekday_ecpc["cost"] / 1000) / ad1_weekday_ecpc["clicks"]
ad1_weekday_ecpc["weekday"] = ad1_weekday_price.index
ad1_weekday_ecpc


# In[75]:


fig, ax = plt.subplots() 
# plt.grid(True)
plt.xlabel('Weekday')
plt.ylabel('eCPC')
plt.bar(ad1_weekday_ecpc["weekday"], ad1_weekday_ecpc["ecpc"])


# # HOUR VS ECPC

# In[76]:


ad1_hour_ecpc = pd.DataFrame()
ad1_hour_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("hour").sum()["payprice"]
ad1_hour_ecpc["clicks"] = df[df["advertiser"]==advertiser2].groupby("hour").sum()["click"]
ad1_hour_ecpc["ecpc"] = (ad1_hour_ecpc["cost"] / 1000) / ad1_hour_ecpc["clicks"]
ad1_hour_ecpc["hour"] = ad1_hour_price.index
ad1_hour_ecpc.head()


# In[77]:


fig, ax = plt.subplots() 
# plt.grid(True)
plt.xlabel('Hour')
plt.ylabel('eCPC')
plt.bar(ad1_hour_ecpc["hour"], ad1_hour_ecpc["ecpc"])


# # OS VS ECPC

# In[79]:


ad2_os_ecpc = pd.DataFrame()
ad2_os_ecpc["click"] = df[df["advertiser"]==advertiser2].groupby("os").sum()["click"]
ad2_os_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("os").sum()["payprice"]
ad2_os_ecpc["ecpc"] = (ad2_os_ecpc["cost"] / 1000) / ad2_os_ecpc["click"]
ad2_os_ecpc["os"] = ad2_os_ecpc.index
ad2_os_ecpc


# In[81]:


fig, ax = plt.subplots() 
# plt.grid(True)
plt.xlabel('OS')
plt.ylabel('eCPC')
plt.bar(ad2_os_ecpc["os"], ad2_os_ecpc["ecpc"])


# # BROWSER VS ECPC

# In[83]:


ad2_browser_ecpc = pd.DataFrame()
ad2_browser_ecpc["click"] = df[df["advertiser"]==advertiser2].groupby("browser").sum()["click"]
ad2_browser_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("browser").sum()["payprice"]
ad2_browser_ecpc["ecpc"] = (ad2_browser_ecpc["cost"] / 1000) / ad2_browser_ecpc["click"]
ad2_browser_ecpc["browser"] = ad2_browser_ecpc.index
ad2_browser_ecpc


# In[84]:


fig, ax = plt.subplots() 
# plt.grid(True)
plt.xlabel('Browser')
plt.ylabel('eCPC')
plt.bar(ad2_browser_ecpc["browser"], ad2_browser_ecpc["ecpc"])


# # REGION VS ECPC

# In[95]:


ad2_region_ecpc = pd.DataFrame()
ad2_region_ecpc["click"] = df[df["advertiser"]==advertiser2].groupby("region").sum()["click"]
ad2_region_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("region").sum()["payprice"]
ad2_region_ecpc["ecpc"] = (ad2_region_ecpc["cost"] / 1000) / ad2_region_ecpc["click"]
ad2_region_ecpc["region"] = ad2_region_ecpc.index
ad2_region_ecpc.sort_values(['region'])
ad2_region_ecpc["region"] = ad2_region_ecpc["region"].apply(str)
ad2_region_ecpc.head()


# In[96]:


fig, ax = plt.subplots(figsize=(20,10)) 
# plt.grid(True)
plt.xlabel('Region')
plt.ylabel('eCPC')
plt.bar(ad2_region_ecpc["region"], ad2_region_ecpc["ecpc"])


# # AD EXCHANGE VS ECPC

# In[97]:


ad2_adexchange_ecpc = pd.DataFrame()
ad2_adexchange_ecpc["click"] = df[df["advertiser"]==advertiser2].groupby("adexchange").sum()["click"]
ad2_adexchange_ecpc["cost"] = df[df["advertiser"]==advertiser2].groupby("adexchange").sum()["payprice"]
ad2_adexchange_ecpc["ecpc"] = (ad2_adexchange_ecpc["cost"] / 1000) / ad2_adexchange_ecpc["click"]
ad2_adexchange_ecpc["adexchange"] = ad2_adexchange_ecpc.index
ad2_adexchange_ecpc.sort_values(['adexchange'])
ad2_adexchange_ecpc["adexchange"] = ad2_adexchange_ecpc["adexchange"].apply(str)
ad2_adexchange_ecpc.head()


# In[98]:


fig, ax = plt.subplots() 
# plt.grid(True)
plt.xlabel('Ad Exchange')
plt.ylabel('eCPC')
plt.bar(ad2_adexchange_ecpc["adexchange"], ad2_adexchange_ecpc["ecpc"])

