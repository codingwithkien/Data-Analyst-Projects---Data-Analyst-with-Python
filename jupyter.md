#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-weight: 900; color: #fff; text-align:center;">Weather Data Analyst - Data Analyst with Python</h1>

# <style>
#     h2{
#         color:bisque;
#         text-decoration:underline;
#     }
#     li{
#         font-size: 16px;
#         margin: 10px 0;
#     }
#     span{
#         font-weight: 700;
#     }
# </style>
# <h2 style="font-weight: bold; font-size: 20px;">💻 Problem Solving: </h2>
# <ul>
#     <li>
#         <span>Question 01:</span> Find all the unique 'Wind Speed' values in the data.
#     </li>
#     <li>
#         <span>Question 02:</span> Find the number of times when the 'Weather is exactly Clear'
#     </li>
#     <li>
#         <span>Question 03:</span> Find the number of times when the 'Wind Speed was exactly 4 km/h'.
#     </li>
#     <li>
#         <span>Question 04:</span> Find out all the Null Values in the data.
#     </li>
#     <li>
#         <span>Question 05:</span> Rename the column name 'Weather' of the dataframe to 'Weather Condition'.
#     </li>
#     <li>
#         <span>Question 06:</span> What is the mean 'Visibility' ?
#     </li>
#     <li>
#         <span>Question 07:</span> What is the Standard Deviation of 'Pressure'  in this data?
#     </li>
#     <li>
#         <span>Question 08:</span> What is the Variance of 'Relative Humidity' in this data ?
#     </li>
#     <li>
#         <span>Question 09:</span> Find all instances when 'Snow' was recorded.
#     </li>
#     <li>
#         <span>Question 10:</span> Find all instances when 'Wind Speed is above 24' and 'Visibility is 25'.
#     </li>
#     <li>
#         <span>Question 11:</span> What is the Mean value of each column against each 'Weather Condition ?
#     </li>
#     <li>
#         <span>Question 12:</span> What is the Minimum & Maximum value of each column against each 'Weather Condition ?
#     </li>
#     <li>
#         <span>Question 13:</span> Show all the Records where Weather Condition is Fog.
#     </li>
#     <li>
#         <span>Question 14:</span> Find all instances when 'Weather is Clear' or 'Visibility is above 40'.
#     </li>
#     <li>
#         <span>Question 15:</span> Find all instances when 'Weather is Clear' and 'Relative Humidity is greater than 50'
#     </li>
# </ul>

# <h1>------------------</h1>

# <h2 style="font-weight: 900; font-size: 18px;">Exploratory Data Analyst</h2>

# In[1]:


# import important libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


# read dataset from 'data.csv' file
df = pd.read_csv('data.csv')


# In[3]:


# read first line of dataset data.csv
df.head()


# In[4]:


# show basic information about dataframe
df.info()


# In[5]:


# return statistics description of data from dataframe
df.describe()


# In[6]:


# show shape of dataframe
df.shape


# In[7]:


# get all columns of dataframe
df.columns


# In[8]:


# check data types for all columns of dataframe
df.dtypes


# In[9]:


# convert data type of Date/Time column to datetime
df['Date/Time'] = pd.to_datetime(df['Date/Time'])


# In[10]:


# check datetypes again
df.dtypes


# In[11]:


# get index value
df.index


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 01: </span>Find all the unique 'Wind Speed' values in the data.</h3>

# In[12]:


# reshow dataframe
df.head(2) # two first line of dataframe


# In[13]:


# answer the question 01
asn_01 = df['Wind Speed_km/h'].unique()
asn_01


# In[14]:


# sorted in ascending order
np.sort(asn_01)

# len(np.sort(asn_01)) # 34


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 02: </span>Find the number of times when the 'Weather is exactly Clear'.</h3>

# In[15]:


# reshow dataframe
df.head(2) # first two lines of dataframe


# In[16]:


# answer the question 02
ans_02 = df[df['Weather'] == 'Clear']
ans_02.shape[0]


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 03: </span>Find the number of times when the 'Wind Speed was exactly 4 km/h'.</h3>

# In[17]:


# reshow dataframe
df.head(2)


# In[18]:


# answer the question 03
ans_03 = df[df['Wind Speed_km/h'] == 4]
ans_03.shape[0]


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 04: </span>Find out all the Null Values in the data.</h3>
# 

# In[19]:


# reshow two first line of dataframe
df.head(2)


# In[20]:


# answer the question 04
df.isna().sum()


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 05: </span>Rename the column name 'Weather' of the dataframe to 'Weather Condition'.</h3>
# 

# In[21]:


# reshow first line of dataframe
df.head(2)


# In[22]:


# answer the question 05
df.rename(columns={'Weather' : 'Weather Conditions'}, inplace=True)
df.head()


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 06: </span>What is the mean 'Visibility' ?</h3>
# 

# In[23]:


# reshow the dataframe
df.head(2)


# In[24]:


# answer the question 06
ans_06 = df['Visibility_km'].mean()
round(ans_06, 2)


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 07: </span>What is the Standard Deviation of 'Pressure' in this data?</h3>
# 

# In[25]:


# reshow the dataframe
df.head(2)


# In[26]:


# answer the question 07
ans_07 = df.Press_kPa.std()
ans_07


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 08: </span>What is the variance of 'Relative Humidity'?</h3>
# 

# In[27]:


# reshow the dataframe
df.head(2)


# In[28]:


# answer the question 08
ans_08 = df['Rel Hum_%'].var()
ans_08


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 09: </span>Find all instances when "Snow" was recorded</h3>
# 

# In[29]:


# reshow the dataframe
df.head(2)


# In[30]:


# answer the question 09
ans_09 = df[df['Weather Conditions'] == 'Snow']
ans_09
# ans_09.shape # (390, 8)


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 10: </span>Find all instances when 'Wind Speed is above 24' and 'Visibility is 25'.</h3>
# 

# In[31]:


# reshow the dataframe
df.head(2)


# In[32]:


# answer the question 10
ans_10 = df[(df['Wind Speed_km/h'] > 24) & (df['Visibility_km'] == 25)]
ans_10


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 11: </span>What is the Mean value of each column against each 'Weather Condition ?</h3>
# 

# In[33]:


# reshow the dataframe
df.head(2)


# In[34]:


# drop Date/Time column in dataframe
df_drop_datetime = df.drop('Date/Time', axis=1)


# In[35]:


# answer the question 11
ans_11 = df_drop_datetime.groupby('Weather Conditions').mean()
ans_11


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 12: </span>What is the Minimum & Maximum value of each column against each 'Weather Condition ?</h3>
# 

# In[36]:


# reshow the dataframe drop Date/Time column
df_drop_datetime.head(2)


# In[37]:


# answer the question 12
ans_12 = df_drop_datetime.groupby('Weather Conditions').agg(['min', 'max'])
ans_12


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 13: </span>Show all the Records where Weather Condition is Fog.</h3>
# 

# In[38]:


# reshow the orgin dataframe
df.head(2)


# In[39]:


# answer the question 13
ans_13 = df[df['Weather Conditions'] == 'Fog']
ans_13


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 14: </span>Find all instances when 'Weather is Clear' or 'Visibility is above 40'.</h3>
# 

# In[40]:


# reshow the original dataframe
df.head(2)


# In[41]:


# answer the question 14
ans_14 = df[(df['Weather Conditions'] == 'Clear') & (df['Visibility_km'] > 40)]
ans_14


# <h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 15: </span>Find all instances when 'Weather is Clear' and 'Relative Humidity is greater than 50'.</h3>
# 

# In[42]:


# reshow the original dataframe
df.head(2)


# In[43]:


# answer the question 15
ans_15 = df[(df['Weather Conditions'] == 'Clear') & (df['Rel Hum_%'] > 50)]
ans_15

1
