<h1 style="font-weight: 900; color: #fff; text-align:center;">Covid-19 Data Analyst - Data Analyst with Python</h1>

<style>
    h2{
        color:bisque;
        text-decoration:underline;
    }
    li{
        font-size: 16px;
        margin: 10px 0;
    }
    span{
        font-weight: 700;
    }
</style>
<h2 style="font-weight: bold; font-size: 20px;">💻 Problem Solving: </h2>
<ul>
    <li>
        <span>Question 01:</span> Show the number of Confirmed, Deaths and Recovered cases in each Region.
    </li>
    <li>
        <span>Question 02:</span> Remove all the records where the Confirmed Cases is Less Than 10.
    </li>
    <li>
        <span>Question 03:</span> In which Region, maximum number of Confirmed cases were recorded ?
    </li>
    <li>
        <span>Question 04:</span> In which Region, minimum number of Deaths cases were recorded ?
    </li>
    <li>
        <span>Question 05:</span> How many Confirmed, Deaths & Recovered cases were reported from India till 29 April 2020 ?
    </li>
    <li>
        <span>Question 06:</span> Sort the entire data wrt No. of Confirmed cases in ascending order.
    </li>
    <li>
        <span>Question 07:</span> Sort the entire data wrt No. of Recovered cases in descending order.
    </li>
</ul>

<h1>------------------</h1>

<h2 style="font-weight: 900; font-size: 18px;">Exploratory Data Analyst</h2>


```python
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
# read data from data.csv file
df = pd.read_csv("data.csv")
```


```python
# read first line of dataset data.csv
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Algeria</td>
      <td>3848</td>
      <td>444</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Andorra</td>
      <td>743</td>
      <td>42</td>
      <td>423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Angola</td>
      <td>27</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# show basic information
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 321 entries, 0 to 320
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   Date       321 non-null    object
     1   State      140 non-null    object
     2   Region     321 non-null    object
     3   Confirmed  321 non-null    int64 
     4   Deaths     321 non-null    int64 
     5   Recovered  321 non-null    int64 
    dtypes: int64(3), object(3)
    memory usage: 15.2+ KB



```python
# return statistics of data from dataframe
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>321.000000</td>
      <td>321.000000</td>
      <td>321.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9949.800623</td>
      <td>709.152648</td>
      <td>3030.277259</td>
    </tr>
    <tr>
      <th>std</th>
      <td>31923.853086</td>
      <td>3236.162817</td>
      <td>14364.870365</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>104.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>653.000000</td>
      <td>12.000000</td>
      <td>73.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4655.000000</td>
      <td>144.000000</td>
      <td>587.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>299691.000000</td>
      <td>27682.000000</td>
      <td>132929.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# show shape of dataframe
df.shape
```




    (321, 6)




```python
# couting missing value
df.isna().sum()
```




    Date           0
    State        181
    Region         0
    Confirmed      0
    Deaths         0
    Recovered      0
    dtype: int64




```python
# get all columns
df.columns
```




    Index(['Date', 'State', 'Region', 'Confirmed', 'Deaths', 'Recovered'], dtype='object')




```python
# check data types for all columns of dataframe
df.dtypes
```




    Date         object
    State        object
    Region       object
    Confirmed     int64
    Deaths        int64
    Recovered     int64
    dtype: object



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 01: </span>Show the number of Confirmed, Deaths and Recovered cases in each Region.</h3>


```python
# reshow dataframe
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
  </tbody>
</table>
</div>




```python
answer_01 = df.groupby("Region")[['Confirmed', 'Recovered', 'Deaths']].sum()
answer_01
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Confirmed</th>
      <th>Recovered</th>
      <th>Deaths</th>
    </tr>
    <tr>
      <th>Region</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Afghanistan</th>
      <td>1939</td>
      <td>252</td>
      <td>60</td>
    </tr>
    <tr>
      <th>Albania</th>
      <td>766</td>
      <td>455</td>
      <td>30</td>
    </tr>
    <tr>
      <th>Algeria</th>
      <td>3848</td>
      <td>1702</td>
      <td>444</td>
    </tr>
    <tr>
      <th>Andorra</th>
      <td>743</td>
      <td>423</td>
      <td>42</td>
    </tr>
    <tr>
      <th>Angola</th>
      <td>27</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>West Bank and Gaza</th>
      <td>344</td>
      <td>71</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Western Sahara</th>
      <td>6</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Yemen</th>
      <td>6</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Zambia</th>
      <td>97</td>
      <td>54</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Zimbabwe</th>
      <td>32</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>187 rows × 3 columns</p>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 02: </span>Remove all the records where the Confirmed Cases is Less Than 10.</h3>


```python
# reshow dataframe
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
  </tbody>
</table>
</div>




```python
answer_02 = df[~(df['Confirmed'] < 10)]
answer_02.sort_values(by=['Confirmed'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>156</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Suriname</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>70</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Holy See</td>
      <td>10</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>59</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Gambia</td>
      <td>10</td>
      <td>1</td>
      <td>8</td>
    </tr>
    <tr>
      <th>318</th>
      <td>4/29/2020</td>
      <td>Yukon</td>
      <td>Canada</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>217</th>
      <td>4/29/2020</td>
      <td>Greenland</td>
      <td>Denmark</td>
      <td>11</td>
      <td>0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>France</td>
      <td>165093</td>
      <td>24087</td>
      <td>48228</td>
    </tr>
    <tr>
      <th>168</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>UK</td>
      <td>165221</td>
      <td>26097</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Italy</td>
      <td>203591</td>
      <td>27682</td>
      <td>71252</td>
    </tr>
    <tr>
      <th>153</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Spain</td>
      <td>236899</td>
      <td>24275</td>
      <td>132929</td>
    </tr>
    <tr>
      <th>265</th>
      <td>4/29/2020</td>
      <td>New York</td>
      <td>US</td>
      <td>299691</td>
      <td>23477</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>304 rows × 6 columns</p>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 03: </span> In which Region, maximum number of Confirmed cases were recorded ?</h3>


```python
# reshow dataframe
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Region')['Confirmed'].sum().sort_values(ascending=False).head()
```




    Region
    US        1039909
    Spain      236899
    Italy      203591
    France     166543
    UK         166441
    Name: Confirmed, dtype: int64



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 04: </span> In which Region, minimum number of Deaths cases were recorded ?</h3>


```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Algeria</td>
      <td>3848</td>
      <td>444</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Andorra</td>
      <td>743</td>
      <td>42</td>
      <td>423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Angola</td>
      <td>27</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Region')['Deaths'].sum().sort_values(ascending=True).head()
```




    Region
    Laos          0
    Mongolia      0
    Mozambique    0
    Cambodia      0
    Fiji          0
    Name: Deaths, dtype: int64



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 05: </span>  How many Confirmed, Deaths & Recovered cases were reported from India till 29 April 2020 ?</h3>


```python
# reshow dataframe
df.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4/29/2020</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Date'] = pd.to_datetime(df['Date'])
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Algeria</td>
      <td>3848</td>
      <td>444</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Andorra</td>
      <td>743</td>
      <td>42</td>
      <td>423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Angola</td>
      <td>27</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df[(df['Date'] == '2020-04-29') & (df['Region'] == 'India')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>74</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>India</td>
      <td>33062</td>
      <td>1079</td>
      <td>8437</td>
    </tr>
  </tbody>
</table>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 06: </span>  Sort the entire data wrt No. of Confirmed cases in ascending order.</h3>


```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Algeria</td>
      <td>3848</td>
      <td>444</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Andorra</td>
      <td>743</td>
      <td>42</td>
      <td>423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Angola</td>
      <td>27</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by=['Confirmed'], ascending=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>285</th>
      <td>2020-04-29</td>
      <td>Recovered</td>
      <td>US</td>
      <td>0</td>
      <td>0</td>
      <td>120720</td>
    </tr>
    <tr>
      <th>284</th>
      <td>2020-04-29</td>
      <td>Recovered</td>
      <td>Canada</td>
      <td>0</td>
      <td>0</td>
      <td>20327</td>
    </tr>
    <tr>
      <th>203</th>
      <td>2020-04-29</td>
      <td>Diamond Princess cruise ship</td>
      <td>Canada</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>305</th>
      <td>2020-04-29</td>
      <td>Tibet</td>
      <td>Mainland China</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>289</th>
      <td>2020-04-29</td>
      <td>Saint Pierre and Miquelon</td>
      <td>France</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>France</td>
      <td>165093</td>
      <td>24087</td>
      <td>48228</td>
    </tr>
    <tr>
      <th>168</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>UK</td>
      <td>165221</td>
      <td>26097</td>
      <td>0</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Italy</td>
      <td>203591</td>
      <td>27682</td>
      <td>71252</td>
    </tr>
    <tr>
      <th>153</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Spain</td>
      <td>236899</td>
      <td>24275</td>
      <td>132929</td>
    </tr>
    <tr>
      <th>265</th>
      <td>2020-04-29</td>
      <td>New York</td>
      <td>US</td>
      <td>299691</td>
      <td>23477</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>321 rows × 6 columns</p>
</div>



<h3 style="font-size: 18px; "><span style="font-weight: 900;">Question 07: </span>  Sort the entire data wrt No. of Recovered cases in descending order.</h3>


```python
# reshow dataframe
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>1939</td>
      <td>60</td>
      <td>252</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Albania</td>
      <td>766</td>
      <td>30</td>
      <td>455</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Algeria</td>
      <td>3848</td>
      <td>444</td>
      <td>1702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Andorra</td>
      <td>743</td>
      <td>42</td>
      <td>423</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Angola</td>
      <td>27</td>
      <td>2</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sort_values(by=['Recovered'], ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>State</th>
      <th>Region</th>
      <th>Confirmed</th>
      <th>Deaths</th>
      <th>Recovered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>153</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Spain</td>
      <td>236899</td>
      <td>24275</td>
      <td>132929</td>
    </tr>
    <tr>
      <th>285</th>
      <td>2020-04-29</td>
      <td>Recovered</td>
      <td>US</td>
      <td>0</td>
      <td>0</td>
      <td>120720</td>
    </tr>
    <tr>
      <th>61</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Germany</td>
      <td>161539</td>
      <td>6467</td>
      <td>120400</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Iran</td>
      <td>93657</td>
      <td>5957</td>
      <td>73791</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2020-04-29</td>
      <td>NaN</td>
      <td>Italy</td>
      <td>203591</td>
      <td>27682</td>
      <td>71252</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>248</th>
      <td>2020-04-29</td>
      <td>Maryland</td>
      <td>US</td>
      <td>20849</td>
      <td>1078</td>
      <td>0</td>
    </tr>
    <tr>
      <th>246</th>
      <td>2020-04-29</td>
      <td>Manitoba</td>
      <td>Canada</td>
      <td>275</td>
      <td>6</td>
      <td>0</td>
    </tr>
    <tr>
      <th>243</th>
      <td>2020-04-29</td>
      <td>Louisiana</td>
      <td>US</td>
      <td>27660</td>
      <td>1845</td>
      <td>0</td>
    </tr>
    <tr>
      <th>241</th>
      <td>2020-04-29</td>
      <td>Kentucky</td>
      <td>US</td>
      <td>4537</td>
      <td>234</td>
      <td>0</td>
    </tr>
    <tr>
      <th>215</th>
      <td>2020-04-29</td>
      <td>Grand Princess</td>
      <td>Canada</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>321 rows × 6 columns</p>
</div>




```python
np.random.randint(10, size=6)
```




    array([1, 0, 6, 2, 8, 5])




```python

```
