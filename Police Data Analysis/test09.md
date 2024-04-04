# Linear Regression with SciKit-Learn

We saw how to create a very simple best fit line, but now let's greatly expand our toolkit to start thinking about the considerations of overfitting, underfitting, model evaluation, as well as multiple features!

## Imports


```python
# In[1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## Sample Data

This sample data is from ISLR. It displays sales (in thousands of units) for a particular product as a function of advertising budgets (in thousands of dollars) for TV, radio, and newspaper media.


```python
# In[2]
df = pd.read_csv("Advertising.csv")
```


```python
# In[3]
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.1</td>
      <td>37.8</td>
      <td>69.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.5</td>
      <td>41.3</td>
      <td>58.5</td>
      <td>18.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
      <td>12.9</td>
    </tr>
  </tbody>
</table>
</div>



### Expanding the Questions

Previously, we explored **Is there a relationship between *total* advertising spend and *sales*?** as well as predicting the total sales for some value of total spend. Now we want to expand this to **What is the relationship between each advertising channel (TV,Radio,Newspaper) and sales?**

### Multiple Features (N-Dimensional)


```python
# In[4]
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();
```
    



```python
# Relationships between features
sns.pairplot(df,diag_kind='kde')
```




    <seaborn.axisgrid.PairGrid at 0x216014fb648>


    


## Introducing SciKit Learn

We will work a lot with the scitkit learn library, so get comfortable with its model estimator syntax, as well as exploring its incredibly useful documentation!

---


```python
X = df.drop('sales',axis=1)
y = df['sales']
```

## Train | Test Split

Make sure you have watched the Machine Learning Overview videos on Supervised Learning to understand why we do this step


```python
from sklearn.model_selection import train_test_split
```


```python
# random_state: 
# https://stackoverflow.com/questions/28064634/random-state-pseudo-random-number-in-scikit-learn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```


```python
X_train
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>85</th>
      <td>193.2</td>
      <td>18.4</td>
      <td>65.7</td>
    </tr>
    <tr>
      <th>183</th>
      <td>287.6</td>
      <td>43.0</td>
      <td>71.8</td>
    </tr>
    <tr>
      <th>127</th>
      <td>80.2</td>
      <td>0.0</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>53</th>
      <td>182.6</td>
      <td>46.2</td>
      <td>58.7</td>
    </tr>
    <tr>
      <th>100</th>
      <td>222.4</td>
      <td>4.3</td>
      <td>49.8</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>63</th>
      <td>102.7</td>
      <td>29.6</td>
      <td>8.4</td>
    </tr>
    <tr>
      <th>70</th>
      <td>199.1</td>
      <td>30.6</td>
      <td>38.7</td>
    </tr>
    <tr>
      <th>81</th>
      <td>239.8</td>
      <td>4.1</td>
      <td>36.9</td>
    </tr>
    <tr>
      <th>11</th>
      <td>214.7</td>
      <td>24.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>95</th>
      <td>163.3</td>
      <td>31.6</td>
      <td>52.9</td>
    </tr>
  </tbody>
</table>
<p>140 rows × 3 columns</p>
</div>




```python
y_train
```




    85     15.2
    183    26.2
    127     8.8
    53     21.2
    100    11.7
           ... 
    63     14.0
    70     18.3
    81     12.3
    11     17.4
    95     16.9
    Name: sales, Length: 140, dtype: float64




```python
X_test
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>37</th>
      <td>74.7</td>
      <td>49.4</td>
      <td>45.7</td>
    </tr>
    <tr>
      <th>109</th>
      <td>255.4</td>
      <td>26.9</td>
      <td>5.5</td>
    </tr>
    <tr>
      <th>31</th>
      <td>112.9</td>
      <td>17.4</td>
      <td>38.6</td>
    </tr>
    <tr>
      <th>89</th>
      <td>109.8</td>
      <td>47.8</td>
      <td>51.4</td>
    </tr>
    <tr>
      <th>66</th>
      <td>31.5</td>
      <td>24.6</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>119</th>
      <td>19.4</td>
      <td>16.0</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>54</th>
      <td>262.7</td>
      <td>28.8</td>
      <td>15.9</td>
    </tr>
    <tr>
      <th>74</th>
      <td>213.4</td>
      <td>24.6</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>145</th>
      <td>140.3</td>
      <td>1.9</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>142</th>
      <td>220.5</td>
      <td>33.2</td>
      <td>37.9</td>
    </tr>
    <tr>
      <th>148</th>
      <td>38.0</td>
      <td>40.3</td>
      <td>11.9</td>
    </tr>
    <tr>
      <th>112</th>
      <td>175.7</td>
      <td>15.4</td>
      <td>2.4</td>
    </tr>
    <tr>
      <th>174</th>
      <td>222.4</td>
      <td>3.4</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>55</th>
      <td>198.9</td>
      <td>49.4</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>141</th>
      <td>193.7</td>
      <td>35.4</td>
      <td>75.6</td>
    </tr>
    <tr>
      <th>149</th>
      <td>44.7</td>
      <td>25.8</td>
      <td>20.6</td>
    </tr>
    <tr>
      <th>25</th>
      <td>262.9</td>
      <td>3.5</td>
      <td>19.5</td>
    </tr>
    <tr>
      <th>34</th>
      <td>95.7</td>
      <td>1.4</td>
      <td>7.4</td>
    </tr>
    <tr>
      <th>170</th>
      <td>50.0</td>
      <td>11.6</td>
      <td>18.4</td>
    </tr>
    <tr>
      <th>39</th>
      <td>228.0</td>
      <td>37.7</td>
      <td>32.0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>19.6</td>
      <td>20.1</td>
      <td>17.0</td>
    </tr>
    <tr>
      <th>153</th>
      <td>171.3</td>
      <td>39.7</td>
      <td>37.7</td>
    </tr>
    <tr>
      <th>175</th>
      <td>276.9</td>
      <td>48.9</td>
      <td>41.8</td>
    </tr>
    <tr>
      <th>61</th>
      <td>261.3</td>
      <td>42.7</td>
      <td>54.7</td>
    </tr>
    <tr>
      <th>65</th>
      <td>69.0</td>
      <td>9.3</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>50</th>
      <td>199.8</td>
      <td>3.1</td>
      <td>34.6</td>
    </tr>
    <tr>
      <th>42</th>
      <td>293.6</td>
      <td>27.7</td>
      <td>1.8</td>
    </tr>
    <tr>
      <th>129</th>
      <td>59.6</td>
      <td>12.0</td>
      <td>43.1</td>
    </tr>
    <tr>
      <th>179</th>
      <td>165.6</td>
      <td>10.0</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>23.8</td>
      <td>35.1</td>
      <td>65.9</td>
    </tr>
    <tr>
      <th>133</th>
      <td>219.8</td>
      <td>33.5</td>
      <td>45.1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>134.3</td>
      <td>4.9</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>22</th>
      <td>13.2</td>
      <td>15.9</td>
      <td>49.6</td>
    </tr>
    <tr>
      <th>41</th>
      <td>177.0</td>
      <td>33.4</td>
      <td>38.7</td>
    </tr>
    <tr>
      <th>32</th>
      <td>97.2</td>
      <td>1.5</td>
      <td>30.0</td>
    </tr>
    <tr>
      <th>125</th>
      <td>87.2</td>
      <td>11.8</td>
      <td>25.9</td>
    </tr>
    <tr>
      <th>196</th>
      <td>94.2</td>
      <td>4.9</td>
      <td>8.1</td>
    </tr>
    <tr>
      <th>158</th>
      <td>11.7</td>
      <td>36.9</td>
      <td>45.2</td>
    </tr>
    <tr>
      <th>180</th>
      <td>156.6</td>
      <td>2.6</td>
      <td>8.3</td>
    </tr>
    <tr>
      <th>16</th>
      <td>67.8</td>
      <td>36.6</td>
      <td>114.0</td>
    </tr>
    <tr>
      <th>186</th>
      <td>139.5</td>
      <td>2.1</td>
      <td>26.6</td>
    </tr>
    <tr>
      <th>144</th>
      <td>96.2</td>
      <td>14.8</td>
      <td>38.9</td>
    </tr>
    <tr>
      <th>121</th>
      <td>18.8</td>
      <td>21.7</td>
      <td>50.4</td>
    </tr>
    <tr>
      <th>80</th>
      <td>76.4</td>
      <td>26.7</td>
      <td>22.3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>69.2</td>
      <td>20.5</td>
      <td>18.3</td>
    </tr>
    <tr>
      <th>78</th>
      <td>5.4</td>
      <td>29.9</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>48</th>
      <td>227.2</td>
      <td>15.8</td>
      <td>49.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
    </tr>
    <tr>
      <th>15</th>
      <td>195.4</td>
      <td>47.7</td>
      <td>52.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
    </tr>
    <tr>
      <th>43</th>
      <td>206.9</td>
      <td>8.4</td>
      <td>26.4</td>
    </tr>
    <tr>
      <th>102</th>
      <td>280.2</td>
      <td>10.1</td>
      <td>21.4</td>
    </tr>
    <tr>
      <th>164</th>
      <td>117.2</td>
      <td>14.7</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>9</th>
      <td>199.8</td>
      <td>2.6</td>
      <td>21.2</td>
    </tr>
    <tr>
      <th>155</th>
      <td>4.1</td>
      <td>11.6</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>36</th>
      <td>266.9</td>
      <td>43.8</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>190</th>
      <td>39.5</td>
      <td>41.1</td>
      <td>5.8</td>
    </tr>
    <tr>
      <th>33</th>
      <td>265.6</td>
      <td>20.0</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>45</th>
      <td>175.1</td>
      <td>22.5</td>
      <td>31.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_test
```




    37     14.7
    109    19.8
    31     11.9
    89     16.7
    66      9.5
    119     6.6
    54     20.2
    74     17.0
    145    10.3
    142    20.1
    148    10.9
    112    14.1
    174    11.5
    55     23.7
    141    19.2
    149    10.1
    25     12.0
    34      9.5
    170     8.4
    39     21.5
    172     7.6
    153    19.0
    175    27.0
    61     24.2
    65      9.3
    50     11.4
    42     20.7
    129     9.7
    179    12.6
    2       9.3
    12      9.2
    133    19.6
    90     11.2
    22      5.6
    41     17.1
    32      9.6
    125    10.6
    196     9.7
    158     7.3
    180    10.5
    16     12.5
    186    10.3
    144    11.4
    121     7.0
    80     11.8
    18     11.3
    78      5.3
    48     14.8
    4      12.9
    15     22.4
    1      10.4
    43     12.9
    102    14.8
    164    11.9
    9      10.6
    155     3.2
    36     25.4
    190    10.8
    33     17.4
    45     14.9
    Name: sales, dtype: float64



## Creating a Model (Estimator)

#### Import a model class from a model family


```python
from sklearn.linear_model import LinearRegression
```

#### Create an instance of the model with parameters


```python
help(LinearRegression)
```

    Help on class LinearRegression in module sklearn.linear_model._base:
    
    class LinearRegression(sklearn.base.MultiOutputMixin, sklearn.base.RegressorMixin, LinearModel)
     |  LinearRegression(*, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
     |  
     |  Ordinary least squares Linear Regression.
     |  
     |  LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
     |  to minimize the residual sum of squares between the observed targets in
     |  the dataset, and the targets predicted by the linear approximation.
     |  
     |  Parameters
     |  ----------
     |  fit_intercept : bool, default=True
     |      Whether to calculate the intercept for this model. If set
     |      to False, no intercept will be used in calculations
     |      (i.e. data is expected to be centered).
     |  
     |  normalize : bool, default=False
     |      This parameter is ignored when ``fit_intercept`` is set to False.
     |      If True, the regressors X will be normalized before regression by
     |      subtracting the mean and dividing by the l2-norm.
     |      If you wish to standardize, please use
     |      :class:`sklearn.preprocessing.StandardScaler` before calling ``fit`` on
     |      an estimator with ``normalize=False``.
     |  
     |  copy_X : bool, default=True
     |      If True, X will be copied; else, it may be overwritten.
     |  
     |  n_jobs : int, default=None
     |      The number of jobs to use for the computation. This will only provide
     |      speedup for n_targets > 1 and sufficient large problems.
     |      ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
     |      ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
     |      for more details.
     |  
     |  Attributes
     |  ----------
     |  coef_ : array of shape (n_features, ) or (n_targets, n_features)
     |      Estimated coefficients for the linear regression problem.
     |      If multiple targets are passed during the fit (y 2D), this
     |      is a 2D array of shape (n_targets, n_features), while if only
     |      one target is passed, this is a 1D array of length n_features.
     |  
     |  rank_ : int
     |      Rank of matrix `X`. Only available when `X` is dense.
     |  
     |  singular_ : array of shape (min(X, y),)
     |      Singular values of `X`. Only available when `X` is dense.
     |  
     |  intercept_ : float or array of shape (n_targets,)
     |      Independent term in the linear model. Set to 0.0 if
     |      `fit_intercept = False`.
     |  
     |  See Also
     |  --------
     |  sklearn.linear_model.Ridge : Ridge regression addresses some of the
     |      problems of Ordinary Least Squares by imposing a penalty on the
     |      size of the coefficients with l2 regularization.
     |  sklearn.linear_model.Lasso : The Lasso is a linear model that estimates
     |      sparse coefficients with l1 regularization.
     |  sklearn.linear_model.ElasticNet : Elastic-Net is a linear regression
     |      model trained with both l1 and l2 -norm regularization of the
     |      coefficients.
     |  
     |  Notes
     |  -----
     |  From the implementation point of view, this is just plain Ordinary
     |  Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.
     |  
     |  Examples
     |  --------
     |  >>> import numpy as np
     |  >>> from sklearn.linear_model import LinearRegression
     |  >>> X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
     |  >>> # y = 1 * x_0 + 2 * x_1 + 3
     |  >>> y = np.dot(X, np.array([1, 2])) + 3
     |  >>> reg = LinearRegression().fit(X, y)
     |  >>> reg.score(X, y)
     |  1.0
     |  >>> reg.coef_
     |  array([1., 2.])
     |  >>> reg.intercept_
     |  3.0000...
     |  >>> reg.predict(np.array([[3, 5]]))
     |  array([16.])
     |  
     |  Method resolution order:
     |      LinearRegression
     |      sklearn.base.MultiOutputMixin
     |      sklearn.base.RegressorMixin
     |      LinearModel
     |      sklearn.base.BaseEstimator
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, *, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  fit(self, X, y, sample_weight=None)
     |      Fit linear model.
     |      
     |      Parameters
     |      ----------
     |      X : {array-like, sparse matrix} of shape (n_samples, n_features)
     |          Training data
     |      
     |      y : array-like of shape (n_samples,) or (n_samples, n_targets)
     |          Target values. Will be cast to X's dtype if necessary
     |      
     |      sample_weight : array-like of shape (n_samples,), default=None
     |          Individual weights for each sample
     |      
     |          .. versionadded:: 0.17
     |             parameter *sample_weight* support to LinearRegression.
     |      
     |      Returns
     |      -------
     |      self : returns an instance of self.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from sklearn.base.MultiOutputMixin:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.RegressorMixin:
     |  
     |  score(self, X, y, sample_weight=None)
     |      Return the coefficient of determination R^2 of the prediction.
     |      
     |      The coefficient R^2 is defined as (1 - u/v), where u is the residual
     |      sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
     |      sum of squares ((y_true - y_true.mean()) ** 2).sum().
     |      The best possible score is 1.0 and it can be negative (because the
     |      model can be arbitrarily worse). A constant model that always
     |      predicts the expected value of y, disregarding the input features,
     |      would get a R^2 score of 0.0.
     |      
     |      Parameters
     |      ----------
     |      X : array-like of shape (n_samples, n_features)
     |          Test samples. For some estimators this may be a
     |          precomputed kernel matrix or a list of generic objects instead,
     |          shape = (n_samples, n_samples_fitted),
     |          where n_samples_fitted is the number of
     |          samples used in the fitting for the estimator.
     |      
     |      y : array-like of shape (n_samples,) or (n_samples, n_outputs)
     |          True values for X.
     |      
     |      sample_weight : array-like of shape (n_samples,), default=None
     |          Sample weights.
     |      
     |      Returns
     |      -------
     |      score : float
     |          R^2 of self.predict(X) wrt. y.
     |      
     |      Notes
     |      -----
     |      The R2 score used when calling ``score`` on a regressor uses
     |      ``multioutput='uniform_average'`` from version 0.23 to keep consistent
     |      with default value of :func:`~sklearn.metrics.r2_score`.
     |      This influences the ``score`` method of all the multioutput
     |      regressors (except for
     |      :class:`~sklearn.multioutput.MultiOutputRegressor`).
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from LinearModel:
     |  
     |  predict(self, X)
     |      Predict using the linear model.
     |      
     |      Parameters
     |      ----------
     |      X : array_like or sparse matrix, shape (n_samples, n_features)
     |          Samples.
     |      
     |      Returns
     |      -------
     |      C : array, shape (n_samples,)
     |          Returns predicted values.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from sklearn.base.BaseEstimator:
     |  
     |  __getstate__(self)
     |  
     |  __repr__(self, N_CHAR_MAX=700)
     |      Return repr(self).
     |  
     |  __setstate__(self, state)
     |  
     |  get_params(self, deep=True)
     |      Get parameters for this estimator.
     |      
     |      Parameters
     |      ----------
     |      deep : bool, default=True
     |          If True, will return the parameters for this estimator and
     |          contained subobjects that are estimators.
     |      
     |      Returns
     |      -------
     |      params : mapping of string to any
     |          Parameter names mapped to their values.
     |  
     |  set_params(self, **params)
     |      Set the parameters of this estimator.
     |      
     |      The method works on simple estimators as well as on nested objects
     |      (such as pipelines). The latter have parameters of the form
     |      ``<component>__<parameter>`` so that it's possible to update each
     |      component of a nested object.
     |      
     |      Parameters
     |      ----------
     |      **params : dict
     |          Estimator parameters.
     |      
     |      Returns
     |      -------
     |      self : object
     |          Estimator instance.
    



```python
model = LinearRegression()
```

### Fit/Train the Model on the training data

**Make sure you only fit to the training data, in order to fairly evaluate your model's performance on future data**


```python
model.fit(X_train,y_train)
```




    LinearRegression()



# Understanding and utilizing the Model

-----

## Evaluation on the Test Set

### Metrics

Make sure you've viewed the video on these metrics!
The three most common evaluation metrics for regression problems:

**Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:

$$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$

**Mean Squared Error** (MSE) is the mean of the squared errors:

$$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$

**Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:

$$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$

Comparing these metrics:

- **MAE** is the easiest to understand, because it's the average error.
- **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
- **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.

All of these are **loss functions**, because we want to minimize them.

### Calculate Performance on Test Set

We want to fairly evaluate our model, so we get performance metrics on the test set (data the model has never seen before).


```python
# X_test
```


```python
# We only pass in test features
# The model predicts its own y hat
# We can then compare these results to the true y test label value
test_predictions = model.predict(X_test)
```


```python
test_predictions
```




    array([15.74131332, 19.61062568, 11.44888935, 17.00819787,  9.17285676,
            7.01248287, 20.28992463, 17.29953992,  9.77584467, 19.22194224,
           12.40503154, 13.89234998, 13.72541098, 21.28794031, 18.42456638,
            9.98198406, 15.55228966,  7.68913693,  7.55614992, 20.40311209,
            7.79215204, 18.24214098, 24.68631904, 22.82199068,  7.97962085,
           12.65207264, 21.46925937,  8.05228573, 12.42315981, 12.50719678,
           10.77757812, 19.24460093, 10.070269  ,  6.70779999, 17.31492147,
            7.76764327,  9.25393336,  8.27834697, 10.58105585, 10.63591128,
           13.01002595,  9.77192057, 10.21469861,  8.04572042, 11.5671075 ,
           10.08368001,  8.99806574, 16.25388914, 13.23942315, 20.81493419,
           12.49727439, 13.96615898, 17.56285075, 11.14537013, 12.56261468,
            5.50870279, 23.29465134, 12.62409688, 18.77399978, 15.18785675])




```python
from sklearn.metrics import mean_absolute_error,mean_squared_error
```


```python
MAE = mean_absolute_error(y_test,test_predictions)
MSE = mean_squared_error(y_test,test_predictions)
RMSE = np.sqrt(MSE)
```


```python
MAE
```




    1.213745773614481




```python
MSE
```




    2.2987166978863796




```python
RMSE
```




    1.5161519375993884




```python
df['sales'].mean()
```




    14.0225



**Review our video to understand whether these values are "good enough".**

## Residuals

Revisiting Anscombe's Quartet: https://en.wikipedia.org/wiki/Anscombe%27s_quartet

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Anscombe%27s_quartet_3.svg/850px-Anscombe%27s_quartet_3.svg.png">

<table class="wikitable">
<tbody><tr>
<th>Property
</th>
<th>Value
</th>
<th>Accuracy
</th></tr>
<tr>
<td><a href="/wiki/Mean" title="Mean">Mean</a> of <i>x</i>
</td>
<td>9
</td>
<td>exact
</td></tr>
<tr>
<td>Sample <a href="/wiki/Variance" title="Variance">variance</a> of <i>x</i>  :  <span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML"  alttext="{\displaystyle \sigma ^{2}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <msup>
          <mi>&#x03C3;<!-- σ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>2</mn>
          </mrow>
        </msup>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle \sigma ^{2}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/53a5c55e536acf250c1d3e0f754be5692b843ef5" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.338ex; width:2.385ex; height:2.676ex;" alt="\sigma ^{2}"/></span>
</td>
<td>11
</td>
<td>exact
</td></tr>
<tr>
<td>Mean of <i>y</i>
</td>
<td>7.50
</td>
<td>to 2 decimal places
</td></tr>
<tr>
<td>Sample variance of <i>y</i>  :  <span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML"  alttext="{\displaystyle \sigma ^{2}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <msup>
          <mi>&#x03C3;<!-- σ --></mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>2</mn>
          </mrow>
        </msup>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle \sigma ^{2}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/53a5c55e536acf250c1d3e0f754be5692b843ef5" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.338ex; width:2.385ex; height:2.676ex;" alt="\sigma ^{2}"/></span>
</td>
<td>4.125
</td>
<td>±0.003
</td></tr>
<tr>
<td><a href="/wiki/Correlation" class="mw-redirect" title="Correlation">Correlation</a> between <i>x</i> and <i>y</i>
</td>
<td>0.816
</td>
<td>to 3 decimal places
</td></tr>
<tr>
<td><a href="/wiki/Linear_regression" title="Linear regression">Linear regression</a> line
</td>
<td><i>y</i>&#160;=&#160;3.00&#160;+&#160;0.500<i>x</i>
</td>
<td>to 2 and 3 decimal places, respectively
</td></tr>
<tr>
<td><a href="/wiki/Coefficient_of_determination" title="Coefficient of determination">Coefficient of determination</a> of the linear regression  :  <span class="mwe-math-element"><span class="mwe-math-mathml-inline mwe-math-mathml-a11y" style="display: none;"><math xmlns="http://www.w3.org/1998/Math/MathML"  alttext="{\displaystyle R^{2}}">
  <semantics>
    <mrow class="MJX-TeXAtom-ORD">
      <mstyle displaystyle="true" scriptlevel="0">
        <msup>
          <mi>R</mi>
          <mrow class="MJX-TeXAtom-ORD">
            <mn>2</mn>
          </mrow>
        </msup>
      </mstyle>
    </mrow>
    <annotation encoding="application/x-tex">{\displaystyle R^{2}}</annotation>
  </semantics>
</math></span><img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/5ce07e278be3e058a6303de8359f8b4a4288264a" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -0.338ex; width:2.818ex; height:2.676ex;" alt="R^{2}"/></span>
</td>
<td>0.67
</td>
<td>to 2 decimal places
</td></tr></tbody></table>


```python
quartet = pd.read_csv('anscombes_quartet1.csv')
```


```python
# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']
quartet['residual'] = quartet['y'] - quartet['pred_y']

sns.scatterplot(data=quartet,x='x',y='y')
sns.lineplot(data=quartet,x='x',y='pred_y',color='red')
plt.vlines(quartet['x'],quartet['y'],quartet['y']-quartet['residual'])
```




    <matplotlib.collections.LineCollection at 0x21603321888>




    
![png](output_44_1.png)
    



```python
sns.kdeplot(quartet['residual'])
```




    <AxesSubplot:xlabel='residual', ylabel='Density'>




    
![png](output_45_1.png)
    



```python
sns.scatterplot(data=quartet,x='y',y='residual')
plt.axhline(y=0, color='r', linestyle='--')
```




    <matplotlib.lines.Line2D at 0x216032d8a08>




    
![png](output_46_1.png)
    


---


```python
quartet = pd.read_csv('anscombes_quartet2.csv')
```


```python
quartet.columns = ['x','y']
```


```python
# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']
quartet['residual'] = quartet['y'] - quartet['pred_y']

sns.scatterplot(data=quartet,x='x',y='y')
sns.lineplot(data=quartet,x='x',y='pred_y',color='red')
plt.vlines(quartet['x'],quartet['y'],quartet['y']-quartet['residual'])
```




    <matplotlib.collections.LineCollection at 0x21603475dc8>




    
![png](output_50_1.png)
    



```python
sns.kdeplot(quartet['residual'])
```




    <AxesSubplot:xlabel='residual', ylabel='Density'>




    
![png](output_51_1.png)
    



```python
sns.scatterplot(data=quartet,x='y',y='residual')
plt.axhline(y=0, color='r', linestyle='--')
```




    <matplotlib.lines.Line2D at 0x21603410fc8>




    
![png](output_52_1.png)
    



```python
quartet = pd.read_csv('anscombes_quartet4.csv')
```


```python
quartet
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
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8.0</td>
      <td>6.58</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>5.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.0</td>
      <td>7.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.0</td>
      <td>8.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.0</td>
      <td>8.47</td>
    </tr>
    <tr>
      <th>5</th>
      <td>8.0</td>
      <td>7.04</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>5.25</td>
    </tr>
    <tr>
      <th>7</th>
      <td>19.0</td>
      <td>12.50</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>5.56</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8.0</td>
      <td>7.91</td>
    </tr>
    <tr>
      <th>10</th>
      <td>8.0</td>
      <td>6.89</td>
    </tr>
  </tbody>
</table>
</div>




```python
# y = 3.00 + 0.500x
quartet['pred_y'] = 3 + 0.5 * quartet['x']
```


```python
quartet['residual'] = quartet['y'] - quartet['pred_y']
```


```python
sns.scatterplot(data=quartet,x='x',y='y')
sns.lineplot(data=quartet,x='x',y='pred_y',color='red')
plt.vlines(quartet['x'],quartet['y'],quartet['y']-quartet['residual'])
```




    <matplotlib.collections.LineCollection at 0x216035bf808>




    
![png](output_57_1.png)
    



```python
sns.kdeplot(quartet['residual'])
```




    <AxesSubplot:xlabel='residual', ylabel='Density'>




    
![png](output_58_1.png)
    



```python
sns.scatterplot(data=quartet,x='y',y='residual')
plt.axhline(y=0, color='r', linestyle='--')
```




    <matplotlib.lines.Line2D at 0x21603641688>




    
![png](output_59_1.png)
    


### Plotting Residuals

It's also important to plot out residuals and check for normal distribution, this helps us understand if Linear Regression was a valid model choice.


```python
# Predictions on training and testing sets
# Doing residuals separately will alert us to any issue with the split call
test_predictions = model.predict(X_test)
```


```python
# If our model was perfect, these would all be zeros
test_res = y_test - test_predictions
```


```python
sns.scatterplot(x=y_test,y=test_res)
plt.axhline(y=0, color='r', linestyle='--')
```




    <matplotlib.lines.Line2D at 0x216036b5308>




    
![png](output_63_1.png)
    



```python
len(test_res)
```




    60




```python
sns.displot(test_res,bins=25,kde=True)
```




    <seaborn.axisgrid.FacetGrid at 0x2160370e708>




    
![png](output_65_1.png)
    


Still unsure if normality is a reasonable approximation? We can check against the [normal probability plot.](https://en.wikipedia.org/wiki/Normal_probability_plot)


```python
import scipy as sp
```


```python
# Create a figure and axis to plot on
fig, ax = plt.subplots(figsize=(6,8),dpi=100)
# probplot returns the raw values if needed
# we just want to see the plot, so we assign these values to _
_ = sp.stats.probplot(test_res,plot=ax)
```


    
![png](output_68_0.png)
    


-----------

## Retraining Model on Full Data

If we're satisfied with the performance on the test data, before deploying our model to the real world, we should retrain on all our data. (If we were not satisfied, we could update parameters or choose another model, something we'll discuss later on).


```python
final_model = LinearRegression()
```


```python
final_model.fit(X,y)
```




    LinearRegression()



Note how it may not really make sense to recalulate RMSE metrics here, since the model has already seen all the data, its not a fair judgement of performance to calculate RMSE on data its already seen, thus the purpose of the previous examination of test performance.

## Deployment, Predictions, and Model Attributes

### Final Model Fit

Note, we can only do this since we only have 3 features, for any more it becomes unreasonable.


```python
y_hat = final_model.predict(X)
```


```python
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))

axes[0].plot(df['TV'],df['sales'],'o')
axes[0].plot(df['TV'],y_hat,'o',color='red')
axes[0].set_ylabel("Sales")
axes[0].set_title("TV Spend")

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],y_hat,'o',color='red')
axes[1].set_title("Radio Spend")
axes[1].set_ylabel("Sales")

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['radio'],y_hat,'o',color='red')
axes[2].set_title("Newspaper Spend");
axes[2].set_ylabel("Sales")
plt.tight_layout();
```


    
![png](output_76_0.png)
    


### Residuals

Should be normally distributed as discussed in the video.


```python
residuals = y_hat - y
```


```python
sns.scatterplot(x=y,y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
```




    <matplotlib.lines.Line2D at 0x216039d7ac8>




    
![png](output_79_1.png)
    


### Coefficients


```python
final_model.coef_
```




    array([ 0.04576465,  0.18853002, -0.00103749])




```python
coeff_df = pd.DataFrame(final_model.coef_,X.columns,columns=['Coefficient'])
coeff_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Coefficient</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TV</th>
      <td>0.045765</td>
    </tr>
    <tr>
      <th>radio</th>
      <td>0.188530</td>
    </tr>
    <tr>
      <th>newspaper</th>
      <td>-0.001037</td>
    </tr>
  </tbody>
</table>
</div>



Interpreting the coefficients:

---
* Holding all other features fixed, a 1 unit (A thousand dollars) increase in TV Spend is associated with an increase in sales of  0.045 "sales units", in this case 1000s of units . 
* This basically means that for every $1000 dollars spend on TV Ads, we could expect 45 more units sold.
----

---
---
* Holding all other features fixed, a 1 unit (A thousand dollars) increase in Radio Spend is associated with an increase in sales of  0.188 "sales units", in this case 1000s of units . 
* This basically means that for every $1000 dollars spend on Radio Ads, we could expect 188 more units sold.
----
----

* Holding all other features fixed, a 1 unit (A thousand dollars) increase in Newspaper Spend is associated with a **decrease** in sales of  0.001 "sales units", in this case 1000s of units . 
* This basically means that for every $1000 dollars spend on Newspaper Ads, we could actually expect to sell 1 less unit. Being so close to 0, this heavily implies that newspaper spend has no real effect on sales.
---
---

**Note! In this case all our units were the same for each feature (1 unit = $1000 of ad spend). But in other datasets, units may not be the same, such as a housing dataset could try to predict a sale price with both a feature for number of bedrooms and a feature of total area like square footage. In this case it would make more sense to *normalize* the data, in order to clearly compare features and results. We will cover normalization later on.**


```python
df.corr()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>radio</th>
      <th>newspaper</th>
      <th>sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>TV</th>
      <td>1.000000</td>
      <td>0.054809</td>
      <td>0.056648</td>
      <td>0.782224</td>
    </tr>
    <tr>
      <th>radio</th>
      <td>0.054809</td>
      <td>1.000000</td>
      <td>0.354104</td>
      <td>0.576223</td>
    </tr>
    <tr>
      <th>newspaper</th>
      <td>0.056648</td>
      <td>0.354104</td>
      <td>1.000000</td>
      <td>0.228299</td>
    </tr>
    <tr>
      <th>sales</th>
      <td>0.782224</td>
      <td>0.576223</td>
      <td>0.228299</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Prediction on New Data

Recall , X_test data set looks *exactly* the same as brand new data, so we simply need to call .predict() just as before to predict sales for a new advertising campaign.

**Our next ad campaign will have a total spend of 149k on TV, 22k on Radio, and 12k on Newspaper Ads, how many units could we expect to sell as a result of this?**


```python
campaign = [[149,22,12]]
```


```python
final_model.predict(campaign)
```




    array([13.893032])



**How accurate is this prediction? No real way to know! We only know truly know our model's performance on the test data, that is why we had to be satisfied by it first, before training our full model**

-----

## Model Persistence (Saving and Loading a Model)


```python
from joblib import dump, load
```


```python
dump(final_model, 'sales_model.joblib') 
```




    ['sales_model.joblib']




```python
loaded_model = load('sales_model.joblib')
```


```python
loaded_model.predict(campaign)
```




    array([13.893032])



## Up next...
### Is this the best possible performance? Its a simple model still, let's expand on the linear regresion model by taking a further look a regularization!

-------
--------
