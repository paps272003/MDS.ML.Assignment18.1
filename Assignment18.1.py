
# coding: utf-8

# In[10]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
boston = datasets.load_boston()
features = pd.DataFrame(boston.data, columns=boston.feature_names)
targets = boston.target


# In[3]:

features.head()


# In[5]:

targets


# In[ ]:

boston


# In[6]:

model = RandomForestRegressor()


# In[ ]:

model


# In[7]:

model.fit(boston.data, boston.target)


# In[8]:

predicted = model.predict(boston.data)


# In[12]:

print ("Random Forest model \n Boston dataset")
print ("Mean squared error = %0.3f" % mse(targets, predicted))
print("R2 score = %0.3f" % r2_score(targets, predicted)) 


# In[ ]:



