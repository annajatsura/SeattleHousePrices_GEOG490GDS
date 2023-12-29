#!/usr/bin/env python
# coding: utf-8

# # Assignment 5
# ## Anna Jatsura 
# ### 16 February 2023

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


cd downloads


# In[9]:


pd.read_csv('data-5/seattle-house-prices.csv')


# In[10]:


seattle = pd.read_csv('data-5/seattle-house-prices.csv')


# In[11]:


seattle.info()


# In[12]:


seattle.describe()


# In[15]:


print('There are eight features in the dataset that can predict house price')


# In[17]:


print('Null values dont seem to appear in this dataset')


# In[19]:


print('Three variables that are best correlated to house price may be housing median age, meidan income, and proximity to good schools or employemnt opportunities or shopping.')


# In[22]:


print('Three variables that are least correlated to house price are how many squirrels live in an area, type of materials used to build a home, and type of shower in bathrooms.')


# In[23]:


corr_matrix = seattle.corr()


# In[24]:


corr_matrix["price"].sort_values(ascending= False)


# In[26]:


print('The most correlated featrures to house price are sqft_living, bathrooms, and bedrooms. The least correleated features are sqft_lot, the year it was built, and longitude')


# In[27]:


print('Task 2')


# In[49]:


print('Random Forest Regressor')


# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


import sklearn


# In[48]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[19]:


cd downloads


# In[20]:


pd.read_csv('data-5/seattle-house-prices.csv')


# In[21]:


seattle = pd.read_csv('data-5/seattle-house-prices.csv')


# In[22]:


feature_list =  ['price', 'bedrooms', 'bathrooms', 
                 'sqft_living', 'sqft_lot', 'yr_built', 'lat', 'long']


# In[24]:


X = seattle[feature_list]
y = seattle['price']


# In[25]:


scaler = StandardScaler()  


# In[26]:


X_scaled = scaler.fit_transform(X)


# In[36]:


pd.DataFrame(X_scaled, columns=feature_list)


# In[37]:


seattle_scaled = pd.DataFrame(X_scaled, columns=feature_list)


# In[38]:


seattle_scaled


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(seattle_scaled, y, test_size=0.2, random_state=42)


# In[41]:


from sklearn.ensemble import RandomForestRegressor


# In[42]:


forest_reg = RandomForestRegressor(n_estimators = 30)


# In[43]:


forest_reg.fit(X_train, y_train)


# In[44]:


predictions = forest_reg.predict(X_test)


# In[45]:


final_mse = mean_squared_error(y_test , predictions)


# In[46]:


final_rmse = np.sqrt(final_mse)


# In[47]:


final_rmse


# In[50]:


print('RandomForests predicted that house prices in Seattle are about $33,035.')


# In[51]:


print('House prices and geolocation data')


# In[1]:


import geopandas as gpd


# In[3]:


import pandas as pd


# In[5]:


cd downloads


# In[6]:


seattle = pd.read_csv('data-5/seattle-house-prices.csv')
waterbodies = gpd.read_file('data-5/waterbodies.shp')


# In[8]:


gdf = gpd.GeoDataFrame(seattle, geometry=gpd.points_from_xy(seattle['long'], seattle['lat']))
gdf = gdf.set_crs(4326, allow_override=True)


# In[9]:


gdf_utm = gdf.to_crs('EPSG:32610')
waterbodies_utm = waterbodies.to_crs('EPSG:32610')


# In[16]:


distance_to_waterbodies = []
for i in range(gdf_utm.shape[0]):
    distance_to_waterbodies.append(waterbodies_utm.distance(gdf_utm['geometry'].iloc[i]).min())


# In[17]:


gdf_utm['distance_to_waterbodies'] = distance_to_waterbodies


# In[20]:


import matplotlib.pyplot as plt


# In[22]:


fig, ax = plt.subplots(figsize=(8, 6)) 
ax.scatter(gdf_utm['geometry'].x, gdf_utm['geometry'].y, c=gdf_utm['distance_to_waterbodies'])


# In[24]:


print('This plot shows different house prices in Seattle. The yellows are concentrated near the upper central left while different shades of blue and blue-green are seen pretty much everywhere else. The relationship between house price and distance to waterbodies seems like the closer a house is to a body of water, the pricier it may be. The yellow area is most likely near Elliot Bay and is the best location that shows how house price may be higher near waterbodies.  ')


# In[26]:


corr_matrix = gdf_utm.corr()


# In[27]:


corr_matrix["price"].sort_values(ascending= False)

