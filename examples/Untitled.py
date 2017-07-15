
# coding: utf-8

# # Trajectory Test

# In[29]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import netCDF4


# In[30]:


url = 'http://geoport.whoi.edu/thredds/dodsC/examples/bora_feb.nc'


# In[31]:


nc = netCDF4.Dataset(url)
lon = nc['lon_rho'][:]
lat = nc['lat_rho'][:]
temp = nc['temp'][-1,-1,:,:]


# In[32]:


plt.pcolormesh(lon,lat,ma.masked_invalid(temp),vmin=5,vmax=15,cmap='jet');
plt.grid()
plt.colorbar();


# In[33]:


xe=[13.0, 15.0]
ye=[45.0, 43.5]


# In[35]:


x = np.linspace(13., 15.)
y = np.linspace(45.3, 43.5)


# In[39]:





# In[ ]:




