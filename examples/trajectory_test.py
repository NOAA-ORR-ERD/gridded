
# coding: utf-8

# # Trajectory Test

# In[1]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import netCDF4


# In[2]:


url = 'http://geoport.whoi.edu/thredds/dodsC/examples/bora_feb.nc'


# In[3]:


nc = netCDF4.Dataset(url)
lon = nc['lon_rho'][:]
lat = nc['lat_rho'][:]
temp = nc['temp'][-1,-1,:,:]


# In[4]:


x = np.linspace(13., 15.)
y = np.linspace(45.3, 43.5)


# In[5]:


plt.pcolormesh(lon,lat,ma.masked_invalid(temp),vmin=5,vmax=15,cmap='jet');
plt.plot(x,y,'-')
plt.grid()
plt.colorbar();


# In[ ]:





# In[ ]:




