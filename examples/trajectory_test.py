
# coding: utf-8

# # Trajectory Test

# In[1]:


get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy.ma as ma
import numpy as np
import netCDF4
import gridded


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
len(x)


# In[5]:


plt.pcolormesh(lon,lat,ma.masked_invalid(temp),vmin=5,vmax=15,cmap='jet');
plt.plot(x,y,'-')
plt.grid()
plt.colorbar();


# In[6]:


temperature = gridded.Variable.from_netCDF(filename=url, name='Temperature', varname='temp')
salinity = gridded.Variable.from_netCDF(filename=url, name='Salinity', varname='salt', grid=temperature.grid)
points = np.column_stack((x,y))

t0 = temperature.time.max_time


# ## Interpolate values at array of lon,lat points at specific time

# In[7]:


salts = salinity.at(points, t0)


# In[8]:


temps = temperature.at(points, t0)


# In[9]:


plt.plot(temps)


# In[10]:


times = temperature.time.data


# ## Interpolate values at lon,lat points with changing time values

# In[11]:


over_time = [temperature.at((x[i],y[i]), val)[0] for i,val in enumerate(times)]


# In[12]:


plt.plot(over_time)

