import numpy as np
import netCDF4
import datetime
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Open the NetCDF file
ncfile = netCDF4.Dataset('CRU PRECIP DATA', 'r')

# Read in the relevant variables
lon_var = ncfile.variables['lon']
lat_var = ncfile.variables['lat']
time_var = ncfile.variables['time']
precip_var = ncfile.variables['pre']

# Define the bounding box for CONUS
lon_min, lon_max = -124.736342, -66.945392
lat_min, lat_max = 24.521208, 49.382808

# Extract the indices for the bounding box
lon_indices = np.where((lon_var[:] >= lon_min) & (lon_var[:] <= lon_max))[0]
lat_indices = np.where((lat_var[:] >= lat_min) & (lat_var[:] <= lat_max))[0]

# Extract the time period of interest
start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2020, 12, 31)

time_in_days = time_var[:]
start_index = np.argmin(np.abs(netCDF4.num2date(time_in_days, time_var.units) - start_date))
end_index = np.argmin(np.abs(netCDF4.num2date(time_in_days, time_var.units) - end_date))
time_subset = time_var[start_index:end_index+1]
precip_data_subset = precip_var[start_index:end_index+1, lat_indices, lon_indices]

# Compute the trend of the precipitation data
def trend(data):
    x = np.arange(len(data))
    A = np.vstack([x, np.ones(len(data))]).T
    m, c = np.linalg.lstsq(A, data, rcond=None)[0]
    return m

trend_vals = np.apply_along_axis(trend, axis=0, arr=precip_data_subset)

# Mask out values greater than 20
trend_vals = np.where(trend_vals < -.9, np.nan, trend_vals)

# Calculate the average weighted area
lon_weights = np.cos(np.radians(lat_var[lat_indices]))
area_weights = np.outer(lon_weights, np.ones(len(lon_indices)))
avg_area = np.average(trend_vals, weights=area_weights)

# Create a map with the trend values plotted
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Set the extent of the map
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Add the coastline and state borders
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.STATES)

# Add the trend values to the plot
lons = lon_var[lon_indices]
lats = lat_var[lat_indices]
trend_data = trend_vals.reshape(len(lat_indices), len(lon_indices))
cmap = plt.get_cmap('BrBG')
contour = ax.pcolormesh(lons, lats, trend_data,vmin = -0.07,vmax = 0.07, cmap=cmap,
                      transform=ccrs.PlateCarree())

# Add a colorbar
cbar = plt.colorbar(contour, ax=ax)
cbar.ax.set_ylabel('Trend (mm/year)')

# Add a title and show the plot
plt.title('Trend in Precipitation (1990-2020)')
plt.show()

