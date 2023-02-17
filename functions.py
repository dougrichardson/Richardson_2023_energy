import xarray as xr

import math

import string

import matplotlib
import matplotlib.pyplot as plt

import cartopy
import cartopy.crs as ccrs

# ============================================================================
# Misc
# ============================================================================

def round_decimals_up(number:float, decimals:int=2):
    """
    Returns a value rounded up to a specific number of decimal places.
    
    From: https://kodify.net/python/math/round-decimals/
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more")
    elif decimals == 0:
        return math.ceil(number)

    factor = 10 ** decimals
    return math.ceil(number * factor) / factor

# ============================================================================
# Data wrangling
# ============================================================================

def get_REZ_boundary():
    """
    Returns a list of bounding coordinates for the REZ
    regions (effectively eastern Australia).
    """
    return [133, 155, -10, -45]

def open_era_data(root_path,
                  variable,
                  years,
                  concat_dim='time',
                  subset_region=None,
                  lat_name='latitude',
                  lon_name='longitude',
                  rename_lon_lat=None):
    """
    Open ERA5 data from NCI.
    
    root_path: path to era5 data
    variable: short name of variable used in path
    years: range(first_year, last_year)
    concat_dim: dimension name to concat over
    subset_region: None, or select subregion using coordinates in a
                    list like:[lon1, lon2, lat1, lat2]
    lat_name: latitude dimension name
    lon_name: longitude dimension name
    rename_lon_lat: None, or list of desired lon/lat name
    """
    ds_list = []
    for year in years:
        fp = root_path+variable+'/'+str(year)+'/*.nc'
        ds = xr.open_mfdataset(fp)
        
        if isinstance(subset_region, list):
            ds = ds.sel({
                lon_name: slice(subset_region[0], subset_region[1]),
                lat_name: slice(subset_region[2], subset_region[3])
            })
            
        if isinstance(rename_lon_lat, list):
            ds = ds.rename({
                lon_name: rename_lon_lat[0],
                lat_name: rename_lon_lat[1],
            })
            
        ds_list.append(ds)
    return xr.concat(ds_list, dim=concat_dim)

# ============================================================================
# Plotting
# ============================================================================

letters = list(string.ascii_lowercase)

def get_plot_params():
    """
    Get the plotting parameters used for figures
    """
    FONT_SIZE = 7
    COASTLINES_LW = 0.5
    LINEWIDTH = 1.3
    PATHEFFECT_LW_ADD = LINEWIDTH * 1.8

    return {'lines.linewidth': LINEWIDTH,
            'hatch.linewidth': 0.5,
            'font.size': FONT_SIZE,
            'legend.fontsize' : FONT_SIZE - 1,
            'legend.columnspacing': 0.7,
            'legend.labelspacing' : 0.03,
            'legend.handlelength' : 1.,
            'axes.linewidth': 0.5,
            'axes.titlesize': 10}