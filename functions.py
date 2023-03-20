import xarray as xr
# import numpy as np
# import dask
import math
import string
from xhistogram.xarray import histogram

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import colorsys

import cartopy
import cartopy.crs as ccrs

# ============================================================================
# Masks
# ============================================================================

def get_REZ_boundary():
    """
    Returns a list of bounding coordinates for the REZ
    regions (effectively eastern Australia).
    """
    return [133, 155, -10, -45]

def get_rez_mask():
    """
    Open Renewable Energy Zones (REZ) mask on the ERA5 grid.
    """
    return xr.open_dataset('/g/data/w42/dr6273/work/projects/Aus_energy/data/rez_mask_era5_grid.nc').REZ

def get_gccsa_mask():
    """
    Open GCCSA mask on the ERA5 grid.
    """
    return xr.open_dataset( '/g/data/w42/dr6273/work/projects/Aus_energy/data/gccsa_mask_era5_grid.nc').GCCSA

def get_regions_from_region_codes(region_codes, mask):
    """
    Return the region IDs (N1, N2, etc.) given the region codes used to identify
    the renewables type in each REZ.
    
    region_codes : list of integers in [1, 8]
    mask : xarray DataArray of the desired mask.
    """
    regions = [mask.region.values[i] for i in range(len(mask.region.values)) if
               mask.region_code.values[i] in region_codes]
    return regions

# ============================================================================
# Data wrangling
# ============================================================================

def open_era_data(root_path,
                  variable,
                  years,
                  concat_dim='time',
                  subset_region=None,
                  lat_name='latitude',
                  lon_name='longitude',
                  rename_lon_lat=None,
                  subset_level=None,
                  level_name='level',
                  preprocess_func=None,
                  mfdataset_chunks=None):
    """
    Open ERA5 data from NCI.
    
    root_path : path to era5 data
    variable : short name of variable used in path
    years : range(first_year, last_year)
    concat_dim : dimension name to concat over
    subset_region : None, or select subregion using coordinates in a
                    list like:[lon1, lon2, lat1, lat2]
    lat_name : latitude dimension name
    lon_name : longitude dimension name
    rename_lon_lat : None, or list of desired lon/lat name
    subset_level : None, or select levels in a list
    level_name : name of level dimension
    preprocess_func : bool. None, or a function passed to open_mfdataset
    mfdataset_chunks : None, or how open_mfdataset should chunk
    """
    ds_list = []
    for year in years:
        fp = root_path+variable+'/'+str(year)+'/*.nc'
        if preprocess_func is None:
            ds = xr.open_mfdataset(fp)
        else:
            ds = xr.open_mfdataset(fp, chunks=mfdataset_chunks, preprocess=preprocess_func)
        
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
            
        if isinstance(subset_level, list):
            ds = ds.sel({
                level_name: subset_level
            })
            
        ds_list.append(ds)
    return xr.concat(ds_list, dim=concat_dim)

def detrend_dim(da, dim, deg=1):
    """
    Detrend along a single dimension.
    
    Author: Ryan Abernathy
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def month_subset(da, months, time_name='time'):
    """
    Subset the dataArray by month.
    """
    return da.isel({time_name: da.time.dt.month.isin(months)})

# def switch_lons(ds, lon_name='lon'):
#     """
#     Switches lons from -180-180 to 0-360 or vice versa
#     """
#     ds = ds.copy()
#     with dask.config.set(**{'array.slicing.split_large_chunks': True}):
#         if np.any(ds.coords[lon_name] < 0): # if current coords are -180 to 180
#             ds.coords[lon_name] = (ds.coords[lon_name] + 360) % 360
#         else:
#             ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
#         return ds.sortby(ds[lon_name])

# ============================================================================
# Defining and computing events
# ============================================================================

def get_events(da, thresh, tail='lower'):
    """
    Return binary DataArray of univariate events. 1 indicates an event, 0 no events.
    Events are defined if they are below (tail='lower') or
    above (tail='upper') a threshold.
    """
    if tail == 'lower':
        return xr.where(da < thresh, 1, 0)
    elif tail == 'upper':
        return xr.where(da > thresh, 1, 0)
    else:
        raise ValueError('Incorrect string for tail')
        
def get_compound_events(da1, da2, thresh1, thresh2, tail='lower'):
    """
    Return binary DataArray of compound events. 1 indicates an event, 0 no events.
    Events are defined if they are both below (tail='lower') or
    above (tail='upper') a threshold.
    
    Currently only lower tail thresholds for both variables is implemented.
    """
    if tail == 'lower':
        events = xr.where(
            (da1 < thresh1) &
            (da2 < thresh2),
            1, 0
        )
    else:
        raise ValueError("Incorrect tail")
        
    events = events.to_dataset(name='compound')
    
    return events['compound']
        
def calculate_event_frequency(da, thresh, tail, time_name='time'):
    """
    Relative frequency of events over a dimension (usually time).
    """
    T = len(da[time_name].values)
    events = get_events(da, thresh, tail)
    freq = events.sum(time_name) / T
            
    return freq

def calculate_compound_frequency(da1, da2, thresh1, thresh2, tail='lower', time_name='time'):
    """
    Relative frequency of compound events over a dimension (usually time).
    """
    events = get_compound_events(da1, da2, thresh1, thresh2, tail=tail)
    return events.sum(time_name) / len(da1[time_name].values)

def get_event_years(da, thresh, tail='lower', time_name='time'):
    """
    Get the event years and other years of da.
    """
    if tail == 'lower':
        event_years = da.where(
            da < thresh
        ).dropna(time_name)[time_name].dt.year.values

        other_years = da.where(
            da >= thresh
        ).dropna(time_name)[time_name].dt.year.values
    elif tail == 'upper':
        event_years = da.where(
            da > thresh
        ).dropna(time_name)[time_name].dt.year.values

        other_years = da.where(
            da <= thresh
        ).dropna(time_name)[time_name].dt.year.values
    
    return event_years, other_years

def concurrent_lulls(da, region_codes, mask):
    """
    Total number of univariate events across desired subset of regions.
    """
    return da.sel(
        region=get_regions_from_region_codes(region_codes, mask)
    ).sum('region')

def concurrent_univariate_or_compound_lulls(compound_da, univ_da, region_codes, mask, var_name='event'):
    """
    Total number of events (univariate or compound) across desired subset of regions.
    """
    events = xr.where(
        compound_da == 0,
        univ_da, 0
    ).sel(
        region=get_regions_from_region_codes(region_codes, mask)
    ).sum('region')
    events = events.to_dataset(name=var_name)
    return events[var_name]

def get_all_events(da1, da2, count_compound_as_double=True, var_name='event'):
    """
    Total number of events for two variables across all regions. Can count
    compound events as double or single.
    """
    events = da1 + da2
    
    if count_compound_as_double == False: # Set compound days (coded as a 2) to 1.
        events = xr.where(events == 2, 1, events)
        
    events = events.sum('region')
    
    return events.to_dataset(name=var_name)[var_name]

def n_simultaneous_droughts(da, thresh, region_codes, mask):
    """
    Number of concurrent droughts across desired subset of regions.
    """
    da = da.sel(region=get_regions_from_region_codes(region_codes, mask))
    return xr.where(da < thresh, 1, 0).sum('region')

def seasonal_mean(da, time_name='time'):
    """
    Calculate seasonal means. Currently assumes da starts in January.
    First aggregates to monthly means, then shifts everything forwards
    one month, then aggregates over 3 months.
    """
    first_month = da[time_name].dt.month.values[0]
    if first_month != 1:
        raise ValueError("First month should be January. Adjust da or function.")
        
    # First aggregate to monthly as we want to shift a month ahead
    m_da = da.resample({time_name: '1MS'}).mean()
    # Now shift a month ahead and calculate 3-monthly means
    return m_da.shift({time_name: 1}).resample({time_name: '3MS'}).mean(skipna=False)

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
            'axes.titlesize': FONT_SIZE + 1}

def adjust_lightness(color, amount=0.5):
    """
    Adjust the lightness of a specified colour.
    """
    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

# ============================================================================
# Misc
# ============================================================================

def get_seasons():
    """
    Return month numbers of each season.
    """
    return {
        'Annual': range(1, 13),
        'Winter': [6, 7, 8],
        'Spring': [9, 10, 11],
        'Summer': [12, 1, 2],
        'Autumn': [3, 4, 5]
    }

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

def hist_data(da, bins):
    """
    Compute the histogram.
    """
    h = histogram(da, bins=[bins], density=True)
    h = h.rename({list(h.coords)[0]: 'bin'})
    return h['bin'], h