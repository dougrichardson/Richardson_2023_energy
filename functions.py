import xarray as xr

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