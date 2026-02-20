"""
The error happens because `mascon_id` is added to the dataset *after* the 
`ds_merged.rio.write_crs()` command is called. Xarray does not automatically 
propagate CRS metadata to newly created variables appended to an existing dataset.

The fix is simple: Move the CRS assignment to happen *after* all variables, 
including `mascon_id`, have been added to the dataset. I have also added an 
explicit CRS tag to the `vars_nearest` subset right before reprojection to be 
absolutely bulletproof.
"""

import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
import dask.array as da
import zarr
import os
from datetime import timedelta
from rasterio.enums import Resampling
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------
DATA_DIR = "data/raw/grace"
MASCON_FILE = os.path.join(DATA_DIR, "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc")
LAND_MASK_FILE = os.path.join(DATA_DIR, "CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc")
OCEAN_MASK_FILE = os.path.join(DATA_DIR, "CSR_GRACE_GRACE-FO_RL06_Mascons_v02_OceanMask.nc")

OUTPUT_ZARR = "data/processed/grace.zarr"

# TARGET CRS: Antarctic Polar Stereographic (EPSG:3031)
TARGET_CRS = "EPSG:3031"
TARGET_RES = 27750  # ~0.25 degrees in meters
MAX_LATITUDE = -50.0 

# Time Filter
START_DATE = "2002-04-01"
END_DATE = "2025-12-31"

def clean_coordinates(ds):
    """
    Standardizes longitude from 0..360 to -180..180 and sorts.
    """
    if 'lon' in ds.coords:
        if ds.lon.max() > 180:
            print("  -> Wrapping longitude (0-360 to -180-180)...")
            ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
            ds = ds.sortby(['lon', 'lat'])
    return ds

def generate_mascon_id_lazy(ds):
    """Generates unique IDs using Dask (Lazy)."""
    lat = ds.coords['lat']
    lon = ds.coords['lon']
    total_pixels = len(lat) * len(lon)
    
    linear_id = da.arange(total_pixels, chunks='auto')
    mascon_id_grid = linear_id.reshape((len(lat), len(lon)))

    return xr.DataArray(
        mascon_id_grid,
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon'),
        name='mascon_id'
    )

if __name__ == "__main__":
    print(f"Starting Pipeline | Method: Split-Domain Bilinear | Target: {TARGET_RES}m")

    # --------------------------------------------------------------------------------
    # 2. LAZY LOADING & TIME DECODING
    # --------------------------------------------------------------------------------
    print(f"Loading Mascons: {MASCON_FILE}")
    
    # Use chunks={} to respect native disk chunks on read and silence the warning.
    # Then explicitly rechunk for our Dask graph execution.
    ds_raw = xr.open_dataset(MASCON_FILE, decode_times=False, chunks={})
    ds_raw = ds_raw.chunk({'time': 1, 'lat': -1, 'lon': -1})

    print("  -> Decoding time manually...")
    raw_times = ds_raw['time'].compute().values
    ref_date = pd.Timestamp("2002-01-01")
    time_decoded = [ref_date + timedelta(days=float(t)) for t in raw_times]
    
    ds_raw = ds_raw.assign_coords(time=pd.to_datetime(time_decoded))
    ds_raw = xr.decode_cf(ds_raw, decode_times=False)

    if 'lwe_thickness' in ds_raw:
        ds_raw = ds_raw.rename({'lwe_thickness': 'lwe_length'})

    ds_main = ds_raw.sel(time=slice(START_DATE, END_DATE))

    # --------------------------------------------------------------------------------
    # 3. MERGE MASKS & ALIGN
    # --------------------------------------------------------------------------------
    print("Loading Masks...")
    ds_land = xr.open_dataset(LAND_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'land_mask'})
    ds_ocean = xr.open_dataset(OCEAN_MASK_FILE, chunks={})[['LO_val']].rename({'LO_val': 'ocean_mask'})

    ds_merged = xr.merge([ds_main, ds_land, ds_ocean], compat='override')

    print("Cleaning coordinates...")
    ds_merged = clean_coordinates(ds_merged)
    
    print(f"Cropping data to latitudes south of {MAX_LATITUDE}°...")
    ds_merged = ds_merged.where(ds_merged.lat <= MAX_LATITUDE, drop=True)
    
    print("Generating IDs...")
    ds_merged['mascon_id'] = generate_mascon_id_lazy(ds_merged)

    # FIX: Explicitly write CRS to the merged dataset *after* adding the mascon_id
    ds_merged.rio.write_crs("EPSG:4326", inplace=True)

    # --------------------------------------------------------------------------------
    # 4. SPLIT-DOMAIN REPROJECTION
    # --------------------------------------------------------------------------------
    print(f"Reprojecting to {TARGET_CRS}...")
    
    # A. SEPARATE LAND AND OCEAN SIGNALS
    print("  -> Splitting Land/Ocean signals...")
    lwe_land_src = ds_merged['lwe_length'].where(ds_merged['land_mask'] == 1)
    lwe_ocean_src = ds_merged['lwe_length'].where(ds_merged['ocean_mask'] == 1)

    # .where() drops CRS on individual DataArrays, re-attach here:
    lwe_land_src.rio.write_crs("EPSG:4326", inplace=True)
    lwe_ocean_src.rio.write_crs("EPSG:4326", inplace=True)

    # B. REPROJECT CONTINUOUS DATA SEPARATELY (Bilinear)
    print("  -> Reprojecting Land signal (Bilinear)...")
    lwe_land_reproj = lwe_land_src.rio.reproject(
        TARGET_CRS,
        resolution=TARGET_RES,
        resampling=Resampling.bilinear,
        nodata=np.nan
    )

    print("  -> Reprojecting Ocean signal (Bilinear)...")
    lwe_ocean_reproj = lwe_ocean_src.rio.reproject(
        TARGET_CRS,
        resolution=TARGET_RES,
        resampling=Resampling.bilinear,
        nodata=np.nan
    )

    # C. RECOMBINE
    print("  -> Recombining Land/Ocean...")
    lwe_combined = lwe_land_reproj.combine_first(lwe_ocean_reproj)
    lwe_combined.name = 'lwe_length'

    # D. REPROJECT DISCRETE VARIABLES (Nearest Neighbor)
    print("  -> Reprojecting Masks/IDs (Nearest)...")
    vars_nearest = ['land_mask', 'ocean_mask', 'mascon_id']
    
    # FIX: Extra safety check to ensure subset has CRS before reprojecting
    ds_discrete = ds_merged[vars_nearest]
    ds_discrete.rio.write_crs("EPSG:4326", inplace=True)

    reproj_nearest = ds_discrete.rio.reproject(
        TARGET_CRS,
        resolution=TARGET_RES,
        resampling=Resampling.nearest,
        nodata=0
    )

    # E. FINAL MERGE
    lwe_combined = lwe_combined.reindex_like(reproj_nearest, method=None, tolerance=1e-5)
    ds_final = xr.merge([lwe_combined, reproj_nearest])

    # --------------------------------------------------------------------------------
    # 5. WRITE TO ZARR
    # --------------------------------------------------------------------------------
    print(f"Writing to Zarr: {OUTPUT_ZARR}")

    zarr_chunks = {'time': 1, 'x': -1, 'y': -1}
    ds_final = ds_final.chunk(zarr_chunks)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
    encoding = {var: {'compressor': compressor} for var in ds_final.data_vars}

    ds_final.to_zarr(
        OUTPUT_ZARR, 
        mode='w', 
        encoding=encoding, 
        consolidated=False, 
        compute=True
    )
    
    print("Consolidating metadata...")
    zarr.consolidate_metadata(OUTPUT_ZARR)
    
    print("Complete.")
