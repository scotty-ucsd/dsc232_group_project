"""
step_03_process_bedmap3.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-12
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION (Optimized)
LOGIC:  Regrids Bedmap3 to 500m Master Grid.
        HOTFIX: Uses Mascon Map directly as coordinate master.
        HOTFIX: Injects mascon_id for downstream regional aggregation.
        HOTFIX: Implemented Zarr Blosc Zstd compression.
        HOTFIX: Forced Zarr Format 2 to prevent V3 metadata compatibility issues.
-------------------------------------------------------------------------------
"""

import os
import warnings
import numpy as np
import xarray as xr
import dask.config
import zarr
from dask.distributed import Client, LocalCluster

# --- MANDATORY PREAMBLE ---
warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)

# --- CONFIGURATION ---
DASK_CONFIG = {
    "n_workers": 4, 
    "threads_per_worker": 2, 
    "memory_limit": "8GB"
}

# EXACT PATHS
INPUT_FILE = "data/raw/bedmap/bedmap3.nc"
MASCON_MAP = "data/processed/mascon_id_master_map.zarr"
OUTPUT_DIR = "data/processed"

# MAPPING: Bedmap3 Name -> Pipeline Name
TASKS = {
    'bed_topography':     {'out': 'bed',       'method': 'linear',  'dtype': np.float32},
    'surface_topography': {'out': 'surface',   'method': 'linear',  'dtype': np.float32},
    'ice_thickness':      {'out': 'thickness', 'method': 'linear',  'dtype': np.float32},
    'bed_uncertainty':    {'out': 'errbed',    'method': 'linear',  'dtype': np.float32},
    'mask':               {'out': 'mask',      'method': 'nearest', 'dtype': np.int8}
}

def main():
    # 1. Setup Dask
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Processing {INPUT_FILE}...")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Missing: {INPUT_FILE}")
    if not os.path.exists(MASCON_MAP):
        raise FileNotFoundError(f"Missing Mascon Map: {MASCON_MAP}")

    # 2. Open Datasets
    # decode_cf=True handles the -9999 FillValue automatically
    ds_src = xr.open_dataset(INPUT_FILE, chunks='auto', decode_cf=True)
    ds_mascon_map = xr.open_zarr(MASCON_MAP, consolidated=True)
    
    # 3. Prepare Output using Mascon Map Coordinates
    ds_out = xr.Dataset(coords=ds_mascon_map.coords)
    ds_out.attrs = ds_src.attrs
    ds_out.attrs['history'] = "Bedmap3 Regridded to 500m. Mascon ID injected."

    # 4. Regrid Loop
    for src_var, cfg in TASKS.items():
        if src_var not in ds_src:
            print(f"  > [Skip] '{src_var}' not found.")
            continue
            
        print(f"  > Interpolating '{src_var}' -> '{cfg['out']}' ({cfg['method']})...")
        
        # Native Interpolation using the Mascon map's x and y
        regridded = ds_src[src_var].interp(
            x=ds_mascon_map.x,
            y=ds_mascon_map.y,
            method=cfg['method']
        )
        
        ds_out[cfg['out']] = regridded.astype(cfg['dtype'])

    # --- INJECT MASCON ID ---
    print("  > Injecting mascon_id from master map...")
    ds_out['mascon_id'] = ds_mascon_map['mascon_id']

    # 5. Physics Clean-up
    print("  > Auditing Physics...")
    if 'thickness' in ds_out:
        # Clip negative thickness (interpolation artifacts near 0)
        ds_out['thickness'] = ds_out['thickness'].where(ds_out['thickness'] >= 0, 0)
        
    # 6. Write to Zarr
    out_path = os.path.join(OUTPUT_DIR, "bedmap3_500m.zarr")
    print(f"  > Writing to {out_path}...")
    
    # Explicit Dask Chunking
    ds_out = ds_out.chunk({'y': 2048, 'x': 2048})
    
    # Implement Zarr Compression
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    encoding = {}
    for var in ds_out.data_vars:
        encoding[var] = {
            'compressor': compressor,
            'chunks': (2048, 2048)  # Must strictly match the Dask chunks above
        }
    
    # Write using V2 Format
    ds_out.to_zarr(
        out_path, 
        mode='w', 
        encoding=encoding, 
        zarr_format=2, 
        consolidated=False, 
        compute=True
    )
    
    zarr.consolidate_metadata(out_path)
    print("  > Success.")

if __name__ == "__main__":
    main()
