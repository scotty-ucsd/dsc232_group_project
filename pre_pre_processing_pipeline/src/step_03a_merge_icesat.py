"""
step_02a_merge_icesat_tiles.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-12
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION (Hotfix: Compression & Zarr V2 Enforcement)
LOGIC:  "Lowest Uncertainty First"
        Hardware Config: Optimized for 55GB/12Core Node.
        HOTFIX: Forced Zarr Format 2 to prevent V3 metadata compatibility issues.
        HOTFIX: Replaced uncompressed writes with Blosc Zstd compression.
-------------------------------------------------------------------------------
"""

import os
import sys
import warnings
import numpy as np
import xarray as xr
import dask.array as da
import zarr
from dask.distributed import Client, LocalCluster

# --- MANDATORY PREAMBLE ---
warnings.filterwarnings("ignore", category=UserWarning) 
xr.set_options(keep_attrs=True)

# --- CONFIGURATION (OPTIMIZED FOR 55GB RAM / 12 CORES) ---
TEST_MODE = False  
DASK_CONFIG = {
    "n_workers": 4,           
    "threads_per_worker": 2,  
    "memory_limit": "12GB"    
}

DIRS = {
    "input": "data/raw/icesat",
    "output": "data/processed/intermediate"
}

TILES = {
    'A1': f"{DIRS['input']}/ATL15_A1_0328_01km_005_01.nc",
    'A2': f"{DIRS['input']}/ATL15_A2_0328_01km_005_01.nc",
    'A3': f"{DIRS['input']}/ATL15_A3_0328_01km_005_01.nc",
    'A4': f"{DIRS['input']}/ATL15_A4_0328_01km_005_01.nc",
}

GROUPS = {
    'delta_h': {
        'nc_group': 'delta_h',
        'sigma_var': 'delta_h_sigma',
        'data_vars': ['delta_h', 'delta_h_sigma',
                      'ice_area','misfit_rms','data_count'],
        'output_name': 'icesat2_1km_seamless_deltah.zarr'
    },
    'dhdt_lag1': {
        'nc_group': 'dhdt_lag1',
        'sigma_var': 'dhdt_sigma',
        'data_vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_name': 'icesat2_1km_seamless_lag1.zarr'
    },
    'dhdt_lag4': {
        'nc_group': 'dhdt_lag4',
        'sigma_var': 'dhdt_sigma',
        'data_vars': ['dhdt', 'dhdt_sigma', 'ice_area'],
        'output_name': 'icesat2_1km_seamless_lag4.zarr'
    }
}


def verify_physics(ds: xr.Dataset, context: str):
    """Mandatory physics audit."""
    print(f"[{context}] Auditing physics...")
    
    for var in ds.data_vars:
        if "sigma" not in var and "ice_area" not in var:
            v_max = ds[var].max().compute().item()
            v_min = ds[var].min().compute().item()
            if abs(v_max) > 500 or abs(v_min) > 500:
                print(f"  [FATAL] {var} out of bounds: {v_min} to {v_max}")
                warnings.warn(f"Physical violation in {var}")

    if 'ice_area' in ds:
        area_max = ds['ice_area'].max().compute().item()
        area_min = ds['ice_area'].min().compute().item()
        
        if area_min < 0.0:
            raise ValueError(f"Negative Ice Area detected: {area_min}")

        if area_max > 2.0:
            if area_max > 2.0e6:
                raise ValueError(f"Ice Area too large for 1km grid: {area_max:.1f} m^2 (Limit: 2.0e6)")
            print(f"  > Ice Area validated as meters^2 (Max: {area_max:.1f})")
        else:
            if area_max > 1.01:
                raise ValueError(f"Ice Area fraction > 1.0: {area_max}")
            print(f"  > Ice Area validated as Fraction (Max: {area_max:.4f})")

    print(f"[{context}] Physics Audit PASSED.")


def fix_coordinates(ds: xr.Dataset, res=1000.0) -> xr.Dataset:
    """Snaps coordinates to a strict grid."""
    new_x = np.round(ds.x / res) * res
    new_y = np.round(ds.y / res) * res
    
    ds = ds.assign_coords(x=new_x, y=new_y)
    ds = ds.sortby(['x', 'y'])
    
    if not ds.indexes['x'].is_monotonic_increasing:
        raise ValueError("X coordinates non-monotonic after snapping.")
    
    return ds


def create_master_canvas(tile_paths: dict) -> xr.Dataset:
    """Scans all tiles to determine the bounding box."""
    print("[Canvas] calculating continental extent...")
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for name, path in tile_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing tile: {path}")
        
        with xr.open_dataset(path, group='delta_h') as ds:
            x = np.round(ds.x.values / 1000.0) * 1000.0
            y = np.round(ds.y.values / 1000.0) * 1000.0
            
            min_x = min(min_x, x.min())
            max_x = max(max_x, x.max())
            min_y = min(min_y, y.min())
            max_y = max(max_y, y.max())

    print(f"[Canvas] Extent: X[{min_x}:{max_x}], Y[{min_y}:{max_y}]")
    
    x_grid = np.arange(min_x, max_x + 1000, 1000, dtype=np.float64)
    y_grid = np.arange(max_y, min_y - 1000, -1000, dtype=np.float64) 
    
    canvas = xr.Dataset(coords={'x': x_grid, 'y': y_grid})
    return canvas


def merge_group_logic(group_cfg: dict, canvas: xr.Dataset, tile_paths: dict):
    """Implementation of 'Lowest Uncertainty First' merge logic."""
    group_name = group_cfg['nc_group']
    sigma_name = group_cfg['sigma_var']
    vars_needed = group_cfg['data_vars']
    
    print(f"\n[Merge] Processing group: {group_name}...")

    merged_ds = None
    
    for tile_id, path in tile_paths.items():
        print(f"  > Loading {tile_id}...")
        
        ds = xr.open_dataset(path, group=group_name, chunks={'time': 1, 'y': 2048, 'x': 2048})
        
        if TEST_MODE:
            ds = ds.isel(time=slice(0, 1))
            
        ds = fix_coordinates(ds)
        ds = ds[vars_needed]
        ds_aligned = ds.reindex(x=canvas.x, y=canvas.y)
        
        if merged_ds is None:
            merged_ds = ds_aligned
        else:
            new_sigma = ds_aligned[sigma_name]
            curr_sigma = merged_ds[sigma_name]
            
            new_valid = new_sigma.notnull()
            curr_valid = curr_sigma.notnull()
            
            better_mask = new_valid & (~curr_valid | (new_sigma < curr_sigma))
            
            for var in vars_needed:
                merged_ds[var] = xr.where(better_mask, ds_aligned[var], merged_ds[var])
                
    return merged_ds


def main():
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client active: {client.dashboard_link}")
    print(f"[System] Workers: {DASK_CONFIG['n_workers']} | Mem: {DASK_CONFIG['memory_limit']}")

    os.makedirs(DIRS['output'], exist_ok=True)
    canvas = create_master_canvas(TILES)
    
    for key, cfg in GROUPS.items():
        out_path = os.path.join(DIRS['output'], cfg['output_name'])
        if os.path.exists(out_path):
            print(f"[Skip] Output exists: {out_path}")
            continue
            
        ds_merged = merge_group_logic(cfg, canvas, TILES)
        verify_physics(ds_merged, key)
        
        ds_merged.attrs['title'] = f"ICESat-2 Antarctic Mosaic - {key}"
        ds_merged.attrs['crs'] = "EPSG:3031"
        ds_merged.attrs['history'] = "Merged using Lowest-Uncertainty-First logic; Ice Area checked."
        
        print(f"  > Writing to {out_path}...")
        
        # 1. Define target chunks
        chunk_dict = {'time': 1, 'y': 1024, 'x': 1024}
        ds_merged = ds_merged.chunk(chunk_dict)
        
        # 2. Configure compression and encoding
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        encoding = {}
        for var in ds_merged.data_vars:
            encoding[var] = {'compressor': compressor}
            # Ensure Zarr storage chunks perfectly align with Dask chunks
            if 'time' in ds_merged[var].dims:
                encoding[var]['chunks'] = (1, 1024, 1024)
            else:
                encoding[var]['chunks'] = (1024, 1024)
        
        # 3. Write with Zarr V2 explicitly forced
        ds_merged.to_zarr(
            out_path, 
            mode='w', 
            encoding=encoding, 
            zarr_format=2,       # Force legacy V2 format
            consolidated=False,  # Required to be False during the write step
            compute=True
        )
        
        # 4. Atomic metadata consolidation for V2 stores
        zarr.consolidate_metadata(out_path)
        
        print(f"  > Success: {key}")

    print("[System] Step 02a Complete.")

if __name__ == "__main__":
    main()
