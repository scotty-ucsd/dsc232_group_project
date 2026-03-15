"""
step_02b_regrid_icesat.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY AUDIT LOG
-------------------------------------------------------------------------------
DATE: 2026-02-12
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION 
LOGIC:  Upsample 1km ICESat-2 to 500m Master Grid using Native Interpolation.
        CRITICAL: Includes 'data_count' and 'misfit_rms' for Phase 5 Validation.
        HOTFIX: Implemented Zarr Blosc Zstd compression.
        HOTFIX: Injects mascon_id from master map into all outputs.
        HOTFIX: Consolidated coordinates (using Mascon Map as the Master Grid).
        HOTFIX: Forced Zarr Format 2 to prevent V3 metadata issues.
-------------------------------------------------------------------------------
"""

import os
import sys
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
    "n_workers": 6, 
    "threads_per_worker": 1, 
    "memory_limit": "8GB"
}

dask.config.set({
    "distributed.worker.memory.target": 0.70,
    "distributed.worker.memory.spill": 0.85,
    "distributed.worker.memory.pause": 0.95,
})

DIRS = {
    "input": "data/processed/intermediate",
    "output": "data/processed",
    "mascon_map": "data/processed/mascon_id_master_map.zarr" 
}

TASKS = {
    'delta_h': {
        'input': 'icesat2_1km_seamless_deltah.zarr',
        'output': 'icesat2_500m_deltah.zarr',
        'var_list': ['delta_h', 'delta_h_sigma', 'ice_area', 'data_count', 'misfit_rms']
    },
    'lag1': {
        'input': 'icesat2_1km_seamless_lag1.zarr',
        'output': 'icesat2_500m_lag1.zarr',
        'var_list': ['dhdt', 'dhdt_sigma', 'ice_area']
    },
    'lag4': {
        'input': 'icesat2_1km_seamless_lag4.zarr',
        'output': 'icesat2_500m_lag4.zarr',
        'var_list': ['dhdt', 'dhdt_sigma', 'ice_area']
    }
}

def process_task(task_key: str, config: dict, ds_mascon_map: xr.Dataset, client):
    """Executes the interpolation for a single product."""
    in_path = os.path.join(DIRS['input'], config['input'])
    out_path = os.path.join(DIRS['output'], config['output'])
    
    if not os.path.exists(in_path):
        print(f"[Skip] Input not found: {in_path}")
        return

    if os.path.exists(out_path):
        print(f"[Warn] Overwriting existing output: {out_path}")

    print(f"\n[Task] Processing {task_key} -> {config['output']}...")
    
    ds_src = xr.open_zarr(in_path)
    
    missing = [v for v in config['var_list'] if v not in ds_src]
    if missing:
        print(f"  > [CRITICAL WARN] Missing variables in source: {missing}")
        valid_vars = [v for v in config['var_list'] if v in ds_src]
    else:
        valid_vars = config['var_list']
        
    ds_subset = ds_src[valid_vars]
    
    print("  > Applying Native Bilinear Interpolation (xr.interp)...")
    # Using ds_mascon_map directly for coordinates!
    ds_regridded = ds_subset.interp(
        x=ds_mascon_map.x, 
        y=ds_mascon_map.y, 
        method="linear",
        kwargs={"fill_value": np.nan} 
    )

    print("  > Injecting mascon_id from master map...")
    ds_regridded['mascon_id'] = ds_mascon_map['mascon_id']
    
    ds_regridded.attrs = ds_src.attrs
    ds_regridded.attrs['history'] = f"Upsampled to 500m using xr.interp (linear). Mascon ID injected. Source: {config['input']}"
    ds_regridded.attrs['resolution'] = "500m"
    
    for var in ds_regridded.data_vars:
        if ds_regridded[var].dtype == np.float64:
             ds_regridded[var] = ds_regridded[var].astype(np.float32)
        if var in ds_src:
            ds_regridded[var].attrs = ds_src[var].attrs
    
    print(f"  > Writing to {out_path}...")
    ds_regridded = ds_regridded.chunk({'time': 1, 'y': 2048, 'x': 2048})
    
    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    encoding = {v: {'compressor': compressor} for v in ds_regridded.data_vars}
    
    for var in ds_regridded.data_vars:
        if 'time' in ds_regridded[var].dims:
            encoding[var]['chunks'] = (1, 2048, 2048)
        else:
            encoding[var]['chunks'] = (2048, 2048)

    # Force V2 and standard consolidation
    ds_regridded.to_zarr(
        out_path, 
        mode='w', 
        encoding=encoding, 
        zarr_format=2, 
        consolidated=False, 
        compute=True
    )
    zarr.consolidate_metadata(out_path)
    
    print(f"  > Success: {task_key}")


def main():
    cluster = LocalCluster(**DASK_CONFIG)
    client = Client(cluster)
    print(f"[System] Dask Client: {client.dashboard_link}")
        
    if os.path.exists(DIRS['mascon_map']):
        print(f"[System] Loading Mascon Map from {DIRS['mascon_map']}")
        ds_mascon_map = xr.open_zarr(DIRS['mascon_map'], consolidated=True)
    else:
        print("[Error] Mascon Map missing. Run the mapping script first.")
        sys.exit(1)

    for key, cfg in TASKS.items():
        try:
            process_task(key, cfg, ds_mascon_map, client)
        except Exception as e:
            print(f"[Error] Failed {key}: {e}")
            import traceback
            traceback.print_exc()

    print("[System] Step 02b Complete.")

if __name__ == "__main__":
    main()
