import xarray as xr
import zarr
import os

# Configuration
MASTER_GRID_PATH = "data/processed_layers/master_grid_template.nc"
GRACE_ZARR_PATH = "data/processed/grace.zarr"
OUTPUT_ZARR = "data/processed/mascon_id_master_map.zarr"

def map_mascons_to_master():
    print(f"Loading Master Template: {MASTER_GRID_PATH}")
    ds_master = xr.open_dataset(MASTER_GRID_PATH, chunks={'x': 2048, 'y': 2048})
    
    print(f"Loading Mascon Data: {GRACE_ZARR_PATH}")
    ds_grace = xr.open_zarr(GRACE_ZARR_PATH)
    
    # Selecting the static 2D mascon_id
    da_ids = ds_grace['mascon_id']
    
    print("Mapping IDs to 500m grid via Nearest Neighbor...")
    da_ids_mapped = da_ids.reindex_like(
        ds_master, 
        method='nearest', 
        tolerance=30000 
    )
    
    # Metadata standardization
    da_ids_mapped.name = "mascon_id"
    ds_output = da_ids_mapped.to_dataset()

    # --- CRITICAL FIX START ---
    # Define our target chunk size
    target_chunks = {'x': 2048, 'y': 2048}
    
    # Explicitly rechunk the dataset to match the storage encoding.
    # This aligns the Dask graph with the Zarr storage blocks.
    print(f"Aligning Dask chunks to target {target_chunks}...")
    ds_output = ds_output.chunk(target_chunks)
    # --- CRITICAL FIX END ---

    print(f"Writing to Zarr: {OUTPUT_ZARR}")
    compressor = zarr.Blosc(cname='zstd', clevel=5, shuffle=2)
    
    # In xarray, it is often safer to let to_zarr infer chunks from the dask array 
    # instead of passing a redundant encoding['chunks'] if they already match.
    encoding = {"mascon_id": {"compressor": compressor}}

    # Write and consolidate metadata separately for version compliance
    ds_output.to_zarr(
        OUTPUT_ZARR, 
        mode='w', 
        encoding=encoding,
        consolidated=False 
    )
    
    zarr.consolidate_metadata(OUTPUT_ZARR)
    
    print("Mapping Complete.")
    print(f"Master Grid Dimensions: {ds_output.dims}")

if __name__ == "__main__":
    map_mascons_to_master()
