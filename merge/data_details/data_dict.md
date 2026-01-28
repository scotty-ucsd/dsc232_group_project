# Data Dictionary 
---

1. Spatial & Temporal 

Column,Type,Source,Definition & ML Importance
"x, y",f64,ICESat-2,"EPSG:3031 Coordinates (m). The native geometry. Essential for spatial splitting (e.g., K-Fold)."
"lat, lon",f64,Derived,EPSG:4326 Coordinates (deg). Used to link global datasets.
timestamp,date,ICESat-2,Time. The exact date of the laser shot. Critical for seasonal trends.
tile,i8,Pipeline,"Granule ID. Metadata for provenance (A1, A2, A3, A4)."


2. Target 

Column,Type,Source,Definition & ML Importance
delta_h,f32,ICESat-2,TARGET VARIABLE. Surface height change relative to reference epoch. This is what you predict.
dhdt_lag1,f32,ICESat-2,Velocity (Short). 3-month rate of change ($\frac{m}{yr}). Provides instantaneous momentum.
dhdt_lag4,f32,ICESat-2,Velocity (Long). 1-year rate of change ($frac{m}{yr}$). Provides annual trend context.

3. Data Quality 

Column,Type,Source,Definition & ML Importance
delta_h_sigma,f32,ICESat-2,Uncertainty. Estimated error. Use as Sample Weight (trust low sigma more).
misfit_rms,f32,ICESat-2,Roughness. High RMS = rough surface (crevasses). Affects albedo/melt.
data_count,f32,ICESat-2,Density. Photon count per pixel. Low count = potentially noisy measurement.
ice_area,f32,ICESat-2,Coverage. Valid land ice area in the cell ($m^{2}). Used to normalize mass calculations.

4. Geometry
Column,Type,Source,Definition & ML Importance
bedmap_surface...,f32,BEDMAP3,Surface Elevation (m). Static reference surface.
bedmap_bed_topo...,f32,BEDMAP3,Bedrock Elevation (m). Determines grounding lines.
bedmap_ice_thick...,f32,BEDMAP3,Ice Thickness (m). Critical for ice dynamics (thicker ice flows differently).
bedmap_mask,f32,BEDMAP3,"State Flag. 1=Grounded, 2=Transient, 3=Floating. The most important categorical feature."
bedmap_bed_unc...,f32,BEDMAP3,Bed Uncertainty. High uncertainty areas might have erroneous grounding lines.
base_elev,f32,Derived,Ice Base (m). Elevation of the ice bottom (surface - thickness).
static_draft...,f32,Derived,Draft (m). Positive-down depth of the ice interface. Used for ocean lookup.

5. Forcing (The Drivers)
Column,Type,Source,Definition & ML Importance
lwe_thickness,f32,GRACE,Gravity (Mass). Liquid Water Equivalent (cm). Indicates large-scale mass loss trends.
thetao_interface,f32,GLORYS,Ocean Temp ($^{\circ}$C). Potential temperature at the ice base. The #1 driver of basal melt.
so_interface,f32,GLORYS,Salinity (psu). Salinity at the ice base. Affects freezing point and density.
mascon_center_...,f32,GRACE,"Region ID. Coordinates of the parent Mascon. Use to group pixels into ""Regions."""
land/ocean_mask,f32,GRACE,Leakage Context. Helps model identify if gravity signal is from land or ocean.

---

## Feature Engineering

---


1. dist_to_grounding_line
- Mechanism: Melt is highest where the ice first un-grounds (the hinge zone).
- Implication: Extract bedmap_mask==2 (Grounding Zone) pixels from the full grid, build a KDTree, and query your subset points against it.

2. surface_slope
- Mechanism: Gravity drives flow down-slope. High slope ≈ high driving stress.
- Implication: Compute np.gradient on the full BEDMAP surface grid (EPSG:3031 is metric, so math works directly).

3. bedmap_ratios
- Why: Are we in a narrow fjord or a wide shelf?
- Implication: Convolve the full Bedmap mask with a kernel (e.g., 10km radius) to get percent floating nearby.

4. cyclic_time
- Why: Ice melts in summer (Jan/Feb).
- Implication: sin and cos of Day-of-Year.

5. thermal_forcing
- Mechanism: Melt is proportional to thermal driving.
- Formula: Forcing = T_ocean - T_freezing
- T_freezing Implementation:
  $$T_{f} \approx -0.057 \cdot S_{ocean} - 0.00076 \cdot Depth$$
  *(Note the negative sign for depth dependence)*

6. retrograde_bed
- Mechanism: If the bed gets deeper inland (negative slope), the ice is unstable (MISI).  
- Implication: Dot product of Ice Flow Direction (assumed surface gradient) and Bed Slope.
