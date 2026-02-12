## **QUESTION 1 — What are the major elevation‑change regimes in antarctic peninsula**  
**Method:** K‑Means Clustering on elevation‑change metrics (dh/dt, seasonal cycle, slope, latitude)

**Purpose:** Identify emergent dynamical zones such as:  
- rapidly thinning outlet glaciers  
- stable interior ice  
- seasonal melt‑driven surface changes  
- marine‑terminating glacier basins  

**Output:** Regime map of antarctic peninsula capturing natural ice‑dynamical classes.

---

## **QUESTION 2 — Can elevation change predict monthly mass change?**  
**Method:** Gradient‑Boosted Trees / Random Forest Regression  
**Target:** GRACE mass‑change anomaly for each mascon  
**Features:**  
- ATL06 elevation‑change rates  
- seasonal metrics  
- latitude/longitude  
- lagged elevation change (lead–lag response)  

**Expected Insight:**  
Regions with strong dynamic thinning should show predictive coupling between ICESat‑2 geometry and GRACE mass.  
Model may reveal where gravitational changes lag elevation changes.

**Output:** Baseline predictive map of mass‑change skill.

---

## **QUESTION 3 — How much of GRACE’s signal is explainable from elevation change alone?**  
**Method:** XGBoost‑style regression with feature importance ranking

**Features:**  
- dh/dt  
- interannual elevation anomaly  
- dynamic thickening/thinning patterns

**Expected Outcome:**  
Moderate reproduction of mass‑change anomaly fields, particularly in:  
- fast outlet glaciers  
- marine‑terminating basins  
- interior accumulation zones  

**Output:** Explained variance heatmap over mascons.

---

## **QUESTION 4 — Where is geometry–mass coupling strongest?**  
**Method:** Spatial correlation between ATL06 elevation time series and GRACE mascons  
**Expected Insight:**  
- Strong positive coupling at major outlet glaciers  
- Weak coupling over high interior plateau  
- Complex coupling in areas with subglacial hydrology activity  

**Output:** antarctic peninsula coupling map.

