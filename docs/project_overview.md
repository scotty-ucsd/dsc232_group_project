## Abstract 
- riff off of template and first 3 key papers will write itself

## Datasets

### 1. [ICESat-2](https://nsidc.org/data/icesat-2): ALT06 dataproduct
    - ATLAS  instrument determines the range between the satellite and the Earth’s surface 
        - measures time delay of light *bouncing* off a surface
    - ATL06: Land Ice Elevation
        - Surface height for each beam pair

    #### 1.1 Features
    - time
    - latitude
    - longitude
    - surface height
        - bounce photon off surface how long until instrument catches it
            - $SH = c\cdot t$

    #### 1.2 Comments
    - longitude: -180 to 180
    - very fine resolution ~20m-40m

### 2. [GRACE/GRACE-FO](https://www2.csr.utexas.edu/grace/RL0603_mascons.html)

    - Description: global mass anomalies anomalies
    - key indicatior of how earth systems interact with one another
        - how energy move from place to place or system to system

    #### 2.1 Features
    - time
    - latitude
    - longitude
    - equivalent water thickness
        - spread thin layer of water over 1 square meter
        - note: are you accounting for solid/liquid density diff??

    #### 2.2 Comments
    - longitude: 0 to 360
    - larger resolution than ICESat-2 0.25 deg ~25km
        -note: dist varies slightly with latitude quick estimate 

### 3. [BedMap3](https://data.bas.ac.uk/full-record.php?id=GB/NERC/BAS/PDC/01615)
    - "what would Anarticica look like if all ice melted?"

    #### 3.1 Features
    - Cell Area Categories
        - note: convert these to binary classifiers (OHE)
        - 1 = grounded ice
        - 2 = transiently grounded ice shelf
        - 3 = floating ice shelf
        - 4 = rock 
    - Bed topography
        - note: ncdump -h on bedmap3.nc to find dataset/variable name
        - can use this with surface height from ICESat-2 (mean_h_li | std_h_li ) 
            - this gives us ice thickness 
    #### 3.2 Comments
    - longitude: not sure but will convert once down loaded
    - larger resolution than ICESat-2 but smaller than GRACE
### 4.Bonus if needed
- https://mcm.lternet.edu/real-time-data-dashboards
    - like U.S. SNOTEL
## Key Papers

### [Understanding Regional Ice Sheet Mass Balance: Remote Sensing, Regional Climate Models, and Deep Learning](https://www.proquest.com/docview/2394919770/abstract/BC130FB0A627484FPQ/1?accountid=14524&sourcetype=Dissertations%20&%20Theses)
- 2019
- "we analyze the mass balance of glaciers across the ice sheets at basin and sub-basin scales using satellite gravimetric data from the Gravity Recovery and Climate Experiment (GRACE) mission using a novel regionally-optimized mascon methodology, as well as Mass Budget Method (MBM) estimates from grounding line discharge measurements and surface mass balance from regional climate models"
- "we focus on improving the monitoring and understanding of glacier dynamics by implementing a deep Convolutional Neural Network (CNN) to automatically delineate glacier calving fronts from Landsat imagery on the Greenland Ice Sheet"
### [Advancing Scalable Methods for Surface Water Monitoring: A Novel Integration of Satellite Observations and Machine Learning Techniques](https://www.mdpi.com/2076-3263/15/7/255)
- 2025
- GRACE/GRACE-FO and ICESat-2 for Surface Water Volume estimates
- "Our results underscore the value of improved vertical DEM availability for global hydrological studies and offer a scalable framework for future applications. Future work will focus on expanding our DEM dataset, further validation, and scaling this methodology for global applications"
### [Decadal Survey for Earth Science and Applications from Space](https://www.nationalacademies.org/read/24938)
- Protips from the pros
### [BEDMAP3](https://www.nature.com/articles/s41597-025-04672-y)
- just info on BEDMAP3 dataset
### [High-Spatial-Resolution Mass Rates From GRACE and GRACE-FO: Global and Ice Sheet Analyses](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021JB023024)
### [Contributions of GRACE to understanding climate change](https://pmc.ncbi.nlm.nih.gov/articles/PMC6750016/)
### https://www.nature.com/articles/s41467-018-05002-0


### other sources for questions

#### this seems very promising 
https://geo-smart.github.io/

#### shot in the dark.. hard maybe??
https://ieeexplore.ieee.org/document/9832203
http://utam.gg.utah.edu/Classes/ML/LAB1/index.labs.html
https://pubs.geoscienceworld.org/seg/books/edited-volume/2729/Machine-Learning-Methods-in-Geoscience

