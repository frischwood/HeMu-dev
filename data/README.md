## Datasets

### Helio dataset (Heliomont + Meteosat + DEM) 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15342144.svg)](https://doi.org/10.5281/zenodo.15342144)

The Helio dataset  integrates data from three primary sources:

1. **HelioMont Algorithm outputs:**
   - Target variables:
     - SISDIR-No-Horizon: Direct radiation
     - SISDIF-No-Horizon: Diffuse radiation
     - SISGHI-No-Horizon: Global horizontal radiation
   - Diagnostic variables:
     - KI: Clear-sky index
     - ALB: Surface albedo

2. **Digital Elevation Model (DEM):**
   - Source: SRTM (Shuttle Radar Topography Mission)
   - Additional products: DEM-derived features computed using HORAYZON software

3. **Meteosat Second Generation (MSG) SEVIRI sensor:**
   - Contains all SEVIRI spectral channels
   - Exceptions: vis006 and vis008 channels are excluded

#### Temporal Coverage and Resolution
- Time period: 2015-2020
- Temporal resolution: Hourly (UTC)
- Daylight-only measurements
  - Solar zenith angle (SZA) filter: All values with SZA > 80° are excluded
  - Due to this filtering, the number of daily timesteps varies throughout the year

#### Spatial Properties
- Coordinate Reference System: WGS84
- Geographic extent:
  - Latitude: 45.75° - 47.75° N
  - Longitude: 5.75° - 10.75° E
- Spatial resolution: 
  - 0.05° in WGS84
  - Approximately 1.7km when projected to CH1903+ (Swiss local coordinate system)
- Grid alignment: All source data are aligned to a common grid from the target HelioMont variables. Potential reprojections and resampling use the nearest neighboor method. The dataset can be rebuilt using the producion version of HeMu: [https://github.com/frischwood/HeMu.git](https://github.com/frischwood/HeMu.git)


