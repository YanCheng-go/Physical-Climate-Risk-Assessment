# Physical-Climate-Risk-Assessment
Physical climate risk assessment framework for power sector designed by WRI in collaboration with EBRD.

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


# Scope of work
This github repository is for reproducing the results in our [journal article]() and WRI [working paper](). We are adapting the scripts to enable more general usages. Before the official version is published, you may encounter errors to run the scripts for your own cases. If so, please do not hesitate to contact us or post issues on the project page.

This python project enables the assessment of physical climate risks for EBRD’s thermal and hydro power plants. The thermal assessment framework consists of five modules corresponding to five climate hazards, i.e., air temperature rise, water temperature rise, droughts, floods, water stress. The hydro assessment relied on an artificial recurrent neural network, namely Long-Short Term Memory (LSTM) and regression models. Please refer to [our paper](URL) for more information. Figure 1 illustrates the assessment framework.

![Physical climate risk assessment framework](docs/ebrd-physical-climate-risks%20-%20high-level%20flowchart-vertical.svg)

# Project folder structure
For the sake of simplicity, we only shared the scripts needed to reproduce the results on this GitHub Repository. After running associated scripts to complete reproducing the results, you will see additional folders added to your local project folder as shown in the second tree diagram. These additional folders are folders for raw datasets, interim output, ready-to-use input datasets, interim assessment results, and final assessment results and figures.
```
.
├─ scripts  
│  │  default.py 
│  │  ear5.py <- pre-process ERA5 datasets and predict wet-bulb temp.
│  │  execute_2030.py <- execution script for period 2030
│  │  execute_india.py <- execution script for validation
│  │  power_plant_physical_climate_risk_assessment.py <- thermal assessment
│  │  utils.py <- general utilities
│  │  visualization.py <- data pre-processing/restructure and viz
│  │  __init__.py
│  │
│  ├─ data
│  │  │  data_configs.py <- dataset configurations/properties
│  │  │  extract_water_temp.py <- extract water temperature from netCDF4 datasets
│  │  │  PL_ebrd_gddp.py <- download GDDP time series from GEE
│  │  │  PL_gee_ebrd_tpp_era5_data_download.py <- download ERA5 time series from GEE
│  │  └─ __init__.py
│  │
│  ├─ exec
│  │  │  exe_download.py <- download datasets
│  │  │  exe_prep.py <- pre-process/restructure raw datasets 
│  │  │  exe_thermal_assessment.py <- thermal assessment
│  │  │  exe_viz.py <- generation figures used in the article
│  │  └─ __init__.py   
│  │
│  ├─ features
│  │  │  correct_bias.py <- harmonize/correct bias GDDP and ERA5 data
│  │  │  feature_generation.py <- calculate additional variables 
│  │  └─ __init__.py  
│  │
│  ├─ report
│  │  │  data_stats.py <- summary statistics of input climate datasets
│  │  │  master_report.py <- combine and organize preliminary results into one table 
│  │  │  report_stats.py <- analyze preliminary results
│  │  └─ __init__.py
│  │
│  ├─ utils
│  │  │  utils.py <- utilities
│  │  └─ __init__.py  
│  │
│  └─ visualizations
│      │  boxplots.py <- group results by plant cooling or fuel-turbine type
│      │  climate_trajectories.py <- plot time series of input climate variables 
│      │  probability_changes.py <- exceedance probability of design water/air temp.
│      │  validation_with_indian_plants.py <- figures in the validation section
│      │  viz_configs.py <- color schemes
│      │  waterfall.py <- visualize portfolio-level results using waterfall chart
│      └─ __init__.py
│  
├─ tpp info
│  │  indian_plant_candidates_v1.xlsx <- information of all Indian plant candidates
│  │  tpp_locs.xlsx <- coordinates of plants and their water sources
│  │  tpp_regroup.xlsx <- regrouping plants by cooling and fuel-turbine types
│  │  tpp_working.xlsx <- information of EBRD invested thermal and hydro plants
│  │  tpp_working_india.xlsx <- information of selected Indian plant for validation
│  │  tpp_working_india_CCPP.xlsx <- information of Indian CCPP plants
│  │  tpp_working_india_valuthur.xlsx <- one of the selected Indian CCPP plants 
│  └─ tpp_working_master_wet-to-air.xlsx <- pseudo upgrade from wet-cool to air-cool tech technology  
│
├─ vulnerability
│  │  error_weekly_vs_daily.xlsx <- pre-calculated coefficients to correct bias caused by inconsistent temporal resolution of historical and projection water temperatures.
│  │  vulnerability_factors_20210409_forSensitivityAnalysis.xlsx <- vulnerability factors used in sensitivity analysis of discharge water temp.
│  └─ vulnerability_factors_20210409.xlsx <- default vulnerability factors from literatures
│
├─ spei <- SPEI time series for all theraml plants, input data for the drought module 
│        
├─ tpp water temp all <- water temperature time series for all thermal plants extracted from global water temperature data by plant geolocations
│
├─ data <- ancillary datasets
│  └─ external
│     └─ india_generation
│        │  dgr2.csv <- daily power generation of Indian plants
│        └─ dgr10.csv <- daily outage records of Indian plants
│
├─ final assessment <- where interim output will be saved
│  └─ hydro_result
│     └─ portfolio_level_hydro_2030_gen_losses.xlsx <- portfolio-level hydro results
│
├─ docs <- documentations, etc..
│
├─ LICENSE
├─ README.md
└─ requirements.txt <- required python libraries/discrepancies
```

# Replication
## Prerequisites
### Fork and clone repository
Download/clone this github repository. The size of this repository is ~2.5 GB and will increase to ~16 GB after reproducing the results.
Note: it is better not to save the repository in OneDrive; otherwise, please turn off sync for the project folder. I used to encounter issues running scripts saved in OneDrive. I would highly recommend you to save the entire project locally. 

### Set up a python environment 
1.	[Download Python 3.8]() and Python IDE, such as PyCharm
2.	Create a virtual environment
3.	Install required python libraries 
4.	Open the project and run with the virtual environment 

### Download data
1. Download [Flood data from AWS S3](https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/index.html) and note the folder path where the data is saved. This file is ~14.3 GB.
2. Download [Water stress from AWS S3]() and note the folder path where the data is saved. This file is ~361 MB.
3. (optional) Download [plant-level climatic datasets]() extracted from ERA5 and GDDP database on Google Earth Engine. You can also choose to run associated scripts to download data from Google Earth Engine yourself. This file is ~335 MB.
4. (optional) Download [interim datasets]() only if you'd like to skip some data pre-processing steps. This file is ~12.3 GB.
5. (optional) Download [final assessment results](). This file is ~ 6.51 MB.

### Google Earth Engine account
If you wish to run the scripts from data downloading, you’d have to have a Google Earth Engine (GEE) account since some of the climate datasets need to be downloaded from GEE. Otherwise, please head to [this website]() to download all raw datasets.

## Thermal assessment
The execution of thermal assessment is divided into three steps, including data downloading, data pre-processing, and risk assessment for thermal plants.

### Download input datasets
To download raw datasets, i.e., EAR5 and NEX-GDDP, please run scripts/exec/exe_download.py. 

If you wish to skip this step, please download [plant-level climatic datasets]() and save files under the root folder of this project. 

### Pre-process input data
Next, please run scripts/exec/exe_prep.py to restructure datasets. 

If you wish to skip this step, please download [interim datasets]() and save files under the root folder of this project.

### Assessment 
Lastly, please run scripts/exec/exe_thermal_assessment.py to quantify generation losses caused by each type of climate hazard. 

If you wish to skip this step, please download [final assessment results]() 

### (Visualization)
You can also regenerate all figures in our article by running scripts/exec/exe_viz.py

### All in one go
To run all steps mentioned above except visualization in one go, please execute execution_2030.py.

# Additional use cases
## Assess individual plant
## Assess individual module
## Analyse new plants other than EBRD portfolio
## Update vulnerability factors
## Update thresholds
