"""
Preprocess input climatic datasets.

1. Data extraction: extract water temperatures from netCDF datasets.
2. Data transformation: reconstruct and uniform input data format/structure.
3. Feature engineering: calculate additional variables, i.e., bias-corrected climate variables, wet-bulb temperatures.
"""

import os
work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))  # work directory

# ######################################################################################################################
# User-defined parameters

daily_nc_file = r'D:\WRI\temp\waterTemperature_mergedV2.nc'  # daily water temperature datasets over the historical period.
weekly_folder = r'D:\WRI\Water Temperature'  # weekly water temperature datasets over the projection period.
# ######################################################################################################################


# User-adjustable parameters
tpp_working_fp = os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx')  # power plant info
# specify what time frame to be assessed
fut_eval_yrs = [2010,
                2049]  # projection period 2030, will also impact the time frame of the bias correction and wet-bulb temperature calculation.

# Constants -> default values are retrieved from input datasets,
# but can be adjusted accordingly for a subset of the default settings.
plant_id_list = range(1, 26)  # all plants
wbtemp_hist_timespan = (1980, 2005)  # time span concerned as baseline condition of wet-bulb temperatures from ERA5.
era5_timespan = (1980, 2019)
tpp_water_temp_folder_name = 'tpp water temp all'  # where to save plant-level water temperature time series retrieved from netCDF files.
tpp_watertemp_post_folder_name = 'watertemp_output_temp_all'
tpp_airtemp_folder_name = 'tpp_climate_gddp_all'
tpp_airtemp_exp_folder_name = 'new tpp climate corr'
tpp_airtemp_post_folder_name = 'tpp_climate_gddp_restructure_all_withAirTempAvg'
tpp_airtemp_biasCorrected_folder_name = 'tpp_climate_gddp_restructure_all_withAirTempAvg_biasCorrected'
tpp_gddp_wbtemp_folder_name = 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols'
tpp_era5_folder_name = 'ear5'
tpp_wbtemp_era5_folder_name = 'ear5_wetbulbtemp'
wbtemp_model_folder_name = 'wbtemp_model_nonorm_ols'

print(f'Start!\n')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Extract water temperature (baseline time frame as defined and entire projection time frame)
# ===========================================================================================

# from scripts.data import extract_water_temp
#
# print('...Start to retrieve water temperatures from netCDF...\n')
# extract_water_temp.master(project_directory=work_directory, daily_nc_file=daily_nc_file, weekly_folder=weekly_folder,
#                           output_folder_name=tpp_water_temp_folder_name, tpp_working_fp=tpp_working_fp)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Restructure input datasets
# ==========================

from scripts import visualization

# Restructure water temperatures
print('\n...Start to restructure water temperature datasets...')
visualization.restructure_dataset_master(dataset='tpp water temp',
                                         work_directory=work_directory,
                                         output_folder_name=tpp_watertemp_post_folder_name,
                                         watertemp_folder=os.path.join(work_directory, tpp_water_temp_folder_name))
# Air temperatures
print('\n...Start to restructure air temperature datasets...')
visualization.restructure_dataset_master(dataset='tpp air temp',
                                         work_directory=work_directory,
                                         output_folder_name=tpp_airtemp_post_folder_name,
                                         airtemp_folder=os.path.join(work_directory, tpp_airtemp_folder_name))
# Air temperatures for the air temperature module
from scripts import power_plant_physical_climate_risk_assessment

print('\n...Start to export restructured air temperatures from GDDP database for the air temperature module...')
tpr = power_plant_physical_climate_risk_assessment.TppPhyRisk()
tpr.export_restructured_airtemp(in_folder=os.path.join(work_directory, tpp_airtemp_folder_name),
                                out_folder=os.path.join(work_directory, tpp_airtemp_exp_folder_name))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Calculate wet-bulb temperatures from ERA5 datasets
# ==================================================

from scripts import ear5

print('\n...Start to calculate wet-bulb temperatures from ERA5 datasets...')
era5 = ear5.Ear5(work_directory=work_directory)
era5.data_folder = os.path.join(era5.work_directory, tpp_era5_folder_name)
era5.output_directory = os.path.join(work_directory, tpp_wbtemp_era5_folder_name)
df_batch = era5.restructure_batch(save_output=False)
era5.cal_wbtemp_batch(df_batch=df_batch, save_output=True, output_directory=era5.output_directory)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Bias corrections for GDDP and ERA5
# ==================================

from scripts.features import feature_generation as feagen

print('\n...Start bias-correction for GDDP datasets...')
gddp_gf = feagen.NexGddp(project_folder=work_directory, gddp_folder=tpp_airtemp_post_folder_name)
gddp_gf.correct_bias_batch(save_output=True, output_folder_name=tpp_airtemp_biasCorrected_folder_name,
                           prj_start_year=fut_eval_yrs[0], prj_end_year=fut_eval_yrs[1],
                           correct_prj=True, correct_bc=True)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Model wet-bulb temperature as a function of air temperatures and precipitations from ERA5 datasets
# ==================================================================================================

from scripts import ear5

print(
    '\n...Start to model the relationship between wet-bulb temperatures and air temperatures and precipitations using ERA5 datasets...')
era5 = ear5.Ear5(work_directory=work_directory)
era5.data_folder = os.path.join(era5.work_directory, tpp_wbtemp_era5_folder_name)
era5.output_directory = os.path.join(era5.work_directory, wbtemp_model_folder_name)
era5.train_mlr_batch(indicator=[era5.wbtemp_name, era5.pr_name, era5.airtemp_name],
                     save_output=True, model='ols', X_range=None,
                     output_directory=era5.output_directory)
# # Read pkl files
# for basename in os.listdir(era5.output_directory):
#     if basename.endswith('.pkl'):
#         print(basename)
#         with open(os.path.join(era5.output_directory, basename), "rb") as input_file:
#             d = pickle.load(input_file)
#             print(d)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Predict wet-bulb temperatures using air temperatures and precipitations from GDDP datasets
# ==========================================================================================

from scripts import ear5

print('\n...Start to calculate wet-bulb temperatures as a function of air temperatures and precipitations from GDDP...')
era5 = ear5.Ear5(work_directory=work_directory,
                 output_directory=os.path.join(work_directory, tpp_gddp_wbtemp_folder_name))
era5.predict_wbtemp_batch(model_folder=os.path.join(work_directory, wbtemp_model_folder_name),
                          data_folder=os.path.join(work_directory, tpp_airtemp_biasCorrected_folder_name),
                          output_directory=era5.output_directory,
                          plant_id_list=plant_id_list,
                          hist_timespan_str='-'.join(map(str, wbtemp_hist_timespan)),
                          futu_timespan_str='-'.join(map(str, fut_eval_yrs)),
                          era5_timespan='-'.join(map(str, era5_timespan)))
