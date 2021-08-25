"""
Validation using daily observations of Indian plants, including the following steps.

1. Prerequisites -> Download ERA5 from Google Earth Engine
2. Pre-process/restructure ERA5 datasets.
3. Restructure ERA5 in the form of input for air_temperature_module() in the power_plant_physical_climate_risk_assessment.py
4. Execute air temperature assessment.
5. Visualize results and save visualization in the form of png, i.e., model_vs_obs.png -> comparison between observed and modelled generation losses at the plant and portfolio level, indian_gen-at_reg.png -> daily generations vs air temperatures.

"""

import os
from datetime import datetime

work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
india_data_folder = os.path.join(work_directory, 'india_data')
if not os.path.exists(india_data_folder):
    os.mkdir(india_data_folder)
report_folder_name = 'reports'
report_folder = os.path.join(work_directory, report_folder_name)
if not os.path.exists(report_folder):
    os.mkdir(report_folder)
assessment_report_folder = os.path.join(report_folder, 'assessment')
if not os.path.exists(assessment_report_folder):
    os.mkdir(assessment_report_folder)

# ---------------------------
# Before execute this script
# ---------------------------
# 1. Please replace RAWDATA_TIMESPAN in data_configs.py file using the following value,
# and change it back to the previous version after running this script.
# RAWDATA_TIMESPAN = {
#     'SPEI': (None, None),  # Retrieve from datasets
#     'GDDP': ('1980-2019', ''), # -> Indian plants for validation, i.e., using ERA5 datastes
#     'ERA5': ('1980-2019', None),
#     'WTUU': ('1965-2010', '2030-2069'),
# }  # timespan that indicates on the file name

# 2. Next, you can run the script or execute each step one after another. If you wish to run step by step,
# please make sure not skipping any steps and run in sequence.


# --------------------------------------------------------------------------------------
# 1. Download ERA5 from Google Earth Engine
# After running this part, please
# 1) Download data from your google drive
# 2) Create a folder under data\raw\india and name the folder as era5, save data in era5
# --------------------------------------------------------------------------------------
tpp_fp = os.path.join(work_directory, r"data\ancillary\indian_plant_candidates_v1.xlsx")
era5_timespan = [1980, 2019]
tpp_era5_folder_name = 'era5'

from scripts.data import PL_gee_ebrd_tpp_era5_data_download

print('...Start to download ERA5 datasets from Google Earth Engine...')
PL_gee_ebrd_tpp_era5_data_download.main(tpp_fp=tpp_fp, imgcol_url="ECMWF/ERA5/DAILY", time_span=era5_timespan,
                                        calcScale=5000, start=0, save_output=True)

if not os.path.exists(os.path.join(india_data_folder, tpp_era5_folder_name)):
    os.mkdir(os.path.join(work_directory, tpp_era5_folder_name))
print(f'Please download era5 datasets from google drive and save it here '
      f'{os.path.join(work_directory, tpp_era5_folder_name)}')

# ----------------------------
# 2. Pre-process ERA5 datasets
# ----------------------------
era5_restructure_folder = os.path.join(work_directory, r'india_data\era5_restructure')
era5_wtbtemp_folder = os.path.join(work_directory, r'india_data\era5_wetbulbtemp')

from scripts import ear5

print('...Start to pre-process ERA5 datasets...')
ear5 = ear5.Ear5(work_directory=work_directory,
                 output_directory=os.path.join(work_directory, 'output_temp'),
                 data_folder=os.path.join(work_directory, r'india_data\era5'))
df_batch = ear5.restructure_batch(save_output=False, output_directory=era5_restructure_folder)
df_batch_wbtemp = ear5.cal_wbtemp_batch(df_batch=df_batch, save_output=True, output_directory=era5_wtbtemp_folder)

# ------------------------------------------
# 3. Prepare ERA5 for air temperature module
# ------------------------------------------
in_folder = os.path.join(work_directory, r'india_data\era5')
out_folder = os.path.join(work_directory, r'india_data\era5_4_airtemp')

from scripts import power_plant_physical_climate_risk_assessment as master

print('...Start to prepare ERA5 for air temperature module...')
tpr = master.TppPhyRisk()
tpr.export_restructured_era5_for_airtemp_module(in_folder=in_folder, out_folder=out_folder)

# ------------------------------------------------------------------------------------------------------------
# 4. Air temperature module
# Before executing this part, please change RAWDATA_TIMESPAN for 'GDDP' in data_configs.py file in data folder
# ------------------------------------------------------------------------------------------------------------
save_output = True  # specify if you'd like to save output locally or investigate results on the fly.
suffix_test = ''  # specify how you'd like to indicate results of the testing run.

vulnerability_factors_fp = os.path.join(work_directory, r'vulnerability\vulnerability_factors_20210409.xlsx')
vul_group_code = 3  # 'Sheet3' -> vulnerability factors table.
thd_group_code = 19  # reference code of the threshold settings.

scenario_id_list = []  # only inspect historical situations, i.e., no future scenarios.
module = ['air temp']  # only assess air temperature-induced generation losses.
air_bc_yrs = [2013, 2016]  # only analyse generation data from 2013 through 2016 given current data availability.
climate_data_folder = os.path.join(work_directory, r'india_data\era5_4_airtemp')  # output of step 3.
tpp_working_fp = os.path.join(work_directory, r'tpp info\tpp_working_india_CCPP.xlsx')  # plant information.

from scripts import power_plant_physical_climate_risk_assessment as master

print('...Start thermal assessment for indian plants...')
# Naming scheme (Please do NOT modify this part if you'd like to produce plots with this package).
suffix_scenarios = '{}{}'.format('rcp', '-'.join(scenario_id_list)) if scenario_id_list != [] else 'historical'
suffix_module = '-'.join([i.replace(' ', '-') for i in module])
suffix_date = datetime.now().strftime("%Y%m%d")
final_assessment_prefix = f'final-assessment_vul{str(vul_group_code)}_thd{str(thd_group_code)}' \
                          f'_{str(suffix_scenarios)}_{str(suffix_module)}_{str(suffix_date)}{str(suffix_test)}'
# print(f'\nFind the output here: {final_assessment_prefix}\n')

tpr = master.TppPhyRisk(module=module,
                        save_output=save_output,
                        final_assessment_prefix=final_assessment_prefix,
                        final_assessment_dir=os.path.join(work_directory, 'final assessment', 'india'),
                        vulnerability_factors_fp=vulnerability_factors_fp,
                        vul_group_code=vul_group_code,
                        thd_group_code=thd_group_code,
                        scenario_id_list=scenario_id_list,
                        air_bc_yrs=air_bc_yrs,
                        climate_data_folder=climate_data_folder,
                        tpp_working_fp=tpp_working_fp)
tpr.thermal_assess()

# -------------------------
# 5. Export visualizations
# -------------------------

print('...Start to generation figures...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'validation_with_indian_plants.py')
exec(open(fp).read())
