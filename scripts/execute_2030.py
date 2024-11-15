"""
Physical climate risk assessment for EBRD's thermal plant portfolio over the baseline period and the projection period 2030.

1. Download GDDP and ERA5 from Google Earth Engine.
2. Retrieve water temperature over the entire historical and projection time frame.
3. Restructure input datasets, i.e., water temperatures, air temperatures,...
4. Feature engineering, i.e.,
    1) Calculate wet-bulb temperatures from ERA5 datasets;
    2) Bias corrections for GDDP and ERA5;
    3) Model wet-bulb temperature as a function of air temperatures and precipitations from ERA5 datasets;
    4) Predict wet-bulb temperatures using air temperatures and precipitations from GDDP datasets.
5. Thermal assessment
    1) Default assessment;
    2) Assess the benefit of upgrading wet-cooling tech to air-cooling tech;
    3) Sensitivity analysis of regulatory discharge limits.
6. Plot portfolio-level waterfall chart.
7. Summary statistics of climatic variables (i.e., input datasets)
8. Export master table -> portfolio-level result.
9. Generate figures in the working paper,
    1) % reduced capacity by turbine-cool types
    2) Exceedance probability of design air and water temperatures.
    3) Time series of climatic variables.
"""

import numpy as np
import os
import itertools
from datetime import datetime

work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))  # work directory
date_code = datetime.now().strftime("%Y%m%d")
# date_code = '20210519' # For testing

# ################################################################################################
# PARAMETERS
# ################################################################################################
# User-defined parameters
exe_wta = True  # Whether run wet cool to air cool analysis
exe_sen = True  # whether run sensitivity analysis for regulatory discharge limits
# input datasets
daily_nc_file = r'D:\WRI\temp\waterTemperature_mergedV2.nc'  # daily water temperature datasets over the historical period.
weekly_folder = r'D:\WRI\Water Temperature'  # weekly water temperature datasets over the projection period.
flood_data_folder = r"D:\WRI\Floods\inland"
aq21base_fp = r'D:\WRI\GIS\baseline\aqueduct_global_maps_21_shp\aqueduct_global_dl_20150409.shp'
aq21futu_fp = r'D:\WRI\GIS\future\aqueduct_projections_20150309_shp\aqueduct_projections_20150309.shp'
spei_data_folder = r'D:\WRI\spei'
hydro_fp = os.path.join(work_directory, 'final assessment', 'external_result',
                        'portfolio_level_hydro_2030_gen_losses.xlsx')

# user-defined parameters in the thermal assessment.
# default ones
module = ['air temp', 'water temp', 'drought', 'flood', 'water stress']
tpp_working_fp = os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx')  # power plant info
vulnerability_factors_fp = os.path.join(work_directory, 'vulnerability',
                                        'vulnerability_factors_20210409.xlsx')  # vulnerability factor table
vul_group_code = 3
thd_id_noreg, thd_id_reg = 19, 21
thd_group_codes = [thd_id_noreg, thd_id_reg]
scenario_ids = [['45'], ['85']]
save_output = True
suffix_test = ''

# wet to air cool
wta_thd_code = 21  # reference id of vulnerability scenario for wet to air cool analysis
tpp_working_fp_2 = os.path.join(work_directory, 'tpp info',
                                'tpp_working_master_wet-to-air.xlsx')  # power plant info for analysing the benefit of upgrading wet to air cooling tech.
wta_output_suffix = '_wet-to-air'

# sensitivity analysis of regulatory limits
sen_module = ['water temp']
sen_thd_codes = range(21,
                      27)  # reference ids of vulnerability scenarios for the sensitivity analysis of regulatory limits
vulnerability_factors_fp_2 = os.path.join(work_directory, 'vulnerability',
                                          'vulnerability_factors_20210409_forSensitivityAnalysis.xlsx')  # vulnerability factor table for sensitivity analysis of regulatory limits

# User-adjustable parameters
# specify what time frame to be assessed
# this assessment was for the projection period 2010 (from 2010 through 2049)
flood_year = 2030
water_stress_year = 2030
air_bc_yrs = [1950, 2005]
rc_bc_yrs = [1980, 2010]
his_eval_yrs = [1965, 2004]
fut_eval_yrs = [2010,
                2049]  # projection period 2010, will also impact the time frame of the bias correction and wet-bulb temperature calculation.

# Constants -> default values are retrieved from input datasets,
# but can be adjusted accordingly for a subset of the default settings.
plant_id_list = range(1, 26)  # all plants
wbtemp_hist_timespan = (1980, 2005)  # time span concerned as baseline condition of wet-bulb temperatures from ERA5.
era5_timespan = (1980, 2019)
gddp_hist_timespan = (1950, 2005)  # setting for data downloading steps.
gddp_futu_timespan = (2006, 2070)  # setting for data downloading steps.
tpp_water_temp_folder_name = 'tpp water temp all'  # where to save plant-level water temperature time series retrieved from netCDF files.
tpp_watertemp_post_folder_name = 'watertemp_output_temp_all'
tpp_airtemp_folder_name = 'tpp_climate_gddp_all'
tpp_airtemp_exp_folder_name = 'new tpp climate corr'
tpp_airtemp_post_folder_name = 'tpp_climate_gddp_restructure_all_withAirTempAvg'
tpp_airtemp_biasCorrected_folder_name = 'tpp_climate_gddp_restructure_all_withAirTempAvg_biasCorrected'
tpp_gddp_wbtemp_folder_name = 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols'
tpp_era5_folder_name = 'era5'
tpp_wbtemp_era5_folder_name = 'era5_wetbulbtemp'
wbtemp_model_folder_name = 'wbtemp_model_nonorm_ols'
final_assessment_dir = os.path.join(work_directory, 'final assessment', 'interim')
wta_assessment_dir = os.path.join(final_assessment_dir, 'wet-to-air')
sen_assessment_dir = os.path.join(final_assessment_dir, 'sensitivity-analysis_regulatory-limits')
water_stress_folder = os.path.join(work_directory, 'data', 'processed', 'water_stress')
processed_assessment_dir = os.path.join(work_directory, 'final assessment', 'processed')
master_fn = 'master'
report_folder_name = 'reports'
report_folder = os.path.join(work_directory, report_folder_name)
if not os.path.exists(report_folder):
    os.mkdir(report_folder)
assessment_report_folder = os.path.join(report_folder, 'assessment')
if not os.path.exists(assessment_report_folder):
    os.mkdir(assessment_report_folder)
data_report_folder = os.path.join(report_folder, 'data')
if not os.path.exists(data_report_folder):
    os.mkdir(data_report_folder)

print(f'Start!\n')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Download GDDP
# =============

from scripts.data import PL_ebrd_gddp

print('...Start to download GDDP data...')
PL_ebrd_gddp.main(tpp_fp=tpp_working_fp, data_folder=os.path.join(work_directory, tpp_airtemp_folder_name),
                  imgcol_url="NASA/NEX-GDDP", hist_timespan=gddp_hist_timespan, futu_timespan=gddp_futu_timespan,
                  scenario_list=None, model_list=None, calcScale=5000, start=0, save_output=True)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Download ERA5 and save in google drive.
# Note: when finishing downloading, please transfer datasets from google drive to local disk.
# ============================================================================================

from scripts.data import PL_gee_ebrd_tpp_era5_data_download

print('...Start to download ERA5 data...')
PL_gee_ebrd_tpp_era5_data_download.main(tpp_fp=tpp_working_fp, imgcol_url="ECMWF/ERA5/DAILY", time_span=era5_timespan,
                                        calcScale=5000, start=0, save_output=True)

if not os.path.exists(os.path.join(work_directory, tpp_era5_folder_name)):
    os.mkdir(os.path.join(work_directory, tpp_era5_folder_name))
print(f'Please download era5 datasets from google drive and save it here '
      f'{os.path.join(work_directory, tpp_era5_folder_name)}')

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

from scripts import era5

print('\n...Start to calculate wet-bulb temperatures from ERA5 datasets...')
era5 = era5.Era5(work_directory=work_directory)
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

from scripts import era5

print(
    '\n...Start to model the relationship between wet-bulb temperatures and air temperatures and precipitations using ERA5 datasets...')
era5 = era5.Era5(work_directory=work_directory)
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

from scripts import era5

print('\n...Start to calculate wet-bulb temperatures as a function of air temperatures and precipitations from GDDP...')
era5 = era5.Era5(work_directory=work_directory,
                 output_directory=os.path.join(work_directory, tpp_gddp_wbtemp_folder_name))
era5.predict_wbtemp_batch(model_folder=os.path.join(work_directory, wbtemp_model_folder_name),
                          data_folder=os.path.join(work_directory, tpp_airtemp_biasCorrected_folder_name),
                          output_directory=era5.output_directory,
                          plant_id_list=plant_id_list,
                          hist_timespan_str='-'.join(map(str, wbtemp_hist_timespan)),
                          futu_timespan_str='-'.join(map(str, fut_eval_yrs)),
                          era5_timespan='-'.join(map(str, era5_timespan)))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Thermal assessment
#
# 1. Default assessment.
# 2. Assess the benefit of upgrading wet-cooling tech to air cooling tech.
# 3. Sensitivity analysis of regulatory discharge limits.
# ==================

from scripts import power_plant_physical_climate_risk_assessment


def master(thd_group_code, scenario_id_list, module=module,
           final_assessment_dir_user=final_assessment_dir, vulnerability_factors_fp_user=vulnerability_factors_fp,
           tpp_working_fp_user=tpp_working_fp):
    """
    :param thd_group_code: int, reference id of pre-defined thresholds listed in sheet 'thresholds' in the vulnerability factor table.
    :param scenario_id_list: list, ['45', '85'] or its subset.
    :param final_assessment_dir_user: folder path-like string, the default value is equal to the global variable final_assessment_dir.
    :param vulnerability_factors_fp_user: file path-like string, excel file, the default value is equal to the global variable vulnerability_factors_fp.
    :param tpp_working_fp_user: file path-like string, excel file, the default value is equal to the global variable tpp_working_fp.
    :return:
    """

    suffix_scenarios = '-'.join(scenario_id_list)
    suffix_module = '-'.join([i.replace(' ', '-') for i in module])
    suffix_date = datetime.now().strftime("%Y%m%d")
    final_assessment_prefix = f'final-assessment_vul{str(vul_group_code)}_thd{str(thd_group_code)}' \
                              f'_rcp{str(suffix_scenarios)}_{str(suffix_module)}_{str(suffix_date)}{suffix_test}'

    tpr = power_plant_physical_climate_risk_assessment.TppPhyRisk(
        project_folder=work_directory,
        module=module, save_output=save_output, final_assessment_prefix=final_assessment_prefix,
        final_assessment_dir=final_assessment_dir_user,
        aq21base_fp=aq21base_fp, aq21futu_fp=aq21futu_fp,
        flood_data_folder=flood_data_folder, spei_data_folder=spei_data_folder,
        wbtemp_folder=os.path.join(work_directory, tpp_wbtemp_era5_folder_name),
        climate_data_folder=os.path.join(work_directory, tpp_airtemp_exp_folder_name),
        tpp_water_temp_folder=os.path.join(work_directory, tpp_water_temp_folder_name), vul_group_code=vul_group_code,
        gddp_recal_folder=os.path.join(work_directory, tpp_gddp_wbtemp_folder_name),
        watertemp_restructure_all=os.path.join(work_directory, tpp_watertemp_post_folder_name),
        vulnerability_factors_fp=vulnerability_factors_fp_user, tpp_working_fp=tpp_working_fp_user,
        thd_group_code=thd_group_code, scenario_id_list=scenario_id_list, air_bc_yrs=air_bc_yrs, rc_bc_yrs=rc_bc_yrs,
        his_eval_yrs=his_eval_yrs, fut_eval_yrs=fut_eval_yrs, flood_year=flood_year,
        water_stress_year=water_stress_year, plant_id_list=plant_id_list)
    tpr.thermal_assess()


# Default
print(f'\n...Start thermal assessment...')
list([master(thd_group_code=thd_group_code, scenario_id_list=scenario_id_list) for thd_group_code, scenario_id_list
      in list(itertools.product(thd_group_codes, scenario_ids))])

# Wet cool to air cool
if exe_wta is True:
    print(f'\n...Start thermal assessment (i.e., wet to air cool)...')
    list([master(scenario_id_list=scenario_id_list, thd_group_code=wta_thd_code,
                 final_assessment_dir_user=wta_assessment_dir, tpp_working_fp_user=tpp_working_fp_2)
          for scenario_id_list in scenario_ids])

# Sensitivity analysis for regulatory limits
if exe_sen is True:
    print(f'\n...Start thermal assessment (i.e., sensitivity analysis of regulatory limits)...')
    list([master(scenario_id_list=scenario_id_list, thd_group_code=thd_group_code, module=sen_module,
                 final_assessment_dir_user=sen_assessment_dir,
                 vulnerability_factors_fp_user=vulnerability_factors_fp_2)
          for scenario_id_list, thd_group_code in itertools.product(scenario_ids, sen_thd_codes)])

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Post-processing
# ===============

from scripts.report import report_stats

pr = report_stats.PrepReport(work_directory=work_directory,
                             processed_folder=processed_assessment_dir.replace(work_directory + '\\', ''))

# Default
print(f'\n...Start post-processing...')
pr.interim_folder = final_assessment_dir.replace(work_directory + '\\', '')
pr.merge_batch(vul_id_list=[vul_group_code], thd_id_list=thd_group_codes,
               scenario_id_list=np.array(scenario_ids).flatten().tolist(), module_list=['*'],
               date_list=[date_code], groupby_list=['ByPlant'], run_code_list=['*'],
               output_fn='final-assessment-merge_{}{}'.format('-'.join([date_code]), suffix_test))

# Wet cool to air cool
if exe_wta is True:
    print(f'\n...Start post-processing (i.e., wet to air cool)...')
    pr.interim_folder = wta_assessment_dir.replace(work_directory + '\\', '')
    pr.merge_batch(vul_id_list=[vul_group_code], thd_id_list=[wta_thd_code],
                   scenario_id_list=np.array(scenario_ids).flatten().tolist(), module_list=['*'],
                   date_list=[date_code], groupby_list=['ByPlant'], run_code_list=['*'],
                   output_fn='final-assessment-merge_{}{}'.format('-'.join([date_code]), wta_output_suffix))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot portfolio-level waterfall chart
# ====================================

from scripts.visualizations import waterfall

# Default
print(f'\n...Start waterfall visualization...')
waterfall.main(work_directory=work_directory,
               in_fp=os.path.join(processed_assessment_dir, f'final-assessment-merge_{date_code}_ByPlant.xlsx'),
               thd_id_noreg=thd_id_noreg, thd_id_reg=thd_id_reg, fn_suffix=suffix_test,
               hydro_fp=hydro_fp, save_fig=True)

# Wet cool to air cool
if exe_wta is True:
    print(f'\n...Start waterfall visualization... (i.e., wet cool to air cool)')
    waterfall.main(work_directory=work_directory,
                   in_fp=os.path.join(processed_assessment_dir,
                                      f'final-assessment-merge_{date_code}{wta_output_suffix}_ByPlant.xlsx'),
                   thd_id_noreg=None, thd_id_reg=thd_id_reg, fn_suffix=wta_output_suffix,
                   hydro_fp=hydro_fp, save_fig=True)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Climatic trajectories -> summary statistics of climatic variables (i.e., input datasets)
# ========================================================================================

from scripts.report import data_stats

data_folder = os.path.join(work_directory, 'reports', 'data')
if fut_eval_yrs == [2030, 2069] or fut_eval_yrs == (2030, 2069):
    year_suffix = ''
    water_stress_year_suffix = ''
elif fut_eval_yrs == [2010, 2049] or fut_eval_yrs == (2010, 2049):
    year_suffix = f'_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}'
    water_stress_year_suffix = f'_{water_stress_year}'
output_fn_dict = {
    'gddp_wbt': f'wet-bulb-temperature_gddp_{wbt_gddp_bc_years[0]}-{wbt_gddp_bc_years[1]}_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
    'spei_neg2': f'spei-neg2_spei_{his_eval_yrs[0]}-{his_eval_yrs[1]}_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
    'spei': f'spei_spei_{his_eval_yrs[0]}-{his_eval_yrs[1]}_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
    'uuwt': f'water-temperature_uuwt_{his_eval_yrs[0]}-{his_eval_yrs[1]}_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
    'gddp': f'air-temperature_gddp_{his_eval_yrs[0]}-{his_eval_yrs[1]}_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
    'era5': f'wet-bulb-temperature_era5_{rc_bc_yrs[0]}-{rc_bc_yrs[1]}',
}

# Summary statistics
print('...Start to calculate summary statistics of climate variables...')
for name in ['gddp_wbt', 'spei_neg2', 'uuwt', 'gddp', 'era5', 'spei']:
    insta = data_stats.InputStats(work_directory=work_directory, name=name, output_fn=output_fn_dict.get(name),
                                  historical_years=his_eval_yrs, projection_years=fut_eval_yrs, year_suffix=year_suffix,
                                  water_stress_year_suffix=water_stress_year_suffix)
    summary_stats = insta.main(output_fn=output_fn_dict.get(name))
print(f'\nFind output here: {data_folder}')

# Calculate changes of median
print('...Start to calculate changes of median...')
wt_fp = os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get('uuwt')))
at_fp = os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get('gddp')))
dt_fp = os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get('spei_neg2')))
wbt_hist_fp = os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get('era5')))
wbt_futu_fp = os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get('gddp_wbt')))
ws_fp = os.path.join(water_stress_folder,
                     f'water-stress_srr{water_stress_year_suffix}.csv')
insta = data_stats.InputStats(name=None, year_suffix=year_suffix, water_stress_year_suffix=water_stress_year_suffix)
insta.var_chg_batch(wt_fp=wt_fp, at_fp=at_fp, dt_fp=dt_fp, ws_fp=ws_fp, wbt_futu_fp=wbt_futu_fp,
                    wbt_hist_fp=wbt_hist_fp)
print(f'Find output here: {data_folder}')

# Evaluate how thresholds retrieved from historical datasets change in the future period
print('\n...Start to evaluate how thresholds retrieved from historical datasets change in the future period...')
for name in ['uuwt', 'gddp']:
    insta = data_stats.InputStats(work_directory=work_directory, name=name, output_fn=output_fn_dict.get(name),
                                  historical_years=his_eval_yrs, projection_years=fut_eval_yrs, year_suffix=year_suffix,
                                  water_stress_year_suffix=water_stress_year_suffix)
    output = insta.find_percentile_change(input_fp=os.path.join(data_folder, '{}.csv'.format(output_fn_dict.get(name))),
                                          save_output=True, thd_val=None)
print(f'Find output here: {data_folder}')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Portfolio-level results in excel table
# ======================================

from scripts.report import master_report

# Default
print(f'\n...Start to create master table...')
mr = master_report.MasterReport(in_fn=f'final-assessment-merge_{date_code}_ByPlant.xlsx', out_fn=f'{master_fn}',
                                work_directory=work_directory)
mr.generate_master_report(year_suffix=f'_{fut_eval_yrs[0]}-{fut_eval_yrs[1]}',
                          year_suffix_water_stress=f'_{water_stress_year}')


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Plot figures for the working paper
# ==================================

# Visualize % reduced capacity by turbine-cool types as boxplots.
print('...Start to generate boxplots of % reduced capacity by turbine-cool types...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'boxplots.py')
exec(open(fp).read())

# Visualize exceedance probability of design air and water temperatures.
print('...Start to visualize exceedance probability of design air and water temperatures...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'probability_changes.py')
exec(open(fp).read())

# Visualize time series of climatic variables.
print('...Start to visualize time series of climatic variables...')
fp = os.path.join(work_directory, 'scripts', 'visualizations', 'climate_trajectories.py')
exec(open(fp).read())
