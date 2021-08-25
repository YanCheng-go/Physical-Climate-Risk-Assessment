"""
Download ERA5 and GDDP for all plants.
"""

import os
work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Parameters or Constants.
# ========================
tpp_working_fp = os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx')
tpp_airtemp_folder_name = 'tpp_climate_gddp_all'
tpp_era5_folder_name = 'era5'
gddp_hist_timespan = (1950, 2005)
gddp_futu_timespan = (2006, 2070)
era5_timespan = (1980, 2019)

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
