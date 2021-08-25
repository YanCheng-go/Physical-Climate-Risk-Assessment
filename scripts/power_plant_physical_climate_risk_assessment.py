"""
Master script of the assessment of physical climate risk, including floods, droughts, air temperature,
water temperature, and water stress.

Author: Tianyi Luo, Yan Cheng
"""

import bisect
import calendar
import gc
import itertools
import os
import glob
import sys
from datetime import datetime
from datetime import timedelta, date
from itertools import repeat

import geopandas as gp
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import tqdm
# Visualization
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from termcolor import colored
import seaborn as sns

from scripts.data import data_configs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class TppPhyRisk:
    """
    Physical climate risk assessment for thermal plants.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: dictionary,
        :param save_output: boolean, save output (i.e., final assessment) locally or not.
        :param module: list, a subset of ['drought', 'air temp', 'water temp', 'flood', 'water stress'].
        :param final_assessment_prefix: string, prefix string of the final assessment output.
        :param flood_data_folder: path-like string, the folder path of flood data, which is the input of the flood module.
        :param daily_nc_file: path-like string, the file path of daily water temperature netCDF files.
        :param weekly_folder: path-like string, the folder path of weekly water temperature netCDF files.
        :param project_folder: path-like string, work directory, i.e., the root folder of this project.
        :param data_folder:
        :param temp_output_folder: path-like string, the folder for temporary output
        :param vulnerability_factors_fp: path-like string, the file path of vulnerability factors
        :param spei_data_folder: path-like string, the folder path of SPEI datasets, which is the input of the drought module.
        :param climate_data_folder: path-like string, the folder path of climate datasets, pre-processed plant-level GDDP datasets. The input datasets of the air temperature module.
        :param wbtemp_folder: path-like string, the folder path of pre-calculated wet-bulb temperature datasets using ERA5 datasets, which is the input of water temperature module for recirculating-cooling plants, i.e., rc_module().
        :param gddp_recal_folder: path-like string, the folder path of restructured GDDP datasets with modeled wet-bulb temperature, which is the input of the water temperature module for recirculating-cooling plants, i.e., rc_module().
        :param tpp_water_temp_folder: path-like string, the folder path of plant-level water temperature, which is the input of the water temperature module for once-through-cooling plants, i.e., water_temperature_module().
        :param temp_power_plant_info_data_folder: path-like string, the folder path of power plant info files.
        :param tpp_working_fp: path-like string, the file path of non-spatial info of power plant.
        :param tpp_locs_fp: path-like string, the file path of spatial info of power plant.
        :param aq21base_fp: path-like string, the file path of baseline water stress data.
        :param aq21futu_fp: path-like string, the file path of projected ater stress data.
        :param his_eval_yrs: tuple or list, the start and end year of the historical period.
        :param fut_eval_yrs: tuple or list, the start and end year of the projection period.
        :param run_code: string, an unique reference code to differentiate different runs.
        :param vul_group_code: int, the reference id of a set of vulnerability factors. 3 refers to "Sheet3" in the vulnerability factor database (i.e., an excel workbook), the file path of which is assigned to vulnerability_factors_fp.
        :param thd_group_code: int, the reference id of a set of thresholds in "thresholds" sheet in the vulnerability factor database, the file path of which is assigned to vulnerability_factors_fp.
        :param scenario_id_list: list, ['45', '85'] or a subset of it or []. '45' refers to RCP4.5 and '85' refers to RCP8.5. If you assign [] to this variable, no future scenarios will be assessed. In other words, only historical conditions will be assessed. This has only been tested
        :param water_module_error_fp: path-like string, statistics used to correct bias caused by inconsistency in temporal resolution,i.e., daily generation losses calculated from weekly observations.
        :param flood_year: int, 2010, 2030, or 2050, indicating the period of future projection.
        :param air_bc_yrs: tuple or list, start year and end year of backcast air temperatures in air temperature module (i.e., air_temperature_module()) .
        :param rc_bc_yrs: tuple or list, start year and end year of historical data in the recirculating module (i.e., rc_module()).
        :param water_stress_year: int, period of future projection of Aqueduct water stress data, which can be 2020, 2030, or 2040...
        :param final_assessment_dir: path-like string, folder path-like string, the folder to save final assessments.
        """
        kwargs = kwargs

        def kwargs_generator(kw, def_val, typ=None):
            if kw not in kwargs or kwargs[kw] is None:
                kwargs[kw] = def_val
            if typ == 'ls':
                if not isinstance(kwargs[kw], list):
                    kwargs[kw] = [kwargs[kw]]
            return kwargs[kw]

        # User-defined parameters
        self.flood_data_folder = kwargs_generator('flood_data_folder', r"D:\WRI\Floods\inland")
        self.daily_nc_file = kwargs_generator('daily_nc_file', r'D:\WRI\temp\waterTemperature_mergedV2.nc')
        self.weekly_folder = kwargs_generator('weekly_folder', r'D:\WRI\Water Temperature')
        self.aq21base_fp = kwargs_generator('aq21base_fp',
                                            r'D:\WRI\GIS\baseline\aqueduct_global_maps_21_shp\aqueduct_global_dl_20150409.shp')
        self.aq21futu_fp = kwargs_generator('aq21futu_fp',
                                            r'D:\WRI\GIS\future\aqueduct_projections_20150309_shp\aqueduct_projections_20150309.shp')
        self.spei_data_folder = kwargs_generator('spei_data_folder', r'D:\WRI\spei')

        # Please do not modify the following variables
        # project folder and temporary folder
        self.project_folder = kwargs_generator('project_folder', os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
        self.data_folder = kwargs_generator('data_folder', os.path.join(self.project_folder, 'data'))
        if not os.path.exists(os.path.join(self.data_folder, 'processed')):
            os.mkdir(os.path.join(self.data_folder, 'processed'))
        self.temp_output_folder = kwargs_generator('temp_output_folder',
                                                   os.path.join(self.project_folder, 'output_temp'))
        if not os.path.exists(self.temp_output_folder):
            os.mkdir(self.temp_output_folder)
        final_assessment_root_folder = os.path.join(self.project_folder, 'final assessment')
        if not os.path.exists(final_assessment_root_folder):
            os.mkdir(final_assessment_root_folder)
        self.final_assessment_dir = kwargs_generator('final_assessment_dir', final_assessment_root_folder)
        if not os.path.exists(self.final_assessment_dir):
            os.mkdir(self.final_assessment_dir)
        # ancillary data path (user-adjustable)
        self.temp_power_plant_info_data_folder = kwargs_generator('temp_power_plant_info_data_folder',
                                                                  os.path.join(self.project_folder, 'tpp info'))
        self.tpp_working_fp = kwargs_generator('tpp_working_fp',
                                               os.path.join(self.temp_power_plant_info_data_folder, 'tpp_working.xlsx'))
        self.tpp_locs_fp = kwargs_generator('tpp_locs_fp',
                                            os.path.join(self.temp_power_plant_info_data_folder, 'tpp_locs.xlsx'))
        self.vulnerability_factors_fp = kwargs_generator('vulnerability_factors_fp', os.path.join(self.project_folder,
                                                                                                  r'vulnerability\vulnerability_factors_20210409.xlsx'))
        # climatic data path
        self.climate_data_folder = kwargs_generator('climate_data_folder',
                                                    os.path.join(self.project_folder, 'new tpp climate corr'))
        self.wbtemp_folder = kwargs_generator('wbtemp_folder', os.path.join(self.project_folder, 'era5_wetbulbtemp'))
        self.gddp_recal_folder = kwargs_generator('gddp_recal_folder',
                                                  os.path.join(self.project_folder, 'tpp_climate_gddp_withWetBulbTemp'))
        self.tpp_water_temp_folder = kwargs_generator('tpp_water_temp_folder',
                                                      os.path.join(self.project_folder, 'tpp water temp'))
        self.watertemp_restructrue_all = kwargs_generator('watertemp_restructrue_all',
                                                          os.path.join(self.project_folder,
                                                                       'watertemp_output_temp_all'))
        self.water_stress_folder = os.path.join(self.project_folder, 'data\\processed\\water_stress')
        if not os.path.exists(os.path.join(self.project_folder, 'data')):
            os.mkdir(os.path.join(self.project_folder, 'data'))
        if not os.path.exists(os.path.join(self.project_folder, 'data', 'processed')):
            os.mkdir(os.path.join(self.project_folder, 'data', 'processed'))
        if not os.path.exists(self.water_stress_folder):
            os.mkdir(self.water_stress_folder)
        # module parameters (user-adjustable)
        self.save_output = kwargs_generator('save_output', True)
        self.final_assessment_prefix = kwargs_generator('final_assessment_prefix', 'final_assessment')
        self.module = kwargs_generator('module', ['drought', 'air temp', 'water temp', 'flood', 'water stress'], 'ls')
        self.vul_group_code = kwargs_generator('vul_group_code', 1)
        self.thd_group_code = kwargs_generator('thd_group_code', 1)
        self.scenario_id_list = kwargs_generator('scenario_id_list', ['45', '85'], 'ls')
        # temporal parameters (user-adjustable)
        self.his_eval_yrs = kwargs_generator('his_eval_yrs', [1965, 2004])
        self.fut_eval_yrs = kwargs_generator('fut_eval_yrs', [2030, 2069])
        self.flood_year = kwargs_generator('flood_year', 2050)
        self.air_bc_yrs = kwargs_generator('air_bc_yrs', [1950, 2005])
        self.rc_bc_yrs = kwargs_generator('rc_bc_yrs', [1980, 2010])
        self.water_stress_year = kwargs_generator('water_stress_year', 2040)
        # spatial parameters (user-adjustable)
        self.plant_id_list = kwargs_generator('plant_id_list', 'all', 'ls')
        # miscellaneous
        self.run_code = kwargs_generator('run_code', int(
            datetime.now().strftime("%Y%m%d%H%M%S")))  # record the time when execution starts
        self.water_module_error_fp = kwargs_generator('water_module_error',
                                                      os.path.join(self.project_folder, 'vulnerability',
                                                                   'error_weekly_vs_daily.xlsx'))
        # parameters for testing
        self.temp_var = None
        self.test_out = []


    def ncDump(self, fp, verb=True):
        nc_fid = netCDF4.Dataset(fp, 'r')
        """
        ncdump outputs dimensions, variables and their attribute information.
        The information is similar to that of NCAR's ncdump utility.
        ncdump requires a valid instance of Dataset.
    
        Parameters
        ----------
        nc_fid : netCDF4.Dataset
            A netCDF4 dateset object
        verb : Boolean
            whether or not nc_attrs, nc_dims, and nc_vars are printed
    
        Returns
        -------
        nc_attrs : list
            A Python list of the NetCDF file global attributes
        nc_dims : list
            A Python list of the NetCDF file dimensions
        nc_vars : list
            A Python list of the NetCDF file variables
        """

        def print_ncattr(key):
            """
            Prints the NetCDF file attributes for a given key

            Parameters
            ----------
            key : unicode
                a valid netCDF4.Dataset.variables key
            """
            try:
                print("\t\ttype:", repr(nc_fid.variables[key].dtype))
                for ncattr in nc_fid.variables[key].ncattrs():
                    print('\t\t%s:' % ncattr,
                          repr(nc_fid.variables[key].getncattr(ncattr)))
            except KeyError:
                print("\t\tWARNING: %s does not contain variable attributes" % key)

        # NetCDF global attributes
        nc_attrs = nc_fid.ncattrs()
        if verb:
            print("NetCDF Global Attributes:")
            for nc_attr in nc_attrs:
                print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
        nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
        # Dimension shape information.
        if verb:
            print("NetCDF dimension information:")
            for dim in nc_dims:
                print("\tName:", dim)
                print("\t\tsize:", len(nc_fid.dimensions[dim]))
                print_ncattr(dim)
        # Variable information.
        nc_vars = [var for var in nc_fid.variables]  # list of nc variables
        if verb:
            print("NetCDF variable information:")
            for var in nc_vars:
                if var not in nc_dims:
                    print('\tName:', var)
                    print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                    print("\t\tsize:", nc_fid.variables[var].size)
                    print_ncattr(var)
        nc_fid.close()
        return nc_attrs, nc_dims, nc_vars

    def getNC4DataByLocation(self, fp, lat, lon, ncvar, nc_dims, buffer=None, is_time_series=False,
                             start_time_index=None, end_time_index=None):
        """
        Extract data series from netCDF files by location.

        :param fp: file path-like string, file path of the netCDF file.
        :param lat: float, latitude
        :param lon: float, longitude
        :param ncvar: string, one of the variables of the netCDF file.
        :param nc_dims: string, netCDF dimensions.
        :param buffer: look for pixels within a radius from the given coordinate.
        :param is_time_series: Boolean, specify whether it is a time series.
        :param start_time_index: int, index of start time.
        :param end_time_index: int, index of end time.
        :return: list
        """
        nc = netCDF4.Dataset(fp, 'r')
        latLabel = [l for l in nc_dims if 'lat' in l][0]
        lonLabel = [l for l in nc_dims if 'lon' in l][0]
        lats = nc.variables[latLabel][:]
        lons = nc.variables[lonLabel][:]
        if start_time_index == None:
            if buffer != None:
                ds = []
                latU, latD, lonL, lonR = lat + buffer, lat - buffer, lon - buffer, lon + buffer
                latlons = [[lat, lon], [latU, lonL], [latU, lonR], [latD, lonL], [latD, lonR]]
                for ll in latlons:
                    x = ll[0]
                    y = ll[1]
                    i = np.abs(lats - x).argmin()
                    j = np.abs(lons - y).argmin()
                    d = nc.variables[ncvar][:, i, j]
                    ds.append(d)
                if is_time_series:
                    value = ds
                else:
                    value = np.max(ds)
            else:
                x = lat
                y = lon
                i = np.abs(lats - x).argmin()
                j = np.abs(lons - y).argmin()
                d = nc.variables[ncvar][:, i, j]
                if is_time_series:
                    value = d
                else:
                    value = d[0]
        else:
            if buffer != None:
                ds = []
                latU, latD, lonL, lonR = lat + buffer, lat - buffer, lon - buffer, lon + buffer
                latlons = [[lat, lon], [latU, lonL], [latU, lonR], [latD, lonL], [latD, lonR]]
                for ll in latlons:
                    x = ll[0]
                    y = ll[1]
                    i = np.abs(lats - x).argmin()
                    j = np.abs(lons - y).argmin()
                    d = nc.variables[ncvar][start_time_index:end_time_index, i, j]
                    ds.append(d)
                # print(nc.variables['time'][start_time_index:end_time_index])
                if is_time_series:
                    value = ds
                else:
                    value = np.max(ds)
            else:
                x = lat
                y = lon
                i = np.abs(lats - x).argmin()
                j = np.abs(lons - y).argmin()
                d = nc.variables[ncvar][start_time_index:end_time_index, i, j]
                # days = nc.variables['time'][start_time_index:end_time_index]
                if is_time_series:
                    value = d
                else:
                    value = d[0]
        nc.close()
        gc.collect()
        return value

    def spei_get_date(self, row):
        y = int(row['year'])
        m = int(row['month'])
        d = datetime(y, m, 1)
        return d

    def spei_explorer(self, obs_fp, fut_fp, his_eval_yrs, fut_eval_yrs):
        """
        Extract spei time series from the database.

        :param obs_fp: file path-like string, file path of spei observations.
        :param fut_fp: file path-like string, file path of future spei data.
        :param his_eval_yrs: tuple or list, start year and end year of the historical period.
        :param fut_eval_yrs: tuple or list, start year and end year of the future period.
        :return: list, spei time series.
        """
        df_obs = pd.read_csv(obs_fp)
        df_obs['date'] = df_obs.apply(lambda row: self.spei_get_date(row), axis=1)
        df_fut = pd.read_csv(fut_fp)
        df_fut['date'] = df_fut.apply(lambda row: self.spei_get_date(row), axis=1)
        data_obs = df_obs[(df_obs['year'] >= his_eval_yrs[0]) & (df_obs['year'] <= his_eval_yrs[1])]
        df_hbc = df_fut[df_fut['simulation_scenario'] == 'historical']
        data_hbc = df_hbc[(df_hbc['year'] >= his_eval_yrs[0]) &
                          (df_hbc['year'] <= his_eval_yrs[1])]
        df_rcp45 = df_fut[df_fut['simulation_scenario'] == 'rcp45']
        data_rcp45 = df_rcp45[(df_rcp45['year'] >= fut_eval_yrs[0]) &
                              (df_rcp45['year'] <= fut_eval_yrs[1])]
        df_rcp85 = df_fut[df_fut['simulation_scenario'] == 'rcp85']
        data_rcp85 = df_rcp85[(df_rcp85['year'] >= fut_eval_yrs[0]) &
                              (df_rcp85['year'] <= fut_eval_yrs[1])]
        fut_mdls = df_fut['model'].unique()
        print(fut_mdls)

        # # MULTI-MODEL VIOLIN PLOTS
        # fig = plt.figure(figsize=(14, 8))
        # gs = fig.add_gridspec(1, 1)
        # ax1 = fig.add_subplot(gs[0,0])
        # data_ls = [data_obs['spei'].values]
        # data_ls1 = []
        # for m in fut_mdls:
        #     dTemp = data_hbc[data_hbc['model'] == m]['spei_12'].values
        #     data_ls1.append(dTemp)
        # data_ls2 = []
        # for m in fut_mdls:
        #     dTemp = data_rcp45[data_rcp45['model'] == m]['spei_12'].values
        #     data_ls2.append(dTemp)
        # for d1, d2 in zip(data_ls1, data_ls2):
        #     data_ls.append(d1)
        #     data_ls.append(d2)
        # r1 = ax1.violinplot(data_ls, showmeans=True, showmedians=True)
        # r1['cmeans'].set_color('black')
        # ax1.grid(b=True, axis='y')
        # xlabels = ['Obs.']
        # for m in fut_mdls:
        #     xlabels.append("%s-bc." % m)
        #     xlabels.append("%s-r45." % m)
        # plt.setp(ax1, xticks=np.arange(1, 16, 1),
        #          xticklabels=xlabels)
        # ax1.title.set_text('None')

        # SINGLE MODEL TIME SERIES & VIOLIN PLOTS
        models_available = ['CCSM4', 'CNRM-CM5', 'GFDL-ESM2M', 'HadGEM2-ES', 'INM-CM4', 'MPI-ESM-LR', 'MRI-CGCM3']
        model = models_available[6]
        dfTemp = df_fut[df_fut['model'] == model]
        dfTemp45 = dfTemp[(dfTemp['simulation_scenario'] == 'rcp45') | (dfTemp['simulation_scenario'] == 'historical')]
        dfTemp85 = dfTemp[(dfTemp['simulation_scenario'] == 'rcp85') | (dfTemp['simulation_scenario'] == 'historical')]
        fig = plt.figure(figsize=(14, 3))
        gs = fig.add_gridspec(1, 5)
        ax1 = fig.add_subplot(gs[0, :3])
        ax1.plot(dfTemp45['date'], dfTemp45['spei_12'], alpha=0.9)
        # # future vs future
        # ax1.plot(dfTemp85['date'], dfTemp85['spei_12'], alpha=0.9)
        # ax1.title.set_text('%s: SPEI backcast & projection (rcp4p5 & 8p5)' % model)
        # ax1.set_ylim([-3, 3])
        # obs. vs backcasted
        ax1.plot(df_obs['date'], df_obs['spei'], alpha=0.9)
        ax1.title.set_text('%s: SPEI backcast & observation' % model)
        ax1.set_ylim([-3, 3])
        ax1.set_xlim([datetime(his_eval_yrs[0], 1, 1), datetime(his_eval_yrs[1], 1, 1)])
        ax2 = fig.add_subplot(gs[0, 3:4])
        histData = dfTemp45[(dfTemp45['year'] >= his_eval_yrs[0]) &
                            (dfTemp45['year'] <= his_eval_yrs[1])]['spei_12'].values
        futuData = dfTemp45[(dfTemp45['year'] >= fut_eval_yrs[0]) &
                            (dfTemp45['year'] <= fut_eval_yrs[1])]['spei_12'].values
        data_ls = [histData, futuData]
        r2 = ax2.violinplot(data_ls, showmeans=True, showmedians=True)
        r2['cmeans'].set_color('black')
        ax2.grid(b=True, axis='y')
        hDate = str(his_eval_yrs[0]) + '-' + str(his_eval_yrs[1])
        fDate = str(fut_eval_yrs[0]) + '-' + str(fut_eval_yrs[1])
        print(hDate, fDate)
        plt.setp(ax2, xticks=np.arange(1  # this gotta be 1
                                       , len(data_ls) + 1, 1), xticklabels=[hDate, fDate])
        ax2.title.set_text('rcp4p5')
        ax2.set_ylim([-3, 3])
        ax3 = fig.add_subplot(gs[0, 4:5])
        histData = dfTemp85[(dfTemp85['year'] >= his_eval_yrs[0]) &
                            (dfTemp85['year'] <= his_eval_yrs[1])]['spei_12'].values
        futuData = dfTemp85[(dfTemp85['year'] >= fut_eval_yrs[0]) &
                            (dfTemp85['year'] <= fut_eval_yrs[1])]['spei_12'].values
        data_ls = [histData, futuData]
        r3 = ax3.violinplot(data_ls, showmeans=True, showmedians=True)
        r3['cmeans'].set_color('black')
        ax3.grid(b=True, axis='y')
        plt.setp(ax3, xticks=np.arange(1  # this gotta be 1
                                       , len(data_ls) + 1, 1), xticklabels=[hDate, fDate])
        ax3.title.set_text('rcp8p5')
        ax3.set_ylim([-3, 3])

        # SHOW GRAPHS IN TIGHT LAYOUT
        plt.tight_layout()
        plt.show()

    def flood_aed_base_futu(self, baseline_data, design_protection, projected_data, vulnerability_dataframe):
        """
        Extract time series from the flood database.

        :param baseline_data: list, data over the baseline period
        :param design_protection: float, e.g.,  250, 1000
        :param projected_data: list, data over the projection period.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors.
        :return: future_protection, baseExpectedDamage, futuExpectedDamage
        """
        # print(baseline_data)
        # print(projected_data)
        if design_protection >= 1000.0:
            design_protection = 999.9999999
        elif design_protection <= 2.0:
            design_protection = 2.0000001
        if (np.max(baseline_data) == 0.0 and np.max(projected_data) == 0.0):
            future_protection = design_protection
            baseExpectedDamage = 0.0
            futuExpectedDamage = 0.0
            # print(design_protection, future_protection, baseExpectedDamage, futuExpectedDamage)
            return future_protection, baseExpectedDamage, futuExpectedDamage
        else:
            ps = np.flip(np.array([1 / 2.0, 1 / 5.0, 1 / 10.0, 1 / 25.0, 1 / 50.0,
                                   1 / 100.0, 1 / 250.0, 1 / 500.0, 1 / 1000.0]))
            bd, dp, pd = np.flip(np.array(baseline_data)), 1 / float(design_protection), np.flip(
                np.array(projected_data))
            dpi = bisect.bisect_right(ps, dp)
            dpiL, dpiR = dpi - 1, dpi
            x1, x2, y1, y2 = ps[dpiL], ps[dpiR], bd[dpiL], bd[dpiR]
            baseInunLv = ((dp - x1) / (x2 - x1)) * (y2 - y1) + y1
            # print(vl, baseVul)
            unprotInunLvUppers = np.append(bd[:dpi], baseInunLv)
            unprotVulUppers = [self.find_vulnerability_threshold_asending(df=vulnerability_dataframe,
                                                                          cut_off_threshold=0.0,
                                                                          threshold_of_interest=inun)
                               for inun in unprotInunLvUppers]
            unprotVulLowers = np.append(unprotVulUppers[1:], 0.0)
            unprotRPUppers = np.append(ps[:dpi], dp)
            unprotRPLowers = np.append(unprotRPUppers[1:], unprotRPUppers[-1])
            baseExpectedDamage = np.sum((unprotVulUppers + unprotVulLowers) * (unprotRPLowers - unprotRPUppers) / 2.0)
            # print('#######################', unprotRPUppers, '\n', unprotVulUppers, '\n', unprotInunLvUppers)
            # print('#######################',unprotVulLowers, '\n', unprotRPLowers)
            # print('#######################','\n' ,unprotInunLvUppers, '\n', baseInunLv)
            if np.max(pd) == 0.0:
                future_protection = 1000.0
                futuExpectedDamage = 0.0
                # testprint = '%s_%s_%s_%s' % (design_protection, future_protection, baseExpectedDamage, futuExpectedDamage)
                # print(colored(testprint), 'red')
                return future_protection, baseExpectedDamage, futuExpectedDamage
            else:
                ps, pd = np.flip(ps), np.flip(pd)
                pdmax, pdmin = np.max(pd), np.min(pd)
                if baseInunLv >= pdmax:
                    baseInunLv = pdmax - 0.001
                elif baseInunLv <= pdmin:
                    baseInunLv = pdmin + 0.001
                pdi = bisect.bisect_right(pd, baseInunLv)
                pdiL, pdiR = pdi - 1, pdi
                # print(baseInunLv, pd, pdmax, pdmin)
                x1, x2, y1, y2 = ps[pdiL], ps[pdiR], pd[pdiL], pd[pdiR]
                futuProb = ((baseInunLv - y1) / (y2 - y1)) * (x2 - x1) + x1
                future_protection = np.ceil(1 / futuProb)
                unprotFutuInunLvUppers = np.append(np.flip(pd[pdi:]), baseInunLv)
                unprotFutuVulUppers = [self.find_vulnerability_threshold_asending(df=vulnerability_dataframe,
                                                                                  cut_off_threshold=0.0,
                                                                                  threshold_of_interest=inun)
                                       for inun in unprotFutuInunLvUppers]
                unprotFutuVulLowers = np.append(unprotFutuVulUppers[1:], 0.0)
                unprotFutuRPUppers = np.append(np.flip(ps[pdi:]), futuProb)
                unprotFutuRPLowers = np.append(unprotFutuRPUppers[1:], unprotRPUppers[-1])
                futuExpectedDamage = np.sum((unprotFutuVulUppers + unprotFutuVulLowers) *
                                            (unprotFutuRPLowers - unprotFutuRPUppers) / 2.0)
                # print(baseInunLv, '\n', unprotRPUppers, '\n', unprotInunLvUppers, '\n', unprotVulUppers, '\n',
                #       unprotFutuRPUppers, '\n', unprotFutuInunLvUppers, '\n', unprotFutuVulUppers)
                # testprint = '%s_%s_%s_%s' % (design_protection, future_protection, baseExpectedDamage, futuExpectedDamage)
                # print(colored(testprint, 'red'))
                return future_protection, baseExpectedDamage, futuExpectedDamage

    def find_vulnerability_threshold_desending(self, df, cut_off_threshold, threshold_of_interest):
        """
        look up or Interpolate vulnerability factor of a given threshold.

        :param df: pandas dataframe, vulnerability factor table.
        :param cut_off_threshold: float, return 0 if the threshold of interest is smaller than 0.
        :param threshold_of_interest: float, threshold of interest.
        :return: float, a vulnerability factor.
        """
        # df = df[(df['Turbine'] == turbine & df['Cooling'] == cooling)]
        cut_off_threshold = float(cut_off_threshold)
        d = float(threshold_of_interest)
        if d > cut_off_threshold:
            toi_vul = 0.0
        elif d == cut_off_threshold:
            toi_vul = df[df['Threshold'] == cut_off_threshold]['Vulnerability'].values[0]
        else:
            df = df[df['Threshold'] <= cut_off_threshold]
            df.sort_values(by='Threshold', inplace=True)
            # print(df)
            t = df['Threshold'].values
            v = df['Vulnerability'].values
            d = float(threshold_of_interest)
            di = bisect.bisect_right(t, d)
            diL, diR = di - 1, di
            x1, x2, y1, y2 = t[diL], t[diR], v[diL], v[diR]
            toi_vul = ((d - x1) / (x2 - x1)) * (y2 - y1) + y1
            if toi_vul > np.max(v):
                toi_vul = np.max(v)
        return toi_vul

    def find_vulnerability_threshold_asending(self, df, cut_off_threshold, threshold_of_interest):
        """
        look up or Interpolate vulnerability factor of a given threshold.

        :param df: pandas dataframe, vulnerability factor table.
        :param cut_off_threshold: float, return 0 if threshold of interest is smaller than 0.
        :param threshold_of_interest: float, threshold of interest.
        :return: float, a vulnerability factor.
        """
        # df = df[(df['Turbine'] == turbine & df['Cooling'] == cooling)]
        cut_off_threshold = float(cut_off_threshold)
        d = float(threshold_of_interest)
        if d < cut_off_threshold:
            toi_vul = 0.0
        elif d == cut_off_threshold:
            toi_vul = df[df['Threshold'] == cut_off_threshold]['Vulnerability'].values[0]
        else:
            df = df[df['Threshold'] >= cut_off_threshold]
            df.sort_values(by='Threshold', inplace=True)
            # print(df)
            t = df['Threshold'].values
            v = df['Vulnerability'].values
            d = float(threshold_of_interest)
            if d >= np.max(t):
                toi_vul = np.max(v)
            else:
                di = bisect.bisect_right(t, d)
                diL, diR = di - 1, di
                x1, x2, y1, y2 = t[diL], t[diR], v[diL], v[diR]
                toi_vul = ((d - x1) / (x2 - x1)) * (y2 - y1) + y1
                if toi_vul > np.max(v):
                    toi_vul = np.max(v)
        return toi_vul

    def flood_file_batch_moving(self, root, new):
        """
        Simply restructuring directories and subdirectories here

        :param root: folder path-like string, root folder path.
        :param new: folder path-like string, new folder path.
        :return: None
        """
        for path, subdirs, files in os.walk(root):
            for name in files:
                print(os.path.join(path, name))
                os.rename(os.path.join(path, name), os.path.join(new, name))

    def gee_climate_data_restructuring(self, folder_path, file_name):
        """
        Restructure climate data, i.e., GDDP data downloaded from GEE

        :param folder_path: string, folder path.
        :param file_name: string, file name.
        :return: pandas dataframe, reconstructed climate data.
        """

        def time_series_reconstruct(folder_path, file_name):
            fp = os.path.join(folder_path, file_name)
            df = {
                '.csv': pd.read_csv,
                '.xlsx': pd.read_excel
            }.get(os.path.splitext(fp)[-1])(fp)
            df['check_year'] = df.apply(lambda row: number_of_days_in_a_year(row), axis=1)
            df['data_new'] = df.apply(lambda row: data_string_to_numbers(row), axis=1)
            df['check_data'] = df.apply(lambda row: counting_data_points(row), axis=1)
            df['check_compare'] = df['check_year'] - df['check_data']
            # print(df['check_compare'])
            # if df['check_compare'].sum() == 0:
            #     print('data points checked okay.')
            # else:
            #     print('errors in matching dates with data points. If data are projections, could be the leap year issue.')
            dateList, dataList, indicatorList = [], [], []
            indicator_col = [i for i in df.columns if 'indicator' in i][0]
            for y, d, i in zip(df['year'], df['data_new'], df[indicator_col]):
                sd, ed = date(y, 1, 1), date(y, 12, 31)
                delta = ed - sd
                days = [sd + timedelta(days=i) for i in range(delta.days + 1)]

                # # OPT 1. take out feb 29 data for projections only
                # if '1980_2005' in file_name or '2006_2099' in file_name:
                #     if len(days) == 366:
                #         days.remove(date(y,2,29))

                # OPT 2. take out feb 29 data for everything
                if len(days) == 366:
                    days.remove(date(y, 2, 29))

                for day, data in zip(days, d):
                    dateList.append(day)
                    dataList.append(data)
                    indicatorList.append(i)
            df_data = np.transpose([dateList, dataList, indicatorList])
            columns = ['date', 'data', 'indicator']
            dfNew = pd.DataFrame(data=df_data, columns=columns)
            # print(dfNew)
            # outfp = os.path.join(temp_output_folder, '%s_reformatted.xlsx' % file_name)
            # dfNew.to_excel(outfp, index=False)
            return dfNew

        def data_string_to_numbers(row):
            ds = row['data']
            ds = ds.replace('[', '').replace(']', '')
            ds = ds.split(', ')
            ds = [float(d) for d in ds]
            return ds

        def number_of_days_in_a_year(row):
            year = row['year']
            if calendar.isleap(year):
                return 366
            else:
                return 365

        def counting_data_points(row):
            p = row['data_new']
            return len(p)

        df = time_series_reconstruct(folder_path=folder_path, file_name=file_name)
        return df

    def export_restructured_airtemp(self, in_folder, out_folder):
        """
        Export output of gee_climate_data_restructuring() as csv files to local disk.

        :param in_folder: folder path-like string, input folder path.
        :param out_folder: folder path-like string, output folder path.
        :return: None
        """
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        basename_list = [i for i in os.listdir(in_folder) if not i.endswith('.zip')]

        def f(basename):
            df = self.gee_climate_data_restructuring(folder_path=in_folder, file_name=basename)
            out_fn = os.path.splitext(basename)[0].replace('V2_', '')
            df.to_csv(os.path.join(out_folder, out_fn + '.csv'))

        list(map(f, tqdm.tqdm(basename_list)))
        print(f'Done exporting restructured air temperatures for the air temperature module. '
              f'Find output here: {out_folder}')

    def export_restructured_era5_for_airtemp_module(self, in_folder, out_folder):
        """
        Restructure era5 datasets downloaded from EE in the right format for running air temperature module

        :param in_folder: folder path-like string, input folder path.
        :param out_folder: folder path-like string, output folder path.
        :return: None
        """
        if not os.path.exists(out_folder):
            os.mkdir(out_folder)
        basename_list = [i for i in os.listdir(in_folder) if not i.endswith('.zip')]

        def f(basename):
            df = self.gee_climate_data_restructuring(folder_path=in_folder, file_name=basename)
            indicator_col = [i for i in df.columns if 'indicator' in i][0]
            df[indicator_col] = df[indicator_col].replace({'mean_2m_air_temperature': 'tasavg',
                                                           'maximum_2m_air_temperature': 'tasmax',
                                                           'minimum_2m_air_temperature': 'tasmin'})
            timespan_str = os.path.splitext(basename)[0].split('_')[-1]
            year_min, year_max = timespan_str.split('-')[-2], timespan_str.split('-')[-1]
            out_fn = basename.replace(f'ERA5_{year_min}-{year_max}', f'historical_NAN_{year_min}_{year_max}')
            df.to_csv(os.path.join(out_folder, out_fn))

        list(map(f, tqdm.tqdm(basename_list)))
        print(f'Done exporting restructured era5 datasets for the air temperature module. '
              f'Find output here: {out_folder}')

    def debug_gddp_ts_com(self):
        indicator = 'tasmin'
        for plant_id in range(1, 14):
            df1 = self.gee_climate_data_restructuring(folder_path=r'C:\Users\yan.cheng\Downloads',
                                                      file_name=f'V2_PL_EBRD_TPP{plant_id}_historical_GFDL-ESM2M_1965_2005.xlsx')
            df1 = df1[(df1['indicator'] == indicator)]
            df1['date'] = pd.to_datetime(df1['date'])

            df2 = self.gee_climate_data_restructuring(
                folder_path=r'C:\Users\yan.cheng\PycharmProjects\EBRD\tpp_climate_gddp_all',
                file_name=f'PL_EBRD_TPP{plant_id}_historical_GFDL-ESM2M_1950_2005.csv')
            df2 = df2[(df2['indicator'] == indicator)]
            df2['date'] = pd.to_datetime(df2['date'])

            fig, ax = plt.subplots()
            ax.plot(df1.date, df1.data, label='new')
            ax.plot(df2.date, df2.data, label='old')
            ax.legend()
            fig.savefig(
                os.path.join(self.project_folder, 'output_temp',
                             f'timeseries_gddp_{plant_id}_historical_{indicator}.png'))

    def debug_gddp_viz_ts(self, plant_id_list, save_fig=None):
        indicator = 'tasmin'
        for plant_id in plant_id_list:
            fig, ax = plt.subplots()
            for scenario in ['historical', 'rcp45', 'rcp85']:
                years = {
                    'historical': '1950_2005',
                    'rcp45': '2006_2070',
                    'rcp85': '2006_2070',
                }.get(scenario)
                for model in ['inmcm4', 'MPI-ESM-LR', 'MRI-CGCM3', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M',
                              'GFDL-ESM2M', 'CCSM4', 'CNRM-CM5']:
                    df1 = self.gee_climate_data_restructuring(
                        folder_path=r'C:\Users\yan.cheng\PycharmProjects\ebrd-gee\data',
                        file_name=f'V2_PL_EBRD_TPP{plant_id}_{scenario}_{model}_{years}.xlsx')
                    df1 = df1[(df1['indicator'] == indicator)]
                    df1['date'] = pd.to_datetime(df1['date'])
                    ax.plot(df1.date, df1.data, label=model)
            plt.show()
            if save_fig is True:
                fig.savefig(
                    os.path.join(self.project_folder, 'output_temp',
                                 f'timeseries_gddp_{plant_id}.png'))
            else:
                plt.show()

    def debug_gddp_viz_boxplot(self, plant_id_list, save_fig=None):
        indicator = 'tasmin'
        for plant_id in plant_id_list:
            fig, axes = plt.subplots()
            df_list = []
            for scenario in ['historical', 'rcp45', 'rcp85']:
                years = {
                    'historical': '1950_2005',
                    'rcp45': '2006_2070',
                    'rcp85': '2006_2070',
                }.get(scenario)
                for model in ['inmcm4', 'MPI-ESM-LR', 'MRI-CGCM3', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M',
                              'GFDL-ESM2M', 'CCSM4', 'CNRM-CM5']:
                    df1 = self.gee_climate_data_restructuring(
                        folder_path=r'C:\Users\yan.cheng\PycharmProjects\ebrd-gee\data',
                        file_name=f'V2_PL_EBRD_TPP{plant_id}_{scenario}_{model}_{years}.xlsx')
                    df1 = df1[(df1['indicator'] == indicator)]
                    df1['model'] = model
                    df1['date'] = pd.to_datetime(df1['date'])
                    df1['scenario'] = scenario
                    df_list.append(df1)
                    # ax.plot(df1.data)
            df_all = pd.concat(df_list)
            sns.boxplot(data=df_all, x='model', y='data', hue='scenario', ax=axes)
            axes.legend(loc='upper right')
            plt.title(f'Plant ID: {plant_id}')
            plt.xticks(rotation=90)
            plt.tight_layout()
            if save_fig is True:
                fig.savefig(
                    os.path.join(self.project_folder, 'output_temp',
                                 f'timeseries_gddp_{plant_id}.png'))
            else:
                plt.show()

    def flood_module(self, lat, lon, folder, buffer, year, design_protection, vulnerability_dataframe):
        """
        Assessment of flood-induced generation losses.

        :param lat: float, latitude.
        :param lon: float, longitude
        :param folder: folder path-like string, folder path of flood datasets.
        :param buffer: float, a radius in arc degree for searching pixels.
        :param year: int, 2010, 2030, or 2050, indicating the period of future projection.
        :param design_protection: float, e.g.,  1/250, 1/1000
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disrupted hours.
        """

        number_of_models = 5
        prots = [design_protection]
        heds = []
        feds = []
        scnrs = ['baseline']
        fs = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        hist_fs = [f for f in fs if 'WATCH' in f]
        hist_ds = []
        for f in hist_fs:
            ncfp = os.path.join(folder, f)
            # print(f)
            nc_attrs, nc_dims, nc_vars = self.ncDump(fp=ncfp, verb=False)
            ncvar = [v for v in nc_vars if 'inundation' in v][0]
            d = self.getNC4DataByLocation(fp=ncfp, lat=lat, lon=lon, ncvar=ncvar, nc_dims=nc_dims,
                                          buffer=buffer
                                          )
            hist_ds.append(d)
        full_futu_data_list = []

        if '45' in self.scenario_id_list:
            rcp45_fs = np.array_split([f for f in fs if str(year) in f and 'rcp4p5' in f], number_of_models)
            for i in np.arange(0, number_of_models, 1):
                ifs = rcp45_fs[i]
                modelName = ifs[0].split('_')[2]
                # print(modelName)
                futu_ds = []
                for f in ifs:
                    ncfp = os.path.join(folder, f)
                    # print(f)
                    nc_attrs, nc_dims, nc_vars = self.ncDump(fp=ncfp, verb=False)
                    ncvar = [v for v in nc_vars if 'inundation' in v][0]
                    d = self.getNC4DataByLocation(fp=ncfp, lat=lat, lon=lon, ncvar=ncvar, nc_dims=nc_dims,
                                                  buffer=buffer
                                                  )
                    futu_ds.append(d)
                # print(futu_ds)
                scnrTemp = str(year) + '-rcp45-' + modelName
                protTemp, hedTemp, fedTemp = self.flood_aed_base_futu(baseline_data=hist_ds,
                                                                      design_protection=design_protection,
                                                                      projected_data=futu_ds,
                                                                      vulnerability_dataframe=vulnerability_dataframe)
                # print(scnrTemp)
                prots.append(protTemp)
                heds.append(hedTemp)
                feds.append(fedTemp)
                scnrs.append(scnrTemp)
                full_futu_data_list.append(futu_ds)
        if '85' in self.scenario_id_list:
            rcp85_fs = np.array_split([f for f in fs if str(year) in f and 'rcp8p5' in f], number_of_models)
            for i in np.arange(0, number_of_models, 1):
                ifs = rcp85_fs[i]
                modelName = ifs[0].split('_')[2]
                # print(modelName)
                futu_ds = []
                for f in ifs:
                    ncfp = os.path.join(folder, f)
                    # print(f)
                    nc_attrs, nc_dims, nc_vars = self.ncDump(fp=ncfp, verb=False)
                    ncvar = [v for v in nc_vars if 'inundation' in v][0]
                    d = self.getNC4DataByLocation(fp=ncfp, lat=lat, lon=lon, ncvar=ncvar, nc_dims=nc_dims,
                                                  buffer=buffer
                                                  )
                    futu_ds.append(d)
                # print(futu_ds)
                scnrTemp = str(year) + '-rcp85-' + modelName
                protTemp, hedTemp, fedTemp = self.flood_aed_base_futu(baseline_data=hist_ds,
                                                                      design_protection=design_protection,
                                                                      projected_data=futu_ds,
                                                                      vulnerability_dataframe=vulnerability_dataframe)
                # print(scnrTemp)
                prots.append(protTemp)
                heds.append(hedTemp)
                feds.append(fedTemp)
                scnrs.append(scnrTemp)
                full_futu_data_list.append(futu_ds)

        aahd_hist_pm = np.array(heds) * 24
        aahd_futu_pm = np.array(feds) * 24
        med_aahd_hist = np.median(aahd_hist_pm)
        med_aahd_futu = np.median(aahd_futu_pm)
        uncert = 90
        q95_aahd_hist = np.percentile(aahd_hist_pm, q=uncert)
        q5_aahd_hist = np.percentile(aahd_hist_pm, q=100 - uncert)
        max_aahd_bc = np.amax(aahd_hist_pm)
        min_aahd_bc = np.amin(aahd_hist_pm)
        avg_aahd_bc = np.mean(aahd_hist_pm)
        q95_aahd_futu = np.percentile(aahd_futu_pm, q=uncert)
        q5_aahd_futu = np.percentile(aahd_futu_pm, q=100 - uncert)
        max_aahd_futu = np.amax(aahd_futu_pm)
        min_aahd_futu = np.amin(aahd_futu_pm)
        avg_aahd_futu = np.mean(aahd_futu_pm)
        # if med_aahd_futu == 0.0 and med_aahd_hist == 0.0:
        #     c = 0.0
        # elif med_aahd_futu == 0.0:
        #     c = 1.0
        # else:
        #     c = (med_aahd_hist - med_aahd_futu) / med_aahd_hist
        # if c == 0.0:
        #     changeText = "No change"
        # elif c > 0.0:
        #     changeText = u'\u25BC'
        # else:
        #     changeText = u'\u25B2'
        # # PRINT INDIVIDUAL PLANT RESULTS
        # print('Annual average hours of disruption due to floods: \n'
        #       'Current:{:.1f} ({:.1f}, +{:.1f}) | Future:{:.1f} ({:.1f}, +{:.1f})'.format(med_aahd_hist,
        #                                                                                   q5_aahd_hist - med_aahd_hist,
        #                                                                                   q95_aahd_hist - med_aahd_hist,
        #                                                                                   med_aahd_futu,
        #                                                                                   q5_aahd_futu - med_aahd_futu,
        #                                                                                   q95_aahd_futu - med_aahd_futu))
        # if c < 0.0:
        #     print(colored(changeText, 'red'), colored("{:.1%}".format(np.abs(c)), 'red'))
        # else:
        #     print(colored(changeText, 'green'), colored("{:.1%}".format(c), 'green'))
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        # print(med_aahd_hist, med_aahd_futu, q5_aahd_hist, q95_aahd_hist, q5_aahd_futu, q95_aahd_futu)
        # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
        return med_aahd_hist, med_aahd_futu, q95_aahd_hist, q95_aahd_hist, max_aahd_bc, min_aahd_bc, avg_aahd_bc, \
               q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu

    def drought_module(self, obs_fp, fut_fp, spei_threshold, vulnerability_dataframe, historical_years,
                       projection_years):
        """
        Assessment of drought-induced generation losses.

        :param obs_fp: file path-like string, file path of observation datasets over the baseline period.
        :param fut_fp:  file path-like string, file path of projected datasets over the projection period.
        :param spei_threshold: float, spei threshold, a tipping point when sever droughts have a negative impact on generation losses.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factor table.
        :param historical_years: list or tuple, start year and end year of the historical period.
        :param projection_years: list or tuple, start year and end year of the projection period.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disruption hours.
        """

        def add_vulnerability(row):
            toi = row['spei_12']
            # print(toi)
            toi_vul = self.find_vulnerability_threshold_desending(df=vulnerability_dataframe,
                                                                  cut_off_threshold=spei_threshold,
                                                                  threshold_of_interest=toi)
            return toi_vul

        models_available = ['CCSM4', 'CNRM-CM5', 'GFDL-ESM2M', 'HadGEM2-ES', 'INM-CM4', 'MPI-ESM-LR', 'MRI-CGCM3']
        scenarios_available = ['historical', 'rcp45', 'rcp85']
        # df_obs = pd.read_csv(obs_fp)
        # df_obs['date'] = df_obs.apply(lambda row: spei_get_date(row), axis=1)
        df_fut = pd.read_csv(fut_fp, index_col=0)
        df_fut.dropna(axis=0, how='any', inplace=True)
        df_fut['date'] = df_fut.apply(lambda row: self.spei_get_date(row), axis=1)
        df_fut['vulnerability'] = df_fut.apply(lambda row: add_vulnerability(row), axis=1)
        df_fut['disrupted_days'] = df_fut['vulnerability'] * 30.5
        # for x, y, z in zip(df_fut['spei_12'].values, df_fut['vulnerability'].values, df_fut['disrupted_days'].values):
        #     if x <= spei_threshold:
        #         print(x, y, z)
        bc_daysums = []
        df_bc = df_fut[(df_fut['simulation_scenario'] == scenarios_available[0]) &
                       (df_fut['year'] >= historical_years[0]) &
                       (df_fut['year'] <= historical_years[1])]
        bc_tot_days = (historical_years[1] - historical_years[0] + 1) * 12 * 30.5
        for m in models_available:
            dftemp = df_bc[df_bc['model'] == m]
            daysum = np.sum(dftemp['disrupted_days'].values)
            bc_daysums.append(daysum)

        r45_85_daysums = []
        r45_daysums = []
        r85_daysums = []
        if '45' in self.scenario_id_list:
            df_r45 = df_fut[(df_fut['simulation_scenario'] == scenarios_available[1]) &
                            (df_fut['year'] >= projection_years[0]) &
                            (df_fut['year'] <= projection_years[1])]
            for m in models_available:
                dftemp = df_r45[df_r45['model'] == m]
                daysum = np.sum(dftemp['disrupted_days'].values)
                r45_85_daysums.append(daysum)
                r45_daysums.append(daysum)
        if '85' in self.scenario_id_list:
            df_r85 = df_fut[(df_fut['simulation_scenario'] == scenarios_available[2]) &
                            (df_fut['year'] >= projection_years[0]) &
                            (df_fut['year'] <= projection_years[1])]
            for m in models_available:
                dftemp = df_r85[df_r85['model'] == m]
                daysum = np.sum(dftemp['disrupted_days'].values)
                r45_85_daysums.append(daysum)
                r85_daysums.append(daysum)

        fut_tot_days = (projection_years[1] - projection_years[0] + 1) * 12 * 30.5
        aahd_bc_pm = np.array(bc_daysums) / bc_tot_days * 365 * 24
        aahd_futu_pm = np.array(r45_85_daysums) / fut_tot_days * 365 * 24
        aahd_r45_pm = np.array(r45_daysums) / fut_tot_days * 365 * 24
        aahd_r85_pm = np.array(r85_daysums) / fut_tot_days * 365 * 24
        med_aahd_bc = np.median(aahd_bc_pm)
        med_aahd_futu = np.median(aahd_futu_pm)
        uncert = 90
        q95_aahd_bc = np.percentile(aahd_bc_pm, q=uncert)
        q5_aahd_bc = np.percentile(aahd_bc_pm, q=100 - uncert)
        max_aahd_bc = np.amax(aahd_bc_pm)
        min_aahd_bc = np.amin(aahd_bc_pm)
        avg_aahd_bc = np.mean(aahd_bc_pm)
        q95_aahd_futu = np.percentile(aahd_futu_pm, q=uncert)
        q5_aahd_futu = np.percentile(aahd_futu_pm, q=100 - uncert)
        max_aahd_futu = np.amax(aahd_futu_pm)
        min_aahd_futu = np.amin(aahd_futu_pm)
        avg_aahd_futu = np.mean(aahd_futu_pm)

        plant_id = int(os.path.basename(fut_fp).split('_')[0])
        model_list = models_available
        hist_loss_list = aahd_bc_pm.tolist()
        r45_loss_list = aahd_r45_pm.tolist()
        r85_loss_list = aahd_r85_pm.tolist()
        test_df = pd.DataFrame(zip(model_list, hist_loss_list, r45_loss_list, r85_loss_list),
                               columns=['model', 'hist_loss_hrs', 'r45_loss_hrs', 'r85_loss_hrs'])
        test_df['plant_id'] = plant_id
        test_df = test_df[['plant_id', 'model', 'hist_loss_hrs', 'r45_loss_hrs', 'r85_loss_hrs']]
        self.test_out.append(test_df)

        return med_aahd_bc, med_aahd_futu, q5_aahd_bc, q95_aahd_bc, max_aahd_bc, min_aahd_bc, avg_aahd_bc, \
               q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu

    def air_temperature_module(self, models, scenarios, plant_id, backcast_years, projection_years,
                               vulnerability_dataframe, desirable_air_temp, shutdown_air_temp, dat_thred_stat,
                               sat_thred_stat):
        """
        Assessment of air temperature-induced generation losses.

        :param models: list, climate models to be assessed.
        :param scenarios: list, climate scenarios to be assessed.
        :param plant_id: int, plant reference id.
        :param backcast_years: list, start year and end year of the backcast/baseline period.
        :param projection_years: list, start year and end year of the projection period.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors table.
        :param desirable_air_temp: int or float or -3001.0, design air temperature. No generation losses are expected at air temperature below or equal to the design air temperature. -3001.0 will allow taking the mean or a specific percentile of values over the baseline period as the threshold depending on the value of dat_thred_stat.
        :param shutdown_air_temp: int or float or -3001.0, design shutdown air temperature. -3001.0 will allow taking the mean or a specific percentile of values over the baseline period as the threshold depending on the value of sat_thred_stat.
        :param dat_thred_stat: string or float, string -> 'mean', float -> percentile value bewteen 0 and 1.
        :param sat_thred_stat: float or -3001.0, float -> percentile value bewteen 0 and 1; -3001.0 -> the shutdown air temperature will be the value when the vulnerability factor is equal to 1.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disrupted hours.
        """

        def add_vulnerability(row, dat, sat):
            tmax = row['tasmax']
            tavg = row['tasavg']
            if tmax >= sat:
                toi_vul = 1.0
            else:
                toi = tavg - dat
                toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe, cut_off_threshold=0.0,
                                                                     threshold_of_interest=toi)
            return toi_vul

        aahd_bc_pm = []
        aahd_futu_pm = []
        backcast_suffix = '_'.join(data_configs.RAWDATA_TIMESPAN['GDDP'][0].split('-'))
        projection_suffix = '_'.join(data_configs.RAWDATA_TIMESPAN['GDDP'][1].split('-'))
        for m in models[:]:
            bc_fn = 'PL_EBRD_TPP%s_%s_%s_%s' % (plant_id, scenarios[0], m, backcast_suffix)
            df_bc = pd.read_csv(os.path.join(self.climate_data_folder, bc_fn + '.csv'), index_col=0)
            df_bc['date'] = pd.to_datetime(df_bc['date'])
            df_bc = df_bc[(df_bc['date'].dt.year >= backcast_years[0]) & (df_bc['date'].dt.year <= backcast_years[1])]
            bc_tmaxs = df_bc[df_bc['indicator'] == 'tasmax']['data'].values - 273.15
            bc_tmins = df_bc[df_bc['indicator'] == 'tasmin']['data'].values - 273.15
            if 'tasavg' in df_bc.indicator.unique():
                bc_tavgs = df_bc[df_bc['indicator'] == 'tasavg']['data'].values - 273.15
            else:
                bc_tavgs = (bc_tmaxs + bc_tmins) / 2.0
            if desirable_air_temp is None or desirable_air_temp == -3001.0:
                if dat_thred_stat == 'mean':
                    desireTemp = np.mean(bc_tavgs)
                elif isinstance(dat_thred_stat, (int, float, complex)):
                    desireTemp = np.percentile(bc_tavgs, dat_thred_stat)
                else:
                    raise NotImplementedError
            else:
                desireTemp = desirable_air_temp
            if shutdown_air_temp is None or shutdown_air_temp == -3001.0:
                if sat_thred_stat is None or sat_thred_stat == -3001.0:
                    shutTemp = \
                        vulnerability_dataframe[vulnerability_dataframe['Vulnerability'] == 1]['Threshold'].values[
                            0] + desireTemp
                elif isinstance(sat_thred_stat, (int, float, complex)):
                    shutTemp = np.percentile(bc_tmaxs, sat_thred_stat)
                else:
                    raise NotImplementedError
            else:
                shutTemp = shutdown_air_temp
            new_line_dat = self.export_temp_var(desireTemp,
                                                prefix=f'Plant ID: {plant_id} | Model: {m} | Desired Air Temperature-{str(dat_thred_stat)}',
                                                output_folder=self.final_assessment_dir)
            new_line_sat = self.export_temp_var(shutTemp,
                                                prefix=f'Plant ID: {plant_id} | Model: {m} | Shutdown Air Temperature-{str(sat_thred_stat)}',
                                                output_folder=self.final_assessment_dir)
            print(f'{new_line_dat}\n{new_line_sat}')

            df_bc = df_bc[df_bc['indicator'] == 'tasmax']
            df_bc.sort_values(by='date', inplace=True)
            df_bc['tasmax'] = bc_tmaxs
            df_bc['tasavg'] = bc_tavgs
            df_bc['vulnerability'] = df_bc.apply(lambda row: add_vulnerability(row, dat=desireTemp, sat=shutTemp),
                                                 axis=1)
            bc_hd_days = np.sum(df_bc['vulnerability'].values)
            bc_tot_days = len(df_bc['date'])
            bc_aahd = bc_hd_days / bc_tot_days * 365 * 24
            aahd_bc_pm.append(bc_aahd)

            if '45' in self.scenario_id_list:
                r45_fn = 'PL_EBRD_TPP%s_%s_%s_%s' % (plant_id, scenarios[1], m, projection_suffix)
                df_r45 = pd.read_csv(os.path.join(self.climate_data_folder, r45_fn + '.csv'), index_col=0)
                df_r45['date'] = pd.to_datetime(df_r45['date'])
                df_r45 = df_r45[
                    (df_r45['date'].dt.year >= projection_years[0]) & (df_r45['date'].dt.year <= projection_years[1])]
                r45_tmaxs = df_r45[df_r45['indicator'] == 'tasmax']['data'].values - 273.15
                r45_tmins = df_r45[df_r45['indicator'] == 'tasmin']['data'].values - 273.15
                r45_tavgs = (r45_tmaxs + r45_tmins) / 2.0
                df_r45 = df_r45[df_r45['indicator'] == 'tasmax']
                df_r45['tasmax'] = r45_tmaxs
                df_r45['tasavg'] = r45_tavgs
                df_r45['vulnerability'] = df_r45.apply(lambda row: add_vulnerability(row, dat=desireTemp, sat=shutTemp),
                                                       axis=1)
                r45_hd_days = np.sum(df_r45['vulnerability'].values)
                r45_tot_days = len(df_r45['date'])
                r45_aahd = r45_hd_days / r45_tot_days * 365 * 24
                aahd_futu_pm.append(r45_aahd)
            if '85' in self.scenario_id_list:
                r85_fn = 'PL_EBRD_TPP%s_%s_%s_%s' % (plant_id, scenarios[2], m, projection_suffix)
                df_r85 = pd.read_csv(os.path.join(self.climate_data_folder, r85_fn + '.csv'), index_col=0)
                df_r85['date'] = pd.to_datetime(df_r85['date'])
                df_r85 = df_r85[
                    (df_r85['date'].dt.year >= projection_years[0]) & (df_r85['date'].dt.year <= projection_years[1])]
                r85_tmaxs = df_r85[df_r85['indicator'] == 'tasmax']['data'].values - 273.15
                r85_tmins = df_r85[df_r85['indicator'] == 'tasmin']['data'].values - 273.15
                r85_tavgs = (r85_tmaxs + r85_tmins) / 2.0
                df_r85 = df_r85[df_r85['indicator'] == 'tasmax']
                df_r85['tasmax'] = r85_tmaxs
                df_r85['tasavg'] = r85_tavgs
                df_r85['vulnerability'] = df_r85.apply(lambda row: add_vulnerability(row, dat=desireTemp, sat=shutTemp),
                                                       axis=1)
                r85_hd_days = np.sum(df_r85['vulnerability'].values)
                r85_tot_days = len(df_r85['date'])
                r85_aahd = r85_hd_days / r85_tot_days * 365 * 24
                aahd_futu_pm.append(r85_aahd)

        uncert = 90

        med_aahd_bc = np.median(aahd_bc_pm)
        q95_aahd_bc = np.percentile(aahd_bc_pm, q=uncert)
        q5_aahd_bc = np.percentile(aahd_bc_pm, q=100 - uncert)
        max_aahd_bc = np.amax(aahd_bc_pm)
        min_aahd_bc = np.amin(aahd_bc_pm)
        avg_aahd_bc = np.mean(aahd_bc_pm)

        if not self.scenario_id_list == []:
            med_aahd_futu = np.median(aahd_futu_pm)
            q95_aahd_futu = np.percentile(aahd_futu_pm, q=uncert)
            q5_aahd_futu = np.percentile(aahd_futu_pm, q=100 - uncert)
            max_aahd_futu = np.amax(aahd_futu_pm)
            min_aahd_futu = np.amin(aahd_futu_pm)
            avg_aahd_futu = np.mean(aahd_futu_pm)
        else:
            med_aahd_futu, q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu = [-9999] * 6

        return med_aahd_bc, med_aahd_futu, q5_aahd_bc, q95_aahd_bc, max_aahd_bc, min_aahd_bc, avg_aahd_bc, \
               q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu

    def histDailyWaterTemp_to_weeklyEfficiencyLoss(self, df, vulnerability_dataframe, desirable_water_temp,
                                                   shutdown_water_temp):
        """
        Calculate weekly power efficiency losses from historical daily water temperature.

        :param df: pandas dataframe, historical daily water temperature.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors.
        :param desirable_water_temp: int or float, design water temperature.
        :param shutdown_water_temp: int or float, shutdown water temperature.
        :return: pandas.DataFrame, date (last day in a week) and aggregated efficiency loss in a week.
        """

        def add_vulnarability(water_temp, dwt, swt):
            if np.isnan(water_temp):
                toi_vul = np.nan
            elif water_temp >= swt:
                toi_vul = 1.0
            elif water_temp <= dwt:
                toi_vul = 0.0
            else:
                toi = water_temp - dwt
                toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe, cut_off_threshold=0.0,
                                                                     threshold_of_interest=toi)
            if toi_vul < 0:
                toi_vul = 0

            return toi_vul

        df['efficiency_loss'] = df.apply(
            lambda row: add_vulnarability(row['water_temp_C'], desirable_water_temp, shutdown_water_temp), axis=1)

        df_efficiency_loss = df[['date', 'value', 'water_temp_C', 'efficiency_loss']].set_index('date')
        df_efficiency_loss['year'] = df_efficiency_loss.index.year
        by_year = df_efficiency_loss.groupby('year')

        def f(x):
            # average every 7 days in a year
            weekly_efficiency_loss = x.resample('7D', loffset='6D').sum()['efficiency_loss']

            weekly_water_temp = x.resample('7D', loffset='6D').mean()[['value', 'water_temp_C']]
            df = pd.concat([weekly_water_temp, weekly_efficiency_loss], axis=1)
            df.columns = ['weekly_water_temp_K', 'weekly_water_temp_C', 'weekly_efficiency_loss']
            diffs = np.setdiff1d(df.index.date, x.index.date)
            # remove the extra days over 364 days
            return df[~np.in1d(df.index.date, diffs)]

        return by_year.apply(lambda x: f(x)).reset_index()

    def futu_loss(self, loc_id, vulnerability_dataframe, projection_years, desirable_water_temp,
                  shutdown_water_temp, model, scenario):
        """
        Calculate efficiency losses over the future period based on weekly water temperatures and vulnerability factors.

        :param loc_id: int, plant id.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors,
        :param projection_years: list or tuple, [start year, end year]
        :param desirable_water_temp: float, desired intake water temperature.
        :param shutdown_water_temp:  float, desired shutdown water temperature.
        :param model: string, climate model.
        :param scenario: string, climate scenario.
        :return: pandas dataframe,
        """

        def add_vulnarability(water_temp, dwt, swt):
            if np.isnan(water_temp):
                toi_vul = np.nan
            elif water_temp >= swt:
                toi_vul = 1.0
            elif water_temp <= dwt:
                toi_vul = 0.0
            else:
                toi = water_temp - dwt  # for once through
                toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe, cut_off_threshold=0.0,
                                                                     threshold_of_interest=toi)
            if toi_vul < 0:
                toi_vul = 0

            return toi_vul

        file_path = glob.glob(os.path.join(self.tpp_water_temp_folder, f'TPP_{str(loc_id)}_waterTemp_weekAvg_output_'
                                                                       f'{model}_{scenario}_*.csv'))[0]
        df_all = pd.read_csv(file_path, index_col=0)
        df_all['date'] = pd.to_datetime(df_all['date'])
        df = df_all[(df_all['date'].dt.year >= projection_years[0]) & (df_all['date'].dt.year <= projection_years[1])]
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df['water_temp_C'] = df['value'] - 273.15
        df['weekly_efficiency_loss'] = df.apply(
            lambda row: add_vulnarability(row['water_temp_C'], desirable_water_temp, shutdown_water_temp), axis=1)
        return df

    def regressor(self, X, y, model, test_size=None, random_state=None):
        """
        linear regression model.

        :param X: pandas.DataFrame or array with shape as (-1, 1)
        :param y: pandas.DataFrame or array with shape as (-1, 1)
        :param model: string, name of models to be applied, either 'rf' or 'ols'
        :param test_size: float, default value as 0.2, the percentage of entries for model evaluation
        :param random_state: int, default value as 0
        :return:
        """

        if test_size is None:
            test_size = 0.2
        if random_state is None:
            random_state = 0

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        if model == 'rf':
            regr = RandomForestRegressor(n_estimators=1000, max_depth=7, random_state=0)
            regr.fit(X_train, y_train)
        elif model == 'ols':
            regr = LinearRegression()
            regr.fit(X_train, y_train)
        else:
            print('Only two models are applicable for now: random forest regressor, OLS linear regressor')

        # Evaluate model performance
        predictions = regr.predict(X_test)
        # Absolute error
        errors = abs(predictions - y_test)
        # print('Mean Absolute Error:', round(np.mean(errors), 5))
        # Mean absolute percentage error
        mape = 100 * errors / y_test
        # Accuracy
        accuracy = round(100 - np.mean(mape), 4)
        # print('Accuracy:', accuracy, '%.')

        return regr, errors, accuracy

    def futuWeekly_to_efficiencyLoss(self, regr, loc_id, climate_model, climate_scenario, vulnerability_dataframe,
                                     backcast_years, projection_years, desirable_water_temp, shutdown_water_temp):
        """
        Predict future weekly power efficiency loss as a function of weekly water temperatures.

        :return: pandas dataframe,
        """

        def fit_model(x):
            if pd.isnull(x):
                y = np.nan
            elif x <= desirable_water_temp:
                y = 0
            elif x >= shutdown_water_temp:
                y = 1
            else:
                y = regr.predict(np.array([[x]]))
                y = np.array(y).flatten()[0]
                if y < 0:
                    y = 0
            return y

        fp_futu = glob.glbo(os.path.join(self.tpp_water_temp_folder,
                                         f'TPP_{loc_id}_waterTemp_weekAvg_output_{climate_model}_'
                                         f'{climate_scenario}_*.csv'))[0]
        df_futu_all = pd.read_csv(fp_futu, index_col=0)
        df_futu_all['date'] = pd.to_datetime(df_futu_all['date'])
        df_futu = df_futu_all[(df_futu_all['date'].dt.year >= projection_years[0]) &
                              (df_futu_all['date'].dt.year['year'] <= projection_years[1])]
        df_futu['weekly_water_temp_C'] = df_futu['value'] - 273.15
        df_futu['weekly_efficiency_loss'] = df_futu.apply(lambda row: fit_model(row['weekly_water_temp_C']), axis=1)
        return df_futu

    def plot_weekly(self, df, loc_id):
        """
        Visualize weekly water temperature and weekly efficiency losses.

        :param df: pandas dataframe, including water temperatures and efficiency losses.
        :param loc_id: string, plant id.
        :return: None
        """

        fig = make_subplots(specs=[[{'secondary_y': True}]])

        fig.add_trace(
            go.Scatter(x=df['date'], y=df['weekly_efficiency_loss'],
                       name='Weekly Power Efficiency Loss'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=df['date'], y=df['weekly_water_temp_C'],
                       name='Weekly Water Temperature (Celsius)'),
            secondary_y=True,
        )

        fig.update_layout(
            title_text='Power Efficiency Loss VS Water Temperature Fluctuation (Power Plant ID {})'.format(loc_id))
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Weekly Power Efficiency Loss', secondary_y=False)
        fig.update_yaxes(title_text='Weekly Water Temperature (Celsius)', secondary_y=True)

        fig.show()

    def kelvin_to_celsius(self, x):
        return x - 273.15

    def export_temp_var(self, temp_var, prefix=None, output_folder=None):
        """
        Export temporary output, normally for testing purpose.

        :param temp_var: string, variable name.
        :param prefix: string, prefix.
        :return: string,
        """

        output_folder = self.temp_output_folder if output_folder is None else output_folder
        new_line = f'{temp_var}' if prefix is None else f'{prefix}: {temp_var}'
        file_path = os.path.join(output_folder,
                                 '{}_ByPlant_{}.txt'.format(
                                     self.final_assessment_prefix.replace('final-assessment', 'temp-var'),
                                     self.run_code))
        mode = 'a' if os.path.exists(file_path) else 'w'
        with open(file_path, mode) as f:
            f.write(f'{new_line}\n')
        return new_line

    def rc_module(self, plant_id, vulnerability_dataframe, desirable_water_temp, dwt_thred_stat,
                  historical_years, projection_years, quantile=0.99):
        """
        Assessment of water temperature-induced generation losses for recirculating-cooling plants.

        :param plant_id: int, plant id.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors table.
        :param desirable_water_temp: int, float, design outlet water temperature.
        :param dwt_thred_stat: string or float, string -> 'mean',  float -> quantile/percentile between 0 and 1.
        :param historical_years: list, start year and end year of the historical period.
        :param projection_years: list, start year and end year of the projection period.
        :param quantile: float, between 0 and 1, the quantile used for calculating wet-bulb temperature threshold.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disrupted hours.
        """

        # Default setting
        def get_to_from_ti(ti):
            """
            Return outlet water temperature for inlet water temperature.
            reference: Figure 6, page 7, https://pdfs.semanticscholar.org/30ea/7b17881dd4d0520f4bd2d4240902e1a03846.pdf

            :param ti: float, inlet water temperature
            :return: float, outlet water temperature
            """
            return 1.0191 * ti + 9.7951

        def get_wbtemp_threshold(df, quantile):
            """
            Return the tipping point of wet-bulb temperature when the efficiency starts to decrease. Only applicable for stream,
            wet cooling, once-through...
            In short, when wet-bulb temperature exceeds 1% of the max wet-bulb temp within the time span(1980-2019).
            reference: https://pdfs.semanticscholar.org/30ea/7b17881dd4d0520f4bd2d4240902e1a03846.pdf

            :param plant_id: int or string, plant unique id
            :param quantile: float, quantile-like value, any values between 0 to 1
            :return: float, temperature, the unit of which inherits from the input data
            """
            ind_list = [i for i in df.columns if i not in ['indicator', 'value']]
            df = df.pivot(index=ind_list, columns='indicator', values='value')
            return np.quantile(df['wet_bulb_temperature'][~np.isnan(df['wet_bulb_temperature'])], q=quantile)

        def get_desiable_to(historical_ti, dwt_thred_stat):
            """
            Calculate design outlet water temperature from historical intake water temperatures.

            :param historical_ti: list or np arrary, intake water temperatures over the historical period.
            :param dwt_thred_stat: string or float, string -> 'mean', float -> percentile value, any value between 0 and 1.
            :return: int or float, design outlet water temperature.
            """
            if dwt_thred_stat == 'mean':
                ti_thrd = np.mean(historical_ti[~np.isnan(historical_ti)])
            elif isinstance(dwt_thred_stat, (int, float, complex)):
                ti_thrd = np.percentile(historical_ti[~np.isnan(historical_ti)], dwt_thred_stat)
            else:
                raise NotImplementedError

            desirable_to = get_to_from_ti(ti_thrd)
            desirable_to = 35.0 if desirable_to >= 35.0 else desirable_to
            return desirable_to

        def add_vulnerability(row, dwt, wbtemp_threshold):
            """
            Look up or interpolate vulnerability factor for a given threshold.

            :param row: one row in a pandas dataframe.
            :param dwt: int or float, design water temperature.
            :param wbtemp_threshold: int or float, wet-bulb temperature threshold.
            :return: float, vulnerability factor of a given threshold, between 0 and 1
            """
            if row['wet_bulb_temperature'] <= wbtemp_threshold:
                toi_vul = 0.0
            else:
                if pd.isna(row['outlet_water_temperature']) or pd.isna(row['wet_bulb_temperature']):
                    toi_vul = np.nan
                elif row['outlet_water_temperature'] <= row['wet_bulb_temperature']:
                    toi_vul = 1.0
                else:
                    toi = row['outlet_water_temperature'] - dwt
                    toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe,
                                                                         cut_off_threshold=0.0,
                                                                         threshold_of_interest=toi)
            return toi_vul

        def switch_datasets(x):
            return {
                'era5_1980-2019':
                    os.path.join(self.wbtemp_folder,
                                 f'PL_EBRD_TPP{str(plant_id)}_ERA5_1980-2019_restructure_withWetBulbTemp.csv'),
                'gddp_1980-2005':
                    os.path.join(self.gddp_recal_folder,
                                 f'PL_EBRD_TPP{str(plant_id)}_GDDP_1980-2005_withAirTempAvg'
                                 f'_biasCorrected_predictedWetBulbTemp.csv'),
                'gddp_2030-2070':
                    os.path.join(self.gddp_recal_folder,
                                 f'PL_EBRD_TPP{str(plant_id)}_GDDP_2030-2070_withAirTempAvg'
                                 f'_biasCorrected_predictedWetBulbTemp.csv'),
                'gddp_2010-2049':
                    os.path.join(self.gddp_recal_folder,
                                 f'PL_EBRD_TPP{str(plant_id)}_GDDP_2010-2049_withAirTempAvg'
                                 f'_biasCorrected_predictedWetBulbTemp.csv'),
                'waterTemp_futu_all':
                    os.path.join(self.watertemp_restructrue_all,
                                 f'PL_EBRD_TPP{str(plant_id)}_waterTemp-weekAvg-output_2006-2069.csv'),
                'waterTemp_hist_all':
                    os.path.join(self.watertemp_restructrue_all,
                                 f'PL_EBRD_TPP{str(plant_id)}_waterTemperature-mergedV2_1965-2010.csv'),
            }.get(x, 'No invalid input!')

        def aggr_weekly(**kwargs):
            """
            Apply various weekly aggregation to one or multiple indicators.

            :param kwargs: dictionary, carries the following arguments
            :param df: columns = ['date', 'year', 'month', 'day', 'scenario', 'model', 'value', '**'], amongst others, the date column must in the format of pd.datetime64.
            :param indicator: list or str, the indicators to which apply aggregation function.
            :param aggr_func: list or solo np function, e.g., np.mean, np.sum, np.max, np.min.
            :return: pd.DataFrame, with default columns = ['date', 'scenario', 'model'] and columns corresponding to indicators to which applied weekly aggregation.
            """

            if len(kwargs['indicator']) == 1 or len(kwargs['aggr_func']) == 1:
                kwargs['indicator_func'] = [i for i in itertools.product(kwargs['indicator'], kwargs['aggr_func'])]
            elif len(kwargs['indicator']) != len(kwargs['aggr_func']):
                sys.exit('The numbers of elements in indicator and aggr_func list are different!')
            else:
                kwargs['indicator_func'] = list(zip(kwargs['indicator'], kwargs['aggr_func']))

            kwargs = dict(zip([i for i in kwargs.keys()], [kwargs[item]
                                                           if item != 'df' and not isinstance(kwargs[item], str)
                                                           else [kwargs[item]] for item in kwargs]))

            df = kwargs['df'][0]
            df = df[[i for i in df.columns if i not in ['year', 'month', 'day']]].set_index('date')
            df.loc[:, 'year'] = df.index.year
            by_year = df.groupby([i for i in df.columns if i in ['year', 'scenario', 'model']])

            def f(x, *args):
                """
                Aggregate values in the same week using a specified method/function.

                :param args: indicator: string, the name of a column, for which to apply weekly aggregation; aggr_func: mean, sum, etc..
                :param x: df[column_name]
                :return:
                """
                indicator, aggr_func = args
                weekly_resampler = x[indicator].resample('7D', loffset='6D')
                weekly_series = weekly_resampler.agg(aggr_func).rename(f'{indicator}_weekly').to_frame()
                diffs = np.setdiff1d(weekly_series.index.date, x.index.date)
                return weekly_series[~np.in1d(weekly_series.index.date, diffs)]  # remove the extra days over 364 days

            df_weekly_list = list(map(lambda args: by_year.apply(lambda x: f(x, *args)), kwargs['indicator_func']))
            df_weekly = pd.concat(df_weekly_list, axis=1).reset_index()
            df_weekly = df_weekly[
                [i for i in df_weekly.columns if i in ['date', 'model', 'scenario']] + [f'{i}_weekly' for i in
                                                                                        kwargs['indicator']]]
            # df_weekly.to_csv(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'temp_weekly.csv'))
            return df_weekly

        def get_regr(historical_weekAgg):
            """
            Linear regression model of historical weekly water temperature aggregated from daily water temperature and weekly efficiency losses.

            :param historical_weekAgg: pandas dataframe, including historical dail water temperature and weekly aggregated water temperature.
            :return: linear regression model
            """
            historical_weekAgg = historical_weekAgg.dropna(subset=['wet_bulb_temperature',
                                                                   'outlet_water_temperature',
                                                                   'vulnerability'])
            regr, *_ = self.regressor(X=historical_weekAgg[['wet_bulb_temperature', 'outlet_water_temperature']],
                                      y=historical_weekAgg['vulnerability'],
                                      model='ols')
            return regr

        def fit_model(row, regr, wbtemp_threshold):
            """
            Fit regression model from get_regr()

            :param row: a row in pandas dataframe.
            :param regr: regression model from get_regr().
            :param wbtemp_threshold: int or float, wet-bulb temperature threshold.
            :return: list or np array, fitted efficiency losses as a function of wet-bulb temperature and outlet water temperature.
            """
            if row['wet_bulb_temperature'] <= wbtemp_threshold:
                y = 0
            else:
                if pd.isna(row['wet_bulb_temperature']) or pd.isna(row['outlet_water_temperature']):
                    y = np.nan
                elif row['outlet_water_temperature'] <= row['wet_bulb_temperature']:
                    y = 1
                else:
                    y = regr.predict([[row['wet_bulb_temperature'], row['outlet_water_temperature']]])
                    y = np.array(y).flatten()[0]
                    y = 0 if y < 0 else y
            return y

        def plot_rc_df(show_plot, plant_id, ti_name, to_name, twb_name, wbtemp_threshold, time_span, df_con):
            if show_plot:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(
                    go.Scatter(
                        x=df_con['date'],
                        y=df_con[ti_name],
                        name='water temperature (inlet water temperature)',
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_con['date'],
                        y=df_con[to_name],
                        name='outlet water temperature',
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_con['date'],
                        y=df_con[twb_name],
                        name='wet bulb temperature',
                    ),
                    secondary_y=False
                )
                fig.add_trace(
                    go.Scatter(
                        x=[np.datetime64(f'{time_span[0]}-01-01'), np.datetime64(f'{time_span[1]}-12-31')],
                        y=[wbtemp_threshold] * 2,
                        name='1% of wet-bulb temperature',
                    )
                )

                fig.update_xaxes(title_text='Date')
                fig.update_traces(mode='lines')
                fig.update_layout(title=f'Plant ID: {plant_id}')
                fig.show()

        def main(df_wbtemp, df_watertemp, time_span, desirable_water_temp, dwt_thred_stat=None, regr=None,
                 wbtemp_threshold=None, show_plot=False, apply_regr=False):
            """
            Calculate statistics of efficiency losses over the projection period, and parameters derived from data over the baseline period, i.e., design wet-bulb temperature, design water temperature, regression model (optional).

            :param df_wbtemp: pandas dataframe, wet-bulb temperature time series.
            :param df_watertemp: pandas dataframe, water temperature time series.
            :param time_span: list or tuple, start and end year of the study period.
            :param desirable_water_temp: int or float or -3001, design outlet water temperature, -3001 or None will allow calculating design outlet water temperature from water temperature time series over the baseline period according to the value of dwt_thred_stat.
            :param dwt_thred_stat: string or float, string -> 'mean', float -> quantile/percentile values between 0 and 1.
            :param regr: linear regression model, output of get_regr(), linear relationship between weekly efficiency losses aggregated from daily efficiency losses over the baseline period and weekly water temperature aggregated from daily water temperatures over the baseline period.
            :param wbtemp_threshold: int or float, wet-bulb temperature threshold.
            :param show_plot: boolean, apply plot_rc_df() and show time series plot.
            :param apply_regr: boolean, apply regression or not to correct bias due to different temporal resolutions of water temperatures over the baseline and the projection period.
            :return: aahd_pm, regr, wbtemp_threshold, desirable_water_temp
            """

            gddp_scenario_list = ('rcp45', 'rcp85')
            gddp_scenarios = [i for i in gddp_scenario_list for id_ in self.scenario_id_list if id_ in i]
            gddp_models = ('GFDL-ESM2M', 'NorESM1-M', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM')
            waterTemp_models = ('gfdl', 'ipsl', 'noresm', 'miroc')
            waterTemp_scenarios_list = ('rcp4p5', 'rcp8p5')
            waterTemp_scenarios = [i for i in waterTemp_scenarios_list for id_ in self.scenario_id_list if id_[0] in i]
            ti_name, to_name, twb_name, vul_name = ['water_temperature', 'outlet_water_temperature',
                                                    'wet_bulb_temperature',
                                                    'vulnerability']

            def switch_aggr_func(i):
                switch = {
                    ti_name: np.mean,
                    to_name: np.mean,
                    twb_name: np.mean,
                    vul_name: np.sum
                }
                return i, switch.get(i, 'No valid input!')

            if time_span == projection_years:
                ti_name = f'{ti_name}_weekly'
                to_name = f'{to_name}_weekly'
                df_wbtemp = df_wbtemp[df_wbtemp['model'].isin(gddp_models) & df_wbtemp['scenario'].isin(gddp_scenarios)]
                df_watertemp = df_watertemp[
                    df_watertemp['model'].isin(waterTemp_models) & df_watertemp['scenario'].isin(waterTemp_scenarios)]
                df_wbtemp.model = df_wbtemp['model'].replace(gddp_models, waterTemp_models)
                df_wbtemp.scenario = df_wbtemp['scenario'].replace(gddp_scenarios, waterTemp_scenarios)
            if time_span == historical_years:
                df_watertemp.scenario = np.nan

            df_con = pd.concat([df_wbtemp, df_watertemp])

            if time_span == historical_years:
                df_con.scenario = 'historical'
                df_con.model = 'historical'

            df_con.loc[df_con['indicator'].str.contains(twb_name), 'indicator'] = twb_name
            df_con.indicator = df_con['indicator'].replace('waterTemp', ti_name)

            df_con = df_con.loc[(df_con['year'] >= time_span[0])
                                & (df_con['year'] <= time_span[1]), :]

            df_con.loc[df_con['indicator'].isin([ti_name, twb_name]), 'value'] = \
                df_con.loc[df_con['indicator'].isin([ti_name, twb_name])] \
                    .apply(lambda row: self.kelvin_to_celsius(row['value']), axis=1)
            df_con.loc[:, 'date'] = pd.to_datetime(df_con[['year', 'month', 'day']])
            ind_list = [i for i in df_con.columns if i not in ['value', 'indicator']]

            if time_span == historical_years:
                # Read wet-bulb temperature and retrieve 1% of wet-bulb temperature
                wbtemp_threshold = get_wbtemp_threshold(df=df_con, quantile=quantile)
                new_line_wbt = self.export_temp_var(wbtemp_threshold,
                                                    prefix=f'Plant ID: {plant_id} | 99% of Wet-bulb Temperature-99',
                                                    output_folder=self.final_assessment_dir)
                print(new_line_wbt)

            df_sub = df_con[df_con['indicator'] == ti_name]
            df_sub.value = df_sub.apply(lambda row: get_to_from_ti(row['value']), axis=1)
            df_sub.indicator = to_name
            df_con = pd.concat([df_con, df_sub])
            # df_con.to_csv(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'temp.csv'))

            df_con = df_con[df_con['indicator'].isin([to_name, ti_name, twb_name])].pivot(
                index=ind_list, columns='indicator', values='value')

            if time_span == historical_years:
                if desirable_water_temp == None or desirable_water_temp == -3001.0:
                    desirable_water_temp = get_desiable_to(historical_ti=df_con[ti_name], dwt_thred_stat=dwt_thred_stat)
                df_con.loc[:, vul_name] = df_con.apply(
                    lambda row: add_vulnerability(row, dwt=desirable_water_temp,
                                                  wbtemp_threshold=wbtemp_threshold), axis=1)

                new_line = self.export_temp_var(desirable_water_temp,
                                                prefix=f'Plant ID: {plant_id} | Desired Outlet Water Temperature-{str(dwt_thred_stat)}',
                                                output_folder=self.final_assessment_dir)
                print(new_line)

            df_con = df_con.reset_index()
            df_con.date = pd.to_datetime(df_con.date)
            # df_con.to_csv(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'temp.csv'))

            if apply_regr is True:
                df_weekly = aggr_weekly(df=df_con,
                                        indicator=[switch_aggr_func(i)[0]
                                                   for i in df_con.columns if
                                                   i in [ti_name, to_name, twb_name, vul_name] and
                                                   'weekly' not in i],
                                        aggr_func=[switch_aggr_func(i)[1]
                                                   for i in df_con.columns if
                                                   i in [ti_name, to_name, twb_name, vul_name] and
                                                   'weekly' not in i])

                # Test interpolation of daily temperature from weekly temperature
                def debug():
                    if time_span == historical_years:
                        df_daily_test = pd.merge(df_con, df_weekly, on=['model', 'scenario', 'date'], how='outer')
                        df_daily_test['water_temperature_weekly'] = df_daily_test['water_temperature_weekly'].fillna(
                            method='bfill')
                        df_daily_test['outlet_water_temperature_weekly'] = df_daily_test[
                            'outlet_water_temperature_weekly'].fillna(method='bfill')
                        real_daily = df_daily_test['vulnerability']
                        date = df_daily_test['date']
                        df_daily_test = df_daily_test.drop(
                            columns=['water_temperature', 'outlet_water_temperature', 'vulnerability_weekly',
                                     'vulnerability', 'wet_bulb_temperature_weekly'])
                        df_daily_test.columns = [i.replace('_weekly', '') for i in df_daily_test.columns]
                        bc_weekly = df_daily_test.apply(lambda row: add_vulnerability(row, dwt=desirable_water_temp,
                                                                                      wbtemp_threshold=wbtemp_threshold),
                                                        axis=1)

                        df_test = pd.concat([date, real_daily, bc_weekly], axis=1)
                        df_test.columns = ['date', 'real', 'interpolate']
                        df_test = df_test[~pd.isna(df_test['interpolate'])]
                        df_test = df_test[~pd.isna(df_test['real'])]
                        diff_test = sum(df_test['interpolate'] - df_test['real']) / sum(df_test['real'])
                        print(f'The error for {plant_id} is {diff_test}')

                # debug()

                alr_weekly = [i for i in df_con.columns if
                              i in [ti_name, to_name, twb_name, vul_name] and 'weekly' in i]
                ind_list_weekly = [i for i in df_con.columns if i in ['date', 'model', 'scenario']]
                df_weekly = pd.merge(df_con[ind_list_weekly + [i for i in alr_weekly]], df_weekly, on=ind_list_weekly,
                                     how='inner')
                df_weekly.columns = [i.replace('_weekly', '') for i in df_weekly.columns]

                # Test whether use regression or not?
                def debug():
                    if time_span == historical_years:
                        # fig = px.scatter(df_weekly, 'water_temperature', 'vulnerability')
                        # fig.show()
                        bc_weekly = df_weekly.apply(
                            lambda row: add_vulnerability(row, dwt=desirable_water_temp,
                                                          wbtemp_threshold=wbtemp_threshold), axis=1)
                        bc_weekly = bc_weekly * 7
                        diff = sum(bc_weekly - df_weekly[vul_name]) / sum(df_weekly[vul_name])

                # debug()

            if time_span == projection_years:
                if apply_regr == True:
                    df_weekly.loc[:, vul_name] = \
                        df_weekly.apply(lambda row: fit_model(row=row, regr=regr, wbtemp_threshold=wbtemp_threshold),
                                        axis=1)
                else:
                    df_pesudo_weekly = df_con.sort_values(by='date')

                    def fillna_func(x):
                        m, s = x
                        df_sub = df_pesudo_weekly[
                            (df_pesudo_weekly['scenario'] == s) & (df_pesudo_weekly['model'] == m)]
                        df_sub[ti_name] = df_sub[ti_name].fillna(method='bfill')
                        df_sub[to_name] = df_sub[to_name].fillna(method='bfill')
                        return df_sub

                    df_sub_list = list(
                        map(fillna_func, list(itertools.product(waterTemp_models, waterTemp_scenarios))))
                    df_pesudo_weekly = pd.concat(df_sub_list)
                    df_pesudo_weekly.columns = [i.replace('_weekly', '') for i in df_pesudo_weekly.columns]
                    df_pesudo_weekly['vulnerability'] = df_pesudo_weekly.apply(
                        lambda row: add_vulnerability(row, dwt=desirable_water_temp,
                                                      wbtemp_threshold=wbtemp_threshold), axis=1)
                waterTemp_cs_cm = list(i for i in itertools.product(waterTemp_scenarios, waterTemp_models))
            elif time_span == historical_years:
                if apply_regr == True:
                    regr = get_regr(df_weekly)
                else:
                    regr = None
                waterTemp_cs_cm = [None]
            else:
                sys.exit('No valid time span input!')

            if apply_regr is True:
                plot_rc_df(show_plot, plant_id, ti_name.replace('_weekly', ''), to_name.replace('_weekly', ''),
                           twb_name.replace('_weekly', ''), wbtemp_threshold, time_span, df_con=df_weekly)
            else:
                plot_rc_df(show_plot, plant_id, ti_name, to_name, twb_name, wbtemp_threshold, time_span, df_con=df_con)

            def cal_aahd(x):
                if x is not None:
                    cs, cm = x
                    if apply_regr is True:
                        df_in = df_weekly[(df_weekly['scenario'] == cs) & (df_weekly['model'] == cm)]
                    else:
                        df_in = df_pesudo_weekly[
                            (df_pesudo_weekly['scenario'] == cs) & (df_pesudo_weekly['model'] == cm)]
                else:
                    if apply_regr is True:
                        df_in = df_weekly
                    else:
                        df_in = df_con
                df_in = df_in.dropna(subset=[vul_name])
                hd_days = np.sum(df_in[vul_name].values)
                tot_days = len(df_in['date'])
                if apply_regr is True:
                    aahd = hd_days / (tot_days * 7) * 52 * 7 * 24
                else:
                    aahd = hd_days / tot_days * 365 * 24
                    if time_span == projection_years:
                        df_error_coef = pd.read_excel(self.water_module_error_fp, engine='openpyxl', index_col=0)
                        error_coef = df_error_coef[df_error_coef['Plant ID'] == plant_id]['Error'].values
                        aahd = aahd * (1 - error_coef)
                return aahd

            aahd_pm = list(map(cal_aahd, waterTemp_cs_cm))
            return aahd_pm, regr, wbtemp_threshold, desirable_water_temp

        uncert = 90  # calculate the 10th and the 90th percentile of climate model ensemble as the uncertainties.

        # historical datasets
        df_wbtemp = pd.read_csv(switch_datasets('era5_1980-2019'), index_col=0)
        df_watertemp = pd.read_csv(switch_datasets('waterTemp_hist_all'), index_col=0)
        # No timespan configuration is needed for df_wbtemp and df_watertemp given that the inner join will be applied
        # to merge two datasets, which will result in the output data covers 1980 through 2010.
        time_span = historical_years
        aahd_bc_pm, regr, wbtemp_threshold, desirable_water_temp = main(df_wbtemp, df_watertemp, time_span,
                                                                        desirable_water_temp=desirable_water_temp,
                                                                        dwt_thred_stat=dwt_thred_stat)
        aahd_bc_pm = np.array(aahd_bc_pm)[~np.isnan(aahd_bc_pm)]
        med_aahd_bc = np.median(aahd_bc_pm)
        q95_aahd_bc = np.percentile(aahd_bc_pm, q=uncert)
        q5_aahd_bc = np.percentile(aahd_bc_pm, q=100 - uncert)
        max_aahd_bc = np.amax(aahd_bc_pm)
        min_aahd_bc = np.amin(aahd_bc_pm)
        avg_aahd_bc = np.mean(aahd_bc_pm)

        # projection
        if projection_years == [2030, 2069]:
            df_wbtemp_all = pd.read_csv(switch_datasets(x=f'gddp_2030-2070'), index_col=0)
        if projection_years == [2025, 2034] or projection_years == [2010, 2049]:
            df_wbtemp_all = pd.read_csv(switch_datasets(x=f'gddp_2010-2049'), index_col=0)
        df_wbtemp = df_wbtemp_all[
            (df_wbtemp_all['year'] >= projection_years[0]) & (df_wbtemp_all['year'] <= projection_years[1])]
        df_watertemp_all = pd.read_csv(switch_datasets(x='waterTemp_futu_all'), index_col=0)
        df_watertemp = df_watertemp_all[(df_watertemp_all['year'] >= projection_years[0])
                                        & (df_watertemp_all['year'] <= projection_years[1])]
        time_span = projection_years
        aahd_futu_pm, *_ = main(df_wbtemp, df_watertemp, time_span, desirable_water_temp=desirable_water_temp,
                                regr=regr, wbtemp_threshold=wbtemp_threshold)
        aahd_futu_pm = np.array(aahd_futu_pm)[~np.isnan(aahd_futu_pm)]
        med_aahd_futu = np.median(aahd_futu_pm)
        q95_aahd_futu = np.percentile(aahd_futu_pm, q=uncert)
        q5_aahd_futu = np.percentile(aahd_futu_pm, q=100 - uncert)
        max_aahd_futu = np.amax(aahd_futu_pm)
        min_aahd_futu = np.amin(aahd_futu_pm)
        avg_aahd_futu = np.mean(aahd_futu_pm)

        return med_aahd_bc, med_aahd_futu, q5_aahd_bc, q95_aahd_bc, max_aahd_bc, min_aahd_bc, avg_aahd_bc, \
               q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu

    def water_temperature_module(self, models, scenarios, plant_id, vulnerability_dataframe, historical_years,
                                 projection_years, desirable_water_temp, shutdown_water_temp, dwt_thred_stat,
                                 swt_thred_stat, regulatory_limits):
        """
        Assessment of water temperature-induced generation losses for once through-cooling plants.

        :param models: list, climate models.
        :param scenarios: list, climate scenarios.
        :param plant_id: int, plant reference id.
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors table.
        :param historical_years: list or tuple, start and end year of the baseline/historical period.
        :param projection_years: list or tuple, start and end year of the projection period.
        :param desirable_water_temp: int or float, design water temperature.
        :param shutdown_water_temp: int or float, shutdown water temperature.
        :param dwt_thred_stat: float, string -> 'mean', float -> percentile/quantile value between 0 and 1.
        :param swt_thred_stat: string, float or -3001.0, string -> 'mean', float -> percentile/quantile value bewteen 0 and 1; -3001.0 -> the shutdown water temperature will be the value when the vulnerability fator is equal to 1.
        :param regulatory_limits: -3001.0 or int, -3001.0 -> no regulatory discharge limits; int -> reference id of the regulatory limit sheet in the vulnerability factor excel workbook.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disrupted hours.
        """

        def cal_dwt(df_bc_daily, dwt_thred_stat):
            if isinstance(dwt_thred_stat, (int, float, complex)):
                desirable_water_temp = np.percentile(
                    df_bc_daily[~df_bc_daily['water_temp_C'].isnull()]['water_temp_C'], dwt_thred_stat)
            elif dwt_thred_stat == 'mean':
                desirable_water_temp = np.mean(df_bc_daily[~df_bc_daily['water_temp_C'].isnull()]['water_temp_C'])
            else:
                raise NotImplementedError
            return desirable_water_temp

        def cal_swt(df_bc_daily, vulnerability_dataframe, desirable_water_temp, swt_thred_stat):
            if swt_thred_stat is None or swt_thred_stat == -3001.0:
                shutdown_water_temp = \
                    vulnerability_dataframe[vulnerability_dataframe['Vulnerability'] == 1]['Threshold'].values[
                        0] + desirable_water_temp
            elif isinstance(swt_thred_stat, (int, float, complex)):
                shutdown_water_temp = np.percentile(
                    df_bc_daily[~df_bc_daily['water_temp_C'].isnull()]['water_temp_C'], swt_thred_stat)
            elif swt_thred_stat == 'mean':
                shutdown_water_temp = np.mean(df_bc_daily[~df_bc_daily['water_temp_C'].isnull()]['water_temp_C'])
            else:
                raise NotImplementedError
            return shutdown_water_temp

        def recal_dfVul(vulnerability_dataframe, desirable_water_temp, regulatory_limits):
            dfVul_regLim = pd.read_excel(self.vulnerability_factors_fp, engine='openpyxl',
                                         sheet_name=f'WT_REG_S{str(int(regulatory_limits))}')
            dfVul_regLim = dfVul_regLim.sort_values(by='Threshold')
            dfVul_noRegLim = vulnerability_dataframe.copy().sort_values(by='Threshold')
            dfVul_noRegLim['Threshold'] = vulnerability_dataframe['Threshold'] + desirable_water_temp
            dfVul_noRegLim_notes = list(dfVul_noRegLim['Threshold Notes'].unique())[0]
            dfVul_noRegLim['Threshold Notes'] = list(dfVul_regLim['Threshold Notes'].unique())[0]

            upper_regLim, lower_regLim = dfVul_regLim['Threshold'].max(), dfVul_regLim[
                'Threshold'].min()  # the corresponding Vulnerability is 1

            def main(dfVul_regLim, row):
                if row['Threshold'] >= upper_regLim:
                    return 1.0
                elif row['Threshold'] <= lower_regLim:
                    return row['Vulnerability']
                else:
                    return self.find_vulnerability_threshold_asending(df=dfVul_regLim, cut_off_threshold=0.0,
                                                                      threshold_of_interest=row['Threshold'])

            dfVul_regLim_new = dfVul_noRegLim.copy()
            dfVul_regLim_new['Vulnerability'] = dfVul_noRegLim.apply(lambda row: main(dfVul_regLim, row), axis=1)
            dfVul_merge = (pd.concat([dfVul_regLim_new, dfVul_noRegLim]).groupby('Threshold').max()).reset_index()
            swt_regLim = min(dfVul_merge[dfVul_merge['Vulnerability'] == 1]['Threshold'].values.min(), upper_regLim)
            dfVul_merge['Threshold'] = dfVul_merge['Threshold'] - desirable_water_temp
            dfVul_merge = dfVul_merge[dfVul_merge['Threshold'] <= round(upper_regLim - desirable_water_temp + 0.5)]
            dfVul_merge['Threshold Notes'] = dfVul_noRegLim_notes
            return dfVul_merge, swt_regLim

        apply_regr = False
        aahd_bc_pm = []
        aahd_futu_pm = []
        scenarios = [i for i in scenarios for id_ in self.scenario_id_list if id_[0] in i]

        # Read daily water temperature
        for ff in os.walk(self.tpp_water_temp_folder):
            file_path = [os.path.join(ff[0], f) for f in ff[2]
                         if 'TPP_' + str(plant_id) + '_waterTemperature_mergedV2_' in f][0]
        df_bc_daily_all = pd.read_csv(file_path, index_col=0)
        df_bc_daily_all['date'] = pd.to_datetime(df_bc_daily_all['date'])
        df_bc_daily = df_bc_daily_all[(df_bc_daily_all['date'].dt.year >= historical_years[0]) &
                                      (df_bc_daily_all['date'].dt.year <= historical_years[1])]
        df_bc_daily['date'] = pd.to_datetime(df_bc_daily['date'], format='%Y-%m-%d')
        df_bc_daily['water_temp_C'] = df_bc_daily['value'] - 273.15
        # Refine desirable_water_temp, shutdown_water_temp...
        if desirable_water_temp is None or desirable_water_temp == -3001.0:
            desirable_water_temp = cal_dwt(df_bc_daily, dwt_thred_stat)
        if shutdown_water_temp is None or shutdown_water_temp == -3001.0:
            shutdown_water_temp = cal_swt(df_bc_daily, vulnerability_dataframe, desirable_water_temp, swt_thred_stat)
        # Refine vulnerability dataframe
        if regulatory_limits != -3001.0:
            vulnerability_dataframe, swt_regLim = recal_dfVul(vulnerability_dataframe, desirable_water_temp,
                                                              regulatory_limits)
            shutdown_water_temp = min(shutdown_water_temp, swt_regLim)
        new_line_dwt = self.export_temp_var(desirable_water_temp,
                                            prefix=f'Plant ID: {plant_id} | Desired Water Temperature-{str(dwt_thred_stat)}',
                                            output_folder=self.final_assessment_dir)
        new_line_swt = self.export_temp_var(shutdown_water_temp,
                                            prefix=f'Plant ID: {plant_id} | Shutdown Water Temperature-{str(swt_thred_stat)}',
                                            output_folder=self.final_assessment_dir)
        print(f'{new_line_dwt}\n{new_line_swt}')

        # Calculate weekly efficiency loss
        df_bc = self.histDailyWaterTemp_to_weeklyEfficiencyLoss(df_bc_daily, vulnerability_dataframe,
                                                                desirable_water_temp, shutdown_water_temp)
        df_bc = df_bc.dropna()
        anual_loss_bc = np.sum(df_bc['weekly_efficiency_loss'].values) / (len(df_bc['date']) * 7) * 52 * 7 * 24
        aahd_bc_pm.append(anual_loss_bc)
        # print(aahd_bc_pm)
        # plot_weekly(df=df_bc, loc_id=plant_id)

        if apply_regr is True:
            df_regr = df_bc[(df_bc['weekly_water_temp_C'] > desirable_water_temp) & (
                    df_bc['weekly_water_temp_C'] < shutdown_water_temp)]
            df_regr = df_regr.dropna()
            regr, *_ = self.regressor(X=df_regr['weekly_water_temp_C'].values.reshape(-1, 1),
                                      y=df_regr['weekly_efficiency_loss'].values.reshape(-1, 1), model='ols')

        # # Test whether use regression or not?
        def debug():
            # fig = px.scatter(df_regr, x='weekly_water_temp_C', y='weekly_efficiency_loss')
            # fig.show()
            def add_vulnarability(water_temp, dwt, swt):
                if np.isnan(water_temp):
                    toi_vul = np.nan
                elif water_temp >= swt:
                    toi_vul = 1.0
                elif water_temp <= dwt:
                    toi_vul = 0
                else:
                    toi = water_temp - dwt  # for once through
                    toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe,
                                                                         cut_off_threshold=0.0,
                                                                         threshold_of_interest=toi)
                if toi_vul < 0:
                    toi_vul = 0

                return toi_vul

            weekly_test = df_regr['weekly_water_temp_C'].apply(
                lambda row: add_vulnarability(row, desirable_water_temp, shutdown_water_temp))
            weekly_test = weekly_test * 7
            diff_test = sum(weekly_test - df_regr['weekly_efficiency_loss']) / sum(
                df_regr['weekly_efficiency_loss'])
            # -0.0019
            print(f'The error for {plant_id} is {diff_test}')

        # debug()

        # def add_vulnarability(water_temp, dwt, swt):
        #     if np.isnan(water_temp):
        #         toi_vul = np.nan
        #     elif water_temp >= swt:
        #         toi_vul = 1.0
        #     elif water_temp <= dwt:
        #         toi_vul = 0
        #     else:
        #         toi = water_temp - dwt  # for once through
        #         toi_vul = self.find_vulnerability_threshold_asending(df=vulnerability_dataframe,
        #                                                              cut_off_threshold=0.0,
        #                                                              threshold_of_interest=toi)
        #     if toi_vul < 0:
        #         toi_vul = 0
        #
        #     return toi_vul
        #
        # df_weekly_copy = df_bc.copy()
        # df_weekly_copy['vulnerability'] = df_weekly_copy['weekly_water_temp_C'] \
        #     .apply(lambda row: add_vulnarability(row, desirable_water_temp, shutdown_water_temp))
        # diff = sum(df_weekly_copy.vulnerability * 7 - df_bc.weekly_efficiency_loss) / sum(df_bc.weekly_efficiency_loss)

        for m, s in itertools.product(models, scenarios):
            if apply_regr is True:
                df_futu = self.futuWeekly_to_efficiencyLoss(regr, plant_id, m, s, vulnerability_dataframe,
                                                            historical_years,
                                                            projection_years, desirable_water_temp, shutdown_water_temp)

            else:
                df_futu = self.futu_loss(plant_id, vulnerability_dataframe, projection_years, desirable_water_temp,
                                         shutdown_water_temp, m, s)
                df_futu['weekly_efficiency_loss'] = df_futu['weekly_efficiency_loss'] * 7

            df_futu = df_futu.dropna()
            anual_loss_futu = np.sum(df_futu['weekly_efficiency_loss'].values) / (
                    len(df_futu['date']) * 7) * 52 * 7 * 24
            aahd_futu_pm.extend([anual_loss_futu])

        aahd_bc_pm = np.array(aahd_bc_pm)[~np.isnan(aahd_bc_pm)]
        aahd_futu_pm = np.array(aahd_futu_pm)[~np.isnan(aahd_futu_pm)]
        if apply_regr is not True:
            df_error_coef = pd.read_excel(self.water_module_error_fp, engine='openpyxl', index_col=0)
            error_coef = df_error_coef[df_error_coef['Plant ID'] == plant_id]['Error'].values
            aahd_futu_pm = aahd_futu_pm * (1 - error_coef)

        med_aahd_bc = np.median(aahd_bc_pm)
        med_aahd_futu = np.median(aahd_futu_pm)
        uncert = 90
        q95_aahd_bc = np.percentile(aahd_bc_pm, q=uncert)
        q5_aahd_bc = np.percentile(aahd_bc_pm, q=100 - uncert)
        max_aahd_bc = np.amax(aahd_bc_pm)
        min_aahd_bc = np.amin(aahd_bc_pm)
        avg_aahd_bc = np.mean(aahd_bc_pm)
        q95_aahd_futu = np.percentile(aahd_futu_pm, q=uncert)
        q5_aahd_futu = np.percentile(aahd_futu_pm, q=100 - uncert)
        max_aahd_futu = np.amax(aahd_futu_pm)
        min_aahd_futu = np.amin(aahd_futu_pm)
        avg_aahd_futu = np.mean(aahd_futu_pm)
        return med_aahd_bc, med_aahd_futu, q5_aahd_bc, q95_aahd_bc, max_aahd_bc, min_aahd_bc, avg_aahd_bc, \
               q5_aahd_futu, q95_aahd_futu, max_aahd_futu, min_aahd_futu, avg_aahd_futu

    def water_stress_module(self, lat, lon, pln, uid, base_geodataframe, futu_geodataframe, turb, cool,
                            vulnerability_dataframe, year=2040):
        """
        Assessment of the water stress-induced generation losses.

        :param lat: float, latitude
        :param lon: float, longitude
        :param pln: string, plant name
        :param uid: int, unique plant id
        :param base_geodataframe: pandas dataframe, water stress over the baseline period.
        :param futu_geodataframe: pandas dataframe, water stress over the projection period.
        :param turb: string, turbine type
        :param cool: string, cooling system
        :param vulnerability_dataframe: pandas dataframe, vulnerability factors table.
        :param year: int, the projection year to be assessed.
        :return: list, a list of statistics, i.e., median, max, min, p95, p5 of annual disrupted hours.
        """
        print(turb, cool)
        dfVul = vulnerability_dataframe
        dfVul = dfVul[(dfVul['Turbine'] == turb) & (dfVul['Cooling'] == cool)]
        dfu = pd.DataFrame({'plant': [pln], 'lat': [lat], 'lon': [lon]})
        dfp = gp.GeoDataFrame(dfu, geometry=gp.points_from_xy(dfu.lon, dfu.lat), crs='EPSG:4326')
        dfp_sj = gp.sjoin(dfp, base_geodataframe, how='left', op='intersects')
        dfp_sj.drop('index_right', axis=1, inplace=True)
        dfp_sj = gp.sjoin(dfp_sj, futu_geodataframe, how='left', op='intersects')
        wscat21b = dfp_sj['BWS_cat'].values[0]
        ws21b = dfp_sj['BWS'].values[0]
        ut21b = dfp_sj['WITHDRAWAL'].values[0]
        ct21b = dfp_sj['CONSUMPTIO'].values[0]
        bt21b = dfp_sj['BT'].values[0]
        ba21b = dfp_sj['BA'].values[0]
        ut21f4024 = dfp_sj[f'ut{str(year)[2:]}24tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        ut21f4028 = dfp_sj[f'ut{str(year)[2:]}28tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        ut21f4038 = dfp_sj[f'ut{str(year)[2:]}38tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        bt21f4024 = dfp_sj[f'bt{str(year)[2:]}24tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        bt21f4028 = dfp_sj[f'bt{str(year)[2:]}28tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        bt21f4038 = dfp_sj[f'bt{str(year)[2:]}38tr'].values[0] * dfp_sj['Area_km2'].values[0] * 1000000
        ws21f4024 = dfp_sj[f'ws{str(year)[2:]}24tr'].values[0]
        ws21f4028 = dfp_sj[f'ws{str(year)[2:]}28tr'].values[0]
        ws21f4038 = dfp_sj[f'ws{str(year)[2:]}38tr'].values[0]
        ba21f4024 = ut21f4024 / ws21f4024
        ba21f4028 = ut21f4028 / ws21f4028
        ba21f4038 = ut21f4038 / ws21f4038
        # print("{:,}\n{:,}\n{:,}\n{:,}".format(ut21b, ut21f4024, ut21f4028, ut21f4038))
        # print("{:,}\n{:,}\n{:,}\n{:,}".format(ba21b, ba21f4024, ba21f4028, ba21f4038))
        print("{:,}, {:,}, {:,}, {:,}".format(ws21b, ws21f4024, ws21f4028, ws21f4038))
        ws_hist = ws21b
        med_ws_futu = np.median([ws21f4024, ws21f4028, ws21f4038])
        ba_hist = ba21b
        med_ba_futu = np.median([ba21f4024, ba21f4028, ba21f4038])
        uncert = 90
        q95_ba_futu = np.percentile([ba21f4024, ba21f4028, ba21f4038], q=uncert)
        q5_ba_futu = np.percentile([ba21f4024, ba21f4028, ba21f4038], q=100 - uncert)
        max_ba_futu = np.amax(np.array([ba21f4024, ba21f4028, ba21f4038]))
        min_ba_futu = np.amin(np.array([ba21f4024, ba21f4028, ba21f4038]))
        avg_ba_futu = np.mean(np.array([ba21f4024, ba21f4028, ba21f4038]))
        ws_max = np.max([ws_hist, ws21f4024, ws21f4028, ws21f4038])
        med_ba_chg = (med_ba_futu - ba_hist) / ba_hist
        q95_ba_chg = (q95_ba_futu - ba_hist) / ba_hist
        q5_ba_chg = (q5_ba_futu - ba_hist) / ba_hist
        max_ba_chg = (max_ba_futu - ba_hist) / ba_hist
        min_ba_chg = (min_ba_futu - ba_hist) / ba_hist
        avg_ba_chg = (avg_ba_futu - ba_hist) / ba_hist

        # Save supply reduction rate
        test_df = pd.DataFrame(zip(['med_ba_chg', 'q95_ba_chg', 'q5_ba_chg', 'max_ba_chg', 'min_ba_chg', 'avg_ba_chg'],
                                   [med_ba_chg, q95_ba_chg, q5_ba_chg, max_ba_chg, min_ba_chg, avg_ba_chg]),
                               columns=['variable', 'value'])
        test_df['plant_id'] = uid
        test_df = test_df[['plant_id', 'variable', 'value']]
        self.test_out.append(test_df)

        print(uid, ws_max)

        if ws_max >= 0.4 and med_ba_chg <= 0:
            med_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                      threshold_of_interest=np.abs(
                                                                          med_ba_chg)) * 365 * 24
            q95_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                      threshold_of_interest=np.abs(
                                                                          q95_ba_chg)) * 365 * 24
            q5_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                     threshold_of_interest=np.abs(q5_ba_chg)) * 365 * 24
            max_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                      threshold_of_interest=np.abs(
                                                                          max_ba_chg)) * 365 * 24
            min_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                      threshold_of_interest=np.abs(
                                                                          min_ba_chg)) * 365 * 24
            avg_aahd_chg = self.find_vulnerability_threshold_asending(df=dfVul, cut_off_threshold=0.0,
                                                                      threshold_of_interest=np.abs(
                                                                          avg_ba_chg)) * 365 * 24
        else:
            med_aahd_chg = 0.0
            q95_aahd_chg = 0.0
            q5_aahd_chg = 0.0
            max_aahd_chg = 0.0
            min_aahd_chg = 0.0
            avg_aahd_chg = 0.0
        print(med_aahd_chg, med_aahd_chg / (365 * 24), wscat21b)

        # print(dfp_sj.index_right)
        return 0.0, med_aahd_chg, 0.0, 0.0, 0.0, 0.0, 0.0, q5_aahd_chg, q95_aahd_chg, max_aahd_chg, min_aahd_chg, avg_aahd_chg

    def hours2generation(self, row, sti, from_excel=None):
        """
        Convert annual disrupted hours, i.e., the output of each module function, into annual generation losses by multiple with installed capacity of each plant unit.

        :param row: a row of a pandas dataframe.
        :param sti: string, stats name, anyone among 'med', 'q5', 'q95', 'max', 'min', and 'avg'.
        :param from_excel: boolean, whether the input row is parsed from an interim excel file.
        :return: float, generation losses, i.e., disrupted hours * installed capacity.
        """
        if from_excel is True:
            hrs = np.fromstring((row['disruption hours'])[1:-1], dtype=float, sep=' ').tolist()
        else:
            hrs = np.array(row['disruption hours'])
        cap = row['capacity']
        out = hrs[sti] * cap if hrs[sti] != -9999 else -9999
        return out

    def thermal_assess(self):
        """
        Execute thermal assessment.
        """

        def unit_data_aggregator_and_printer(df, aggregateBy, custom_text_to_print, stat_list=None):
            """
            Aggregate unit-level results into the plant or country level.

            :param df: pandas dataframe, has the following columns [countries, uniqueids, plants, capacities, disrupted_hours].
            :param aggregateBy: string, 'plant' or 'Country', aggregate by plant or country.
            :param custom_text_to_print: string,
            :param stat_list: list, a subset of ['med', 'q5', 'q95', 'max', 'min', 'avg'], which statistics to be exported.
            :return: pandas dataframe countries, columns in the sequence of plant id, plant name, disrupted hours, plant capacity, statistic, baseline generation losses, future generation losses
            """

            def switch_stat(x):
                switch = {
                    'med': [0, 1],
                    'q5': [2, 7],
                    'q95': [3, 8],
                    'max': [4, 9],
                    'min': [5, 10],
                    'avg': [6, 11]
                }
                return switch[x]

            dfOut = pd.DataFrame(columns=['Group', 'Capacity', 'Statistic Type', 'Baseline Value', 'Projection Value'])

            if stat_list is None:
                stat_list = ['med', 'q5', 'q95', 'max', 'min', 'avg']

            for stat in stat_list:
                # print(df.columns)
                # print(df.head)
                df['gen loss bc ' + stat] = df.apply(lambda row: self.hours2generation(row, sti=switch_stat(stat)[0]),
                                                     axis=1)
                df['gen loss fu ' + stat] = df.apply(lambda row: self.hours2generation(row, sti=switch_stat(stat)[1]),
                                                     axis=1)
                dfworknew = df.drop(['disruption hours', 'unique id'], axis=1)
                dfworknew['capacity'] = pd.to_numeric(df["capacity"])
                if aggregateBy == 'plant':
                    ddd = pd.pivot_table(df.drop(['disruption hours'], axis=1), index=['unique id', 'plant'],
                                         aggfunc=np.sum).reset_index()
                dfPivot = pd.pivot_table(dfworknew, index=aggregateBy, aggfunc=np.sum).reset_index()
                for p in dfPivot[aggregateBy].values:
                    print(p)
                    dftemp = dfPivot[dfPivot[aggregateBy] == p]
                    cap = dftemp['capacity'].values[0]
                    gl_bc_stat = dftemp['gen loss bc ' + stat].values[0]
                    gl_fu_stat = dftemp['gen loss fu ' + stat].values[0]

                    if gl_fu_stat == -9999 or gl_bc_stat == -9999:
                        c = -9999
                    elif gl_bc_stat != 0:
                        c = (gl_bc_stat - gl_fu_stat) / gl_bc_stat
                    elif gl_fu_stat == 0:
                        c = 0.0
                    else:
                        c = np.inf
                    if c == 0.0:
                        changeText = "No change"
                    elif c > 0.0:
                        changeText = u'\u25BC'
                    else:
                        changeText = u'\u25B2'
                    # PRINT INDIVIDUAL PLANT RESULTS
                    print('%s:\n' % custom_text_to_print,
                          'Current-{}:{:.1f} | Future-{}:{:.1f}'.format(stat, gl_bc_stat, stat, gl_fu_stat))
                    if c < 0.0 or c == np.inf:
                        print(colored(changeText, 'red'), colored("{:.1%}".format(np.abs(c)), 'red'))
                    elif changeText == 'No change':
                        print(colored(changeText, 'grey'))
                    else:
                        print(colored(changeText, 'green'), colored("{:.1%}".format(c), 'green'))

                    if aggregateBy == 'plant':
                        p = ddd[ddd['plant'] == p]['unique id'].values[0]

                    dfOut = dfOut.append(pd.Series([p, cap, stat, gl_bc_stat, gl_fu_stat], index=dfOut.columns),
                                         ignore_index=True)

            # print(dfOut)
            return dfOut

        def futu_vs_ba(**kwargs):
            """
            Calculate the following metrics, i.e., proj-base, (proj-base)/base, proj/ideal, base/ideal.
            """
            if 'df' in kwargs:
                df = kwargs['df']
            elif 'file_path' in kwargs:
                df = pd.read_csv(kwargs['file_path'], index_col=0)
            else:
                pass

            if self.scenario_id_list != []:
                df['proj-base'] = df['Projection Value'] - df['Baseline Value']
                df['(proj-base)/base'] = (df['Projection Value'] - df['Baseline Value']) / df['Baseline Value']
                df['proj/ideal'] = df['Projection Value'] / (df['Capacity'] * 365 * 24)
                df['base/ideal'] = df['Baseline Value'] / (df['Capacity'] * 365 * 24)
            else:
                df['proj-base'] = -9999
                df['(proj-base)/base'] = -9999
                df['proj/ideal'] = -9999
                df['base/ideal'] = df['Baseline Value'] / (df['Capacity'] * 365 * 24)

            if 'output_path' in kwargs:
                df.to_csv(kwargs['output_path'])
                print('Find your putput here: {}'.format(kwargs['output_path']))

            return df

        def csv_to_excel(**kwargs):
            """
            Convert csv to excel.
            """
            if 'df' in kwargs:
                df = kwargs['df']
            elif 'file_path' in kwargs:
                df = pd.read_csv(kwargs['file_path'], index_col=0)
                excel_path = os.path.splitext(kwargs['file_path'])[0] + '.xlsx'
            else:
                pass

            if 'group' in kwargs and kwargs['group'] == 'plant':
                tpp_info = pd.read_excel(self.tpp_working_fp,
                                         sheet_name='master', engine='openpyxl')
                left = tpp_info[['Power Plant Name', 'Uloc ID', 'Country', 'Turbine regroup', 'Cooling system']] \
                    .groupby(['Power Plant Name', 'Uloc ID', 'Country'], as_index=False) \
                    .agg({'Turbine regroup': set,
                          'Cooling system':
                              lambda x: set(['Recirculating' if 'Recirculating' in str(i) else str(i) for i in x])})
                left.columns = ['Power Plant Name', 'Group', 'Country', 'Turbine regroup', 'Cooling system']
                df = pd.merge(left, df, on=['Group'], how='outer')

            if 'output_path' in kwargs:
                df.to_excel(kwargs['output_path'], engine='openpyxl')
            else:
                df.to_excel(excel_path, engine='openpyxl')
                kwargs['output_path'] = excel_path

            print('Find your output here: {}'.format(kwargs['output_path']))

            return df

        def export_assessment(df_out_dict, file_name, run_code):
            """
            Export final assessment reports as excel tables.

            :param df_out_dict: dictionary, generation losses aggregated by county and plant, i.e., {'country': dfCtry, 'plant': dfPlt}, dfCtry and dfPlt are the output of each assessor.
            :param file_name: string, prefix of final assessment reports (csv and excel files).
            :param run_code: int, reference code of the run.
            :return: None
            """
            # New empty excel with tabs corresponding to aggregation scenarios
            out_dir = self.final_assessment_dir
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            file_path = {'country': os.path.join(out_dir, file_name + '_ByCountry.csv'),
                         'plant': os.path.join(out_dir, file_name + '_ByPlant.csv')}

            def export_csv(x):
                if not os.path.isfile(file_path[x]):
                    df_out = pd.DataFrame(
                        columns=['Group', 'Capacity', 'Statistic Type', 'Baseline Value', 'Projection Value',
                                 'Risk Type', 'Run Code', 'proj-base', '(proj-base)/base', 'proj/ideal', 'base/ideal'])
                    df_out.to_csv(file_path[x])
                df_out = df_out_dict[x]
                df_out['Run Code'] = run_code
                df_out = futu_vs_ba(df=df_out)
                df_out.to_csv(file_path[x], mode='a', header=False)

            list([export_csv(i) for i in df_out_dict])
            list([csv_to_excel(group=group, file_path=fp) for group, fp in file_path.items()])

            print('\nThe final assessment is saved here: {}'.format(file_path))

        def drought_assessor():
            """
            Execute drought assessment.

            :return: a list of pandas dataframe, dfCtry -> output aggregated by countries, dfPlt -> output aggregated by plants.
            """
            print('\nNow assessing drought risks...')
            # # 1.1 SPEI EXPLORATION
            # tppID = 1
            # spei_fns = ['%s_obs.csv' % tppID, '%s_cmip5.csv' % tppID]  # ['historical observations', 'future projections']
            # drought_test_obs_fp = os.path.join(spei_data_folder, spei_fns[0])
            # drought_test_fut_fp = os.path.join(spei_data_folder, spei_fns[1])
            # spei_explorer(obs_fp=drought_test_obs_fp, fut_fp=drought_test_fut_fp,
            #               his_eval_yrs=his_eval_yrs, fut_eval_yrs=fut_eval_yrs)
            # 1.2 SPEI ASSESS
            temp_unit_type = ''
            temp_unit_data = ''
            countries = []
            plants = []
            capacities = []
            disrupted_hours = []
            uniqueids = []

            dfTppLoc = pd.read_excel(self.tpp_locs_fp, sheet_name='tpp_locs', engine='openpyxl')
            uid_list = [i for i in dfTppLoc.loc[dfTppLoc['Source Water'] == 'Y', 'Unique ID'].values]

            for pln, uid, ctry, unit, turb, cool, cap, cotech in zip(plns[:], uids[:], ctrys[:], units[:], turbs[:],
                                                                     cools[:], caps[:], cotechs[:]):
                print('\n', ctry, '|', pln, '| Location ID:', uid, '| Unit:', int(unit),
                      '| Turbine:', turb, '| Cooling:', cool, '| Cooling technology:', cotech,
                      'is now being assessed...')
                if turb == 'Diesel' or turb == 'Unknown':
                    turb = 'Steam'
                unit_type = '%s_%s_%s_%s' % (uid, turb, cool, cotech)
                if unit_type == temp_unit_type:
                    out = temp_unit_data
                    # print(out)
                else:
                    if not (uid in uid_list):
                        out = [0.0] * 12
                    else:
                        if not (cool in ['Wet', 'Recirculating', 'Once-through']):
                            out = [0.0] * 12
                        else:
                            if 'Recirculating' in str(cotech):
                                dfVulDrought = dfVul[
                                    (dfVul['Type'] == 'Drought') & (dfVul['Turbine'] == turb) & (
                                            dfVul['Cooling'] == 'Recirculating')]
                            elif 'Once-through' in str(cotech):
                                dfVulDrought = dfVul[
                                    (dfVul['Type'] == 'Drought') & (dfVul['Turbine'] == turb) & (
                                            dfVul['Cooling'] == 'Once-through')]
                            else:
                                raise NotImplementedError
                            spei_threshold = dfThd[
                                (dfThd['Type'] == 'Drought') & (dfThd['Parameter Name'] == 'spei_threshold')
                                & (dfThd['Scenario Group'] == self.thd_group_code)]['Value'].values[0]
                            obs_fp = os.path.join(self.spei_data_folder, '%s_obs.csv' % uid)
                            fut_fp = os.path.join(self.spei_data_folder, '%s_cmip5.csv' % uid)
                            out = self.drought_module(obs_fp=obs_fp, fut_fp=fut_fp,
                                                      spei_threshold=spei_threshold,
                                                      vulnerability_dataframe=dfVulDrought,
                                                      historical_years=self.his_eval_yrs,
                                                      projection_years=self.fut_eval_yrs)
                    temp_unit_data = out
                    temp_unit_type = unit_type
                    # print(out)
                countries.append(ctry)
                plants.append(pln)
                capacities.append(cap)
                disrupted_hours.append(np.array(out))
                uniqueids.append(uid)
            dfOut = pd.DataFrame(data=np.transpose([countries, uniqueids, plants, capacities, disrupted_hours]),
                                 columns=['country', 'unique id', 'plant', 'capacity', 'disruption hours'])
            dfCtry = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='country',
                                                      custom_text_to_print='Annual potential generation losses (MWh) due to droughts')
            dfPlt = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='plant',
                                                     custom_text_to_print='Annual potential generation losses (MWh) due to droughts')

            dfCtry['Risk Type'] = 'drought'
            dfPlt['Risk Type'] = 'drought'
            return dfCtry, dfPlt

        def flood_assessor():
            """
            Execute flood assessment.

            :return: a list of pandas dataframe, dfCtry -> output aggregated by countries, dfPlt -> output aggregated by plants.
            """
            print('\nNow assessing flood risks...')
            temp_unit_type = ''
            temp_unit_data = ''
            countries = []
            plants = []
            capacities = []
            disrupted_hours = []
            uniqueids = []
            for pln, uid, ctry, unit, turb, cool, cap, lat, lon in zip(plns[:], uids[:], ctrys[:], units[:],
                                                                       turbs[:], cools[:], caps[:], lats[:], lons[:]):
                print('\n', ctry, '|', pln, '| Location ID:', uid, '| Unit:', int(unit),
                      '| Turbine:', turb, '| Cooling:', cool, 'is now being assessed...')
                if turb == 'Diesel' or turb == 'Unknown':
                    turb = 'Steam'
                unit_type = '%s_%s_%s' % (uid, turb, cool)
                if unit_type == temp_unit_type:
                    out = temp_unit_data
                    # print(out)
                else:
                    try:
                        dfVulFlood = dfVul[
                            (dfVul['Type'] == 'Flood') & (dfVul['Turbine'] == turb) & (dfVul['Cooling'] == cool)]
                        buffer, design_protection = dfThd[(dfThd['Type'] == 'Flood') &
                                                          (dfThd['Scenario Group'] == self.thd_group_code) &
                                                          (dfThd['Parameter Name'].isin(
                                                              ['buffer', 'design_protection']))]['Value']
                        out = self.flood_module(lat=lat, lon=lon, folder=self.flood_data_folder,
                                                buffer=buffer,
                                                # in arc degree; 0.001 is roughly 100m; this is to overcome rounding error
                                                year=self.flood_year, design_protection=design_protection,
                                                vulnerability_dataframe=dfVulFlood)
                        temp_unit_data = out
                        temp_unit_type = unit_type
                    except:
                        print(colored('Not in any flood zone', 'green'))
                        out = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                        temp_unit_data = out
                        temp_unit_type = unit_type
                        pass
                # print(out)
                countries.append(ctry)
                plants.append(pln)
                capacities.append(cap)
                disrupted_hours.append(np.array(out))
                uniqueids.append(uid)
            dfOut = pd.DataFrame(data=np.transpose([countries, uniqueids, plants, capacities, disrupted_hours]),
                                 columns=['country', 'unique id', 'plant', 'capacity', 'disruption hours'])

            dfCtry = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='country',
                                                      custom_text_to_print='Annual potential generation losses (MWh) due to floods')
            dfPlt = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='plant',
                                                     custom_text_to_print='Annual potential generation losses (MWh) due to floods')

            dfCtry['Risk Type'] = 'flood'
            dfPlt['Risk Type'] = 'flood'
            # print(dfCtry, dfPlt)
            return dfCtry, dfPlt

        def air_temperature_assessor(models=None):
            """
            Execute air temperature assessment.

            :return: a list of pandas dataframe, dfCtry -> output aggregated by countries, dfPlt -> output aggregated by plants.
            """
            print('\nNow assessing extreme and chronic air temperature increase risks...')
            if models is None:
                fileNames = [f[:-4] for f in os.listdir(self.climate_data_folder)
                             if not f.endswith('.zip') and os.path.isfile(os.path.join(self.climate_data_folder, f))]
                models = list(set([f.split('_')[4] for f in fileNames]))
            scenarios = ['historical', 'rcp45', 'rcp85'] if self.scenario_id_list != [] else ['historical']
            temp_unit_type = ''
            temp_unit_data = ''
            countries = []
            plants = []
            capacities = []
            disrupted_hours = []
            uniqueids = []
            for pln, uid, ctry, unit, turb, cool, cap, gentech, lat, lon in zip(plns[:], uids[:], ctrys[:], units[:],
                                                                                turbs[:], cools[:], caps[:],
                                                                                gentechs[:],
                                                                                lats[:], lons[:]):
                print('\n', ctry, '|', pln, '| Location ID:', uid, '| Unit:', int(unit),
                      '| Turbine:', turb, '| Cooling:', cool, '| Generation technology:', gentech,
                      'is now being assessed...')
                if turb == 'Diesel' or turb == 'Unknown':
                    turb = 'Steam'
                unit_type = '%s_%s_%s' % (uid, turb, cool)
                if unit_type == temp_unit_type:
                    out = temp_unit_data
                else:
                    if (turb == 'Steam') and (cool in ['Wet', 'Recirculating', 'Once-through']):
                        out = [0.0] * 12
                    else:
                        # For CCGT, the efficiency loss values for gas turbines can be applied to steam turbines
                        if (gentech == 'CCGT') and (turb == 'Steam'):
                            turb = 'Gas'
                            cool = 'Not Applicable'
                            unit_type = '%s_%s_%s' % (uid, turb, cool)

                        dfVulHeatwaveAir = dfVul[
                            (dfVul['Type'] == 'Air temperature') & (dfVul['Turbine'] == turb) & (
                                    dfVul['Cooling'] == cool)]
                        desirable_air_temp, shutdown_air_temp = dfThd[(dfThd['Type'] == 'Air temperature') &
                                                                      (dfThd['Parameter Name'].isin(
                                                                          ['desirable_air_temp', 'shutdown_air_temp']))
                                                                      & (dfThd[
                                                                             'Scenario Group'] == self.thd_group_code)][
                            'Value']
                        dat_thred_stat, sat_thred_stat = dfThd[(dfThd['Type'] == 'Air temperature') &
                                                               (dfThd['Parameter Name'].isin(
                                                                   ['desirable_air_temp', 'shutdown_air_temp']))
                                                               & (dfThd[
                                                                      'Scenario Group'] == self.thd_group_code)][
                            'Threshold Statistic']
                        out = self.air_temperature_module(models=models, scenarios=scenarios, plant_id=uid,
                                                          backcast_years=self.air_bc_yrs,
                                                          projection_years=self.fut_eval_yrs,
                                                          desirable_air_temp=desirable_air_temp,
                                                          shutdown_air_temp=shutdown_air_temp,
                                                          vulnerability_dataframe=dfVulHeatwaveAir,
                                                          dat_thred_stat=dat_thred_stat,
                                                          sat_thred_stat=sat_thred_stat)
                    temp_unit_data = out
                    temp_unit_type = unit_type

                countries.append(ctry)
                plants.append(pln)
                capacities.append(cap)
                disrupted_hours.append(np.array(out))
                uniqueids.append(uid)
            dfOut = pd.DataFrame(data=np.transpose([countries, uniqueids, plants, capacities, disrupted_hours]),
                                 columns=['country', 'unique id', 'plant', 'capacity', 'disruption hours'])
            dfCtry = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='country',
                                                      custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                           ' extreme and chronic air temperature increases:')
            dfPlt = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='plant',
                                                     custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                          ' extreme and chronic air temperature increases:')

            dfCtry['Risk Type'] = 'air temperature'
            dfPlt['Risk Type'] = 'air temperature'
            return dfCtry, dfPlt

        def water_temperature_assessor():
            """
            Execute water temperature assessment.

            :return: a list of pandas dataframe, dfCtry -> output aggregated by countries, dfPlt -> output aggregated by plants.
            """
            print('\n Now assessing water temperature increase risks for units using once-through cooling system...')

            dfTppLoc = pd.read_excel(self.tpp_locs_fp, sheet_name='tpp_locs', engine='openpyxl')

            uid_list = [i for i in dfTppLoc.loc[dfTppLoc['Source Water'] == 'Y', 'Unique ID'].values]

            models = ['noresm', 'miroc', 'gfdl', 'ipsl', 'hadgem']
            scenarios = ['rcp4p5', 'rcp8p5']

            temp_unit_type = ''
            temp_unit_data = ''
            countries = []
            plants = []
            capacities = []
            disrupted_hours = []
            uniqueids = []
            for pln, uid, ctry, unit, turb, cool, cap, cotech, lat, lon in zip(plns[:], uids[:], ctrys[:], units[:],
                                                                               turbs[:], cools[:], caps[:], cotechs[:],
                                                                               lats[:], lons[:]):
                print('\n', ctry, '|', pln, '| Location ID:', uid, '| Unit:', int(unit), '| Turbine:', turb,
                      '| Cooling:', cool, '| Cooling Technology:', cotech, 'is now being assessed...')
                if turb == 'Diesel' or turb == 'Unknow':
                    turb == 'Steam'
                unit_type = '%s_%s_%s_%s' % (uid, turb, cool, cotech)
                if unit_type == temp_unit_type:
                    out = temp_unit_data
                elif (int(uid) in uid_list) & (turb == 'Steam') & (cool == 'Wet') & ('Once-through' in str(cotech)):
                    dfVulWater = dfVul[(dfVul['Cooling'] == 'Once-through') & (dfVul['Type'] == 'Water temperature')]
                    desirable_water_temp, shutdown_water_temp = dfThd[(dfThd['Type'] == 'Water temperature')
                                                                      & (dfThd['Cooling'] == 'Once-through')
                                                                      & dfThd['Parameter Name'].isin(
                        ['desirable_water_temp', 'shutdown_water_temp'])
                                                                      & (dfThd[
                                                                             'Scenario Group'] == self.thd_group_code)][
                        'Value']
                    dwt_thred_stat, swt_thred_stat = dfThd[(dfThd['Type'] == 'Water temperature')
                                                           & (dfThd['Cooling'] == 'Once-through')
                                                           & dfThd['Parameter Name'].isin(
                        ['desirable_water_temp', 'shutdown_water_temp'])
                                                           & (dfThd[
                                                                  'Scenario Group'] == self.thd_group_code)][
                        'Threshold Statistic']
                    regulatory_limits = dfThd[(dfThd['Type'] == 'Water temperature')
                                              & (dfThd['Cooling'] == 'Once-through')
                                              & dfThd['Parameter Name'].isin(
                        ['regulatory_limits'])
                                              & (dfThd[
                                                     'Scenario Group'] == self.thd_group_code)][
                        'Value'].values[0]
                    out = self.water_temperature_module(models=models, scenarios=scenarios, plant_id=uid,
                                                        vulnerability_dataframe=dfVulWater,
                                                        historical_years=self.his_eval_yrs,
                                                        projection_years=self.fut_eval_yrs,
                                                        desirable_water_temp=desirable_water_temp,
                                                        shutdown_water_temp=shutdown_water_temp,
                                                        dwt_thred_stat=dwt_thred_stat,
                                                        swt_thred_stat=swt_thred_stat,
                                                        regulatory_limits=regulatory_limits)
                elif (int(uid) in uid_list) & (turb == 'Steam') & (cool == 'Wet') & ('Recirculating' in str(cotech)):
                    dfVulWater = dfVul[(dfVul['Cooling'] == 'Recirculating') & (dfVul['Type'] == 'Water temperature')]
                    desirable_water_temp = dfThd[(dfThd['Type'] == 'Water temperature')
                                                 & (dfThd['Cooling'] == 'Recirculating')
                                                 & dfThd['Parameter Name'].isin(['desirable_water_temp'])
                                                 & (dfThd['Scenario Group'] == self.thd_group_code)]['Value'].values[0]
                    dwt_thred_stat = dfThd[(dfThd['Type'] == 'Water temperature')
                                           & (dfThd['Cooling'] == 'Recirculating')
                                           & dfThd['Parameter Name'].isin(['desirable_water_temp'])
                                           & (dfThd['Scenario Group'] == self.thd_group_code)][
                        'Threshold Statistic'].values[0]
                    out = self.rc_module(plant_id=uid, vulnerability_dataframe=dfVulWater,
                                         historical_years=self.rc_bc_yrs,
                                         projection_years=self.fut_eval_yrs, desirable_water_temp=desirable_water_temp,
                                         dwt_thred_stat=dwt_thred_stat)
                else:
                    out = [0.0] * 12
                temp_unit_data = out
                temp_unit_type = unit_type

                countries.append(ctry)
                plants.append(pln)
                capacities.append(cap)
                disrupted_hours.append(np.array(out))
                uniqueids.append(uid)
            dfOut = pd.DataFrame(data=np.transpose([countries, uniqueids, plants, capacities, disrupted_hours]),
                                 columns=['country', 'unique id', 'plant', 'capacity', 'disruption hours'])
            dfCtry = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='country',
                                                      custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                           ' water temperature increases:')
            dfPlt = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='plant',
                                                     custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                          ' water temperature increases:')

            dfCtry['Risk Type'] = 'water temperature'
            dfPlt['Risk Type'] = 'water temperature'
            return dfCtry, dfPlt

        def water_stress_assessor():
            """
            Execute water stress assessment.

            :return: a list of pandas dataframe, dfCtry -> output aggregated by countries, dfPlt -> output aggregated by plants.
            """
            print('\n Now assessing water stress increase risks...')

            df21b = gp.read_file(self.aq21base_fp)
            df21f = gp.read_file(self.aq21futu_fp)
            dfVulWaterStress = dfVul[dfVul['Type'] == 'Water stress']
            start = 0  # 76
            temp_unit_type = ''
            temp_unit_data = ''
            countries = []
            plants = []
            capacities = []
            disrupted_hours = []
            uniqueids = []
            for pln, uid, ctry, unit, turb, cool, cap, lat, lon in zip(plns[start:], uids[start:], ctrys[start:],
                                                                       units[start:], turbs[start:], cools[start:],
                                                                       caps[start:], lats[start:], lons[start:]):
                print('\n', ctry, '|', pln, '| UID:', uid)
                if turb == 'Diesel' or turb == 'Unknown':
                    turb = 'Gas'
                    cool = 'Not Applicable'
                unit_type = '%s_%s_%s' % (uid, turb, cool)
                if unit_type == temp_unit_type:
                    out = temp_unit_data
                else:
                    if cool in ['Wet']:
                        out = self.water_stress_module(lat=lat, lon=lon, pln=pln, uid=uid, base_geodataframe=df21b,
                                                       futu_geodataframe=df21f,
                                                       turb=turb, cool=cool, vulnerability_dataframe=dfVulWaterStress,
                                                       year=self.water_stress_year)
                    else:
                        out = [0.0] * 12
                countries.append(ctry)
                plants.append(pln)
                capacities.append(cap)
                disrupted_hours.append(np.array(out))
                uniqueids.append(uid)
            dfOut = pd.DataFrame(data=np.transpose([countries, uniqueids, plants, capacities, disrupted_hours]),
                                 columns=['country', 'unique id', 'plant', 'capacity', 'disruption hours'])
            dfCtry = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='country',
                                                      custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                           ' water stress increases:')
            dfPlt = unit_data_aggregator_and_printer(df=dfOut, aggregateBy='plant',
                                                     custom_text_to_print='Annual potential generation losses (MWh) due to'
                                                                          ' water stress increases:')

            dfCtry['Risk Type'] = 'water stress'
            dfPlt['Risk Type'] = 'water stress'
            return dfCtry, dfPlt

        def switch_module(m):
            """
            Module switcher.

            :param m: string, any value among 'drought', 'flood', 'air temp', 'water temp', 'water stress'.
            :return:
            """
            switch = {
                'drought': drought_assessor,
                'flood': flood_assessor,
                'air temp': air_temperature_assessor,
                'water temp': water_temperature_assessor,
                'water stress': water_stress_assessor
            }
            return switch[m]()

        def execute_assessment(m):
            """
            Execute a specific module.

            :param m: string, any value among 'drought', 'flood', 'air temp', 'water temp', 'water stress'.
            :return: None
            """
            dfCtry, dfPlt = switch_module(m)
            if save_output is True:
                export_assessment(df_out_dict={'country': dfCtry, 'plant': dfPlt},
                                  file_name=final_assessment_prefix,
                                  run_code=run_code)

        working_fp = self.tpp_working_fp
        dfTpp = pd.read_excel(working_fp, sheet_name='master', engine='openpyxl')
        # Subset/filter plants to be assessed by plant id.
        plant_id_list = sorted(list(set([i for i in dfTpp['Uloc ID'] if int(i) != -9999]))) if self.plant_id_list == [
            'all'] else self.plant_id_list
        dfTpp = dfTpp[(dfTpp['Uloc ID'].isin(plant_id_list))]
        lats, lons, uids, plns, ctrys, units, turbs, cools, caps, cotechs, gentechs = dfTpp['Lat'].values, dfTpp[
            'Lon'].values, \
                                                                                      dfTpp['Uloc ID'].values, dfTpp[
                                                                                          'Power Plant Name'], \
                                                                                      dfTpp['Country'], dfTpp['Unit'], \
                                                                                      dfTpp['Turbine regroup'], \
                                                                                      dfTpp['Cooling regroup'], dfTpp[
                                                                                          'Installed Capacity (MW)'], \
                                                                                      dfTpp['Cooling system'], dfTpp[
                                                                                          'Generation technology']
        dfVul = pd.read_excel(self.vulnerability_factors_fp, engine='openpyxl',
                              sheet_name=f'Sheet{self.vul_group_code}')
        dfThd = pd.read_excel(self.vulnerability_factors_fp, engine='openpyxl', sheet_name='thresholds')

        final_assessment_prefix = self.final_assessment_prefix
        run_code = self.run_code
        save_output = self.save_output

        print('Thermal assessment begins...')

        list(map(execute_assessment, self.module))

        # Export water stress data
        if 'water stress' in self.module:
            try:
                test_final = pd.concat(self.test_out)
                test_final.drop_duplicates(inplace=True)
                test_final.to_csv(
                    os.path.join(self.water_stress_folder, f'water-stress_srr_{self.water_stress_year}.csv'))
            except:
                pass
        # # Export drought assessment per module
        # test_final = pd.concat(self.test_out)
        # test_final.to_csv(os.path.join(self.temp_output_folder, 'drought_assessment_pm.csv'))

        print('Thermal assessment done.')
        print(f'\nThe run code is {self.run_code}')

    def plant_info_explorer(self):
        working_fp = self.tpp_working_fp
        df = pd.read_excel(working_fp, sheet_name='master', engine='openpyxl')
        print(df.columns)

        def plant_regrouping(row):
            f = row['Fuel regroup']
            t = row['Turbine regroup']
            c = row['Cooling regroup']
            w = row['Water source regroup']
            p = '%s-%s-%s-%s' % (f, t, c, w)
            return p

        df['Plant regroup'] = df.apply(lambda row: plant_regrouping(row), axis=1)
        for p in list(set(df['Plant regroup'].values)):
            print(p)


if __name__ == '__main__':
    '''
    1. Before run the script, assign values to flood_data_folder, final_assessment_prefix, module accordingly
    2. To run all assessors, call thermal_assess() without assigning values to the module argument.
    '''

    vul_group_code = 3
    thd_group_code = 19
    scenario_id_list = ['45']
    module = ['drought']
    save_output = False
    surfix_test = ''
    his_eval_yrs = [1965, 2004]
    fut_eval_yrs = [2010, 2049]
    flood_year = 2030
    water_stress_year = 2030
    water_stress_folder = r'D:\WRI\GIS'
    # if module == ['drought']:
    #     his_eval_yrs = [1850, 2004]

    surfix_scenarios = '-'.join(scenario_id_list)
    surfix_module = '-'.join([i.replace(' ', '-') for i in module])
    surfix_date = datetime.now().strftime("%Y%m%d")
    print(
        f'\nfinal-assessment_vul{str(vul_group_code)}_thd{str(thd_group_code)}_rcp{str(surfix_scenarios)}_{str(surfix_module)}_{str(surfix_date)}{str(surfix_test)}\n')
    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    tpr = TppPhyRisk(module=module,
                     flood_data_folder=r"D:\WRI\Floods\inland",
                     save_output=save_output,
                     final_assessment_prefix=f'final-assessment_vul{str(vul_group_code)}_thd{str(thd_group_code)}_rcp{str(surfix_scenarios)}_{str(surfix_module)}_{str(surfix_date)}{surfix_test}',
                     final_assessment_dir=None,
                     aq21base_fp=os.path.join(water_stress_folder,
                                              r'baseline\aqueduct_global_maps_21_shp\aqueduct_global_dl_20150409.shp'),
                     aq21futu_fp=os.path.join(water_stress_folder,
                                              r'future\aqueduct_projections_20150309_shp\aqueduct_projections_20150309.shp'),
                     gddp_recal_folder=os.path.join(work_directory,'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols'),
                     vulnerability_factors_fp=os.path.join(work_directory, r'vulnerability\vulnerability_factors_20210409.xlsx'),
                     spei_data_folder=os.path.join(work_directory, 'spei'),
                     tpp_water_temp_folder=os.path.join(work_directory, 'tpp water temp all'),
                     vul_group_code=vul_group_code, thd_group_code=thd_group_code,
                     scenario_id_list=scenario_id_list,
                     air_bc_yrs=[1950, 2005],
                     rc_bc_yrs=[1980, 2010],
                     tpp_working_fp=None,
                     his_eval_yrs=his_eval_yrs,
                     fut_eval_yrs=fut_eval_yrs,
                     flood_year=flood_year,
                     water_stress_year=water_stress_year,
                     plant_id_list=['all'])
    tpr.thermal_assess()  # call specific assessor(s)
    # tpr.debug_gddp_ts(plant_id_list=[i for i in range(13, 26)] + [8, 12], save_fig=True)
