"""
Data preprocessing and visualization.
"""

import ast
import itertools
import os
import glob
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tqdm
from ipywidgets import widgets
from plotly.subplots import make_subplots

from scripts import ear5

warnings.simplefilter(action='ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class TsViz:
    PROJECT_FOLDER = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    WORK_DIRECTORY = PROJECT_FOLDER
    OUTPUT_DIRECTORY = os.path.join(WORK_DIRECTORY, 'output')

    DROUGHT_FOLDER = os.path.join(PROJECT_FOLDER, 'spei')
    AIRTEMP_FOLDER = os.path.join(PROJECT_FOLDER, 'tpp climate')
    WATERTEMP_FOLDER = os.path.join(PROJECT_FOLDER, 'tpp water temp')
    WBTEMP_FOLDER = os.path.join(PROJECT_FOLDER, 'ear5_wetbulbtemp')
    # GDDP_RECAL_FOLDER = os.path.join(PROJECT_FOLDER, 'tpp_climate_gddp_withWetBulbTemp') # old tpp climate gddp (not completed, no biascorrection)
    GDDP_RECAL_FOLDER = os.path.join(PROJECT_FOLDER, 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols')

    PLANT_ID_LIST = list(range(1, 26))
    DROUGHT_SURFIX_LIST = ['obs', 'cmip5']

    INDICATOR_NAME_LIST = ['airTempMax', 'airTempMin', 'spei', 'waterTemp', 'pr_nexGddp']
    LAYER_NAME_LIST = ['air temperature', 'drought', 'water temperature', 'ear5', 'gddp recal']

    GDDP_INDICATOR_LIST = ['tasmax', 'tasmin', 'pr']

    FIGURE_DPI = 300
    FIGURE_FORMAT = 'png'
    FIGURE_PREFIX = ''
    FIGURE_SURFIX = ''
    TABLE_FORMAT = 'csv'
    TABLE_PREFIX = ''
    TABLE_SURFIX = ''

    def __init__(self, work_directory=WORK_DIRECTORY, output_directory=OUTPUT_DIRECTORY, drought_folder=DROUGHT_FOLDER,
                 airtemp_folder=AIRTEMP_FOLDER, watertemp_folder=WATERTEMP_FOLDER, figure_dpi=FIGURE_DPI,
                 figure_format=FIGURE_FORMAT, figure_prefix=FIGURE_PREFIX, figure_surfix=FIGURE_SURFIX,
                 table_format=TABLE_FORMAT, table_prefix=TABLE_PREFIX, table_surfix=TABLE_SURFIX,
                 plant_id_list=PLANT_ID_LIST, indicator_name_list=INDICATOR_NAME_LIST, wbtemp_folder=WBTEMP_FOLDER,
                 layer_name_list=LAYER_NAME_LIST, gddp_indicator_list=GDDP_INDICATOR_LIST,
                 gddp_recal_folder=GDDP_RECAL_FOLDER):

        self.work_directory = work_directory
        self.output_directory = output_directory
        self.drought_folder = drought_folder
        self.airtemp_folder = airtemp_folder
        self.watertemp_folder = watertemp_folder
        self.wbtemp_folder = wbtemp_folder
        self.figure_dpi = figure_dpi
        self.figure_format = figure_format
        self.figure_prefix = figure_prefix
        self.figure_surfix = figure_surfix
        self.table_format = table_format
        self.table_prefix = table_prefix
        self.table_surfix = table_surfix
        self.plant_id_list = plant_id_list
        self.indicator_name_list = indicator_name_list + ear5.Ear5().INDICATOR_NAME_LIST + ['airTempAvg',
                                                                                            'wet_bulb_temperature_predicted']
        self.layer_name_list = layer_name_list
        self.gddp_recal_folder = gddp_recal_folder

        self.gddp_indicator_list = gddp_indicator_list

        self.df = pd.DataFrame(columns=['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'value'])
        self.lat = None
        self.lon = None
        self.timespan = None

        # variables cannot be user-defined
        self.DDMM_DICT_365 = {1: list(range(1, 32)), 2: list(range(1, 29)), 3: list(range(1, 32)),
                              4: list(range(1, 31)),
                              5: list(range(1, 32)), 6: list(range(1, 31)), 7: list(range(1, 32)),
                              8: list(range(1, 32)),
                              9: list(range(1, 31)), 10: list(range(1, 32)),
                              11: list(range(1, 31)), 12: list(range(1, 32))}
        self.DDMM_DICT_366 = {1: list(range(1, 32)), 2: list(range(1, 30)), 3: list(range(1, 32)),
                              4: list(range(1, 31)),
                              5: list(range(1, 32)), 6: list(range(1, 31)), 7: list(range(1, 32)),
                              8: list(range(1, 32)),
                              9: list(range(1, 31)), 10: list(range(1, 32)),
                              11: list(range(1, 31)), 12: list(range(1, 32))}

    def file_name(self, i, **kwargs):
        switch = {
            0: 'EBRD_TPP%s' % '_'.join([str(kwargs['plant_id']), kwargs['climate_scenario'],
                                        kwargs['climate_model'], str(kwargs['start_year']),
                                        str(kwargs['end_year'])]),
            1: '_'.join(str(kwargs['plant_id'], self.drought_surfix_list[1])),
        }
        return switch.get(int(i), 'Invalid case')

    @staticmethod
    def split_filename(i, file_name):
        '''

        :param i: int
        :param file_name: file name-like string, with extension
        :return: dict,
        '''
        str_list = os.path.splitext(file_name)[0].split('_')

        # air temp
        if i == 0:
            out = {
                'plant_id': str_list[1].replace('TPP', ''),
                'climate_scenario': str_list[2],
                'climate_model': str_list[3],
                'start_year': str_list[4],
                'end_year': str_list[5]
            }
        # drought
        elif i == 1:
            out = {'plant_id': str_list[0]}
        # water temp projections
        elif i == 2:
            out = {
                'plant_id': str_list[1],
                'climate_model': str_list[5],
                'climate_scenario': str_list[6],
                'start_year': str_list[-1].split('-')[0],
                'end_year': str_list[-1].split('-')[1]
            }
        # water temp historical
        elif i == 3:
            out = {
                'plant_id': str_list[1],
                'climate_model': None,
                'climate_scenario': 'historical',
                'start_year': str_list[-1].split('-')[0],
                'end_year': str_list[-1].split('-')[1]
            }
        elif i == 'tpp_air_temp_old':
            out = {
                'plant_id': str_list[2].replace('TPP', ''),
                'climate_scenario': str_list[3],
                'climate_model': str_list[4],
                'start_year': str_list[5],
                'end_year': str_list[6]
            }
        elif i == 'tpp_air_temp':
            out = {
                'plant_id': str_list[3].replace('TPP', ''),
                'climate_scenario': str_list[4],
                'climate_model': str_list[5],
                'start_year': str_list[6],
                'end_year': str_list[7]
            }
        else:
            out = None

        return out

    @staticmethod
    def create_folder(work_directory, folder_name):

        return os.path.join(work_directory, folder_name)

    @staticmethod
    def func_day_month_list(x):
        df = pd.DataFrame(
            np.array(list(zip([i for i in x.keys()], [i for i in x.values()]))).reshape(12, 2),
            columns=['month', 'day'])
        month_list = [i for sub in [[m] * len(x[m]) for m in x.keys()] for i in sub]
        day_list = [i for sub in df.day.to_list() for i in sub]
        return day_list, month_list

    @staticmethod
    def switch_layercode(x):
        return {
            'air temperature': 0,
            'drought': 1,
            'water temperature': 2,
            'ear5': 3,
            'gddp recal': 4,  # basine-level gddp datasets
            'tpp gddp recal': 'tpp_air_temp'  # plant-level gddp datasets
        }.get(x, 'Invalid input!')

    @staticmethod
    def switch_filename_info(x):
        switch = {
            'tpp_air_temp':  # plant-level air temperature data (gddp)
                [['inmcm4', 'MRI-CGCM3', 'CCSM4', 'MPI-ESM-LR', 'CNRM-CM5', 'NorESM1-M', 'IPSL-CM5A-LR',
                  'MIROC-ESM-CHEM', 'GFDL-ESM2M'],
                 ['historical', 'rcp45', 'rcp85'],
                 ['10', '7', '23', '11', '15', '21', '17', '5', '19', '25', '3', '14', '18', '22', '6', '8', '4', '12',
                  '13', '16', '20', '24', '1', '9', '2'],
                 ['1950-2005', '2006-2070']],
            0: [['NorESM1-M', 'CCSM4', 'CNRM-CM5', 'MPI-ESM-LR', 'IPSL-CM5A-LR', 'inmcm4', 'MRI-CGCM3',
                 'MIROC-ESM-CHEM', 'GFDL-ESM2M'],
                ['historical', 'rcp45', 'rcp85'],
                ['10', '7', '23', '11', '15', '21', '17', '5', '19', '25', '3', '14', '18', '22', '6', '8', '4', '12',
                 '13', '16', '20', '24', '1', '9', '2'],
                ['1950-2005', '2030-2070']],
            1: [['HadGEM2-ES', 'MPI-ESM-LR', 'MRI-CGCM3', 'CNRM-CM5', 'CCSM4', 'GFDL-ESM2M', 'INM-CM4'],
                ['historical', 'rcp45', 'rcp85'],
                ['10', '7', '11', '18', '14', '15', '6', '2', '25', '5', '19', '13', '12', '8', '1', '24', '23',
                 '20', '21', '4', '22', '3', '17', '16', '9'],
                ['1965-2004', '2030-2069']],
            2: [['hadgem', 'gfdl', 'ipsl', 'noresm', 'miroc'],
                ['historical', 'rcp4p5', 'rcp8p5'],
                ['7', '23', '15', '21', '17', '5', '19', '25', '3', '14', '18', '22', '4', '12', '13', '16',
                 '20', '24', '1', '2'],
                ['1965-2004', '2030-2069']],
            3: [['Nan'], ['historical'], [str(i) for i in range(1, 26)], ['1980-2019', 'Nan']],
            4: [['NorESM1-M', 'CCSM4', 'CNRM-CM5', 'MPI-ESM-LR', 'IPSL-CM5A-LR', 'inmcm4', 'MRI-CGCM3',
                 'MIROC-ESM-CHEM', 'GFDL-ESM2M'],
                ['historical', 'rcp45', 'rcp85'],
                ['10', '7', '23', '11', '15', '21', '17', '5', '19', '25', '3', '14', '18', '22', '6', '8', '4', '12',
                 '13', '16', '20', '24', '1', '9', '2'],
                ['1950-2005', '2030-2070']],
        }
        return switch.get(x, 'Invalid input!')

    def switch_vizfunc(self, **kwargs):
        x = kwargs['layer_code']
        kwargs = {item: kwargs[item] for item in kwargs.keys() if item != 'layer_code'}
        return {
            0: self.viz_airtemp,
            1: self.viz_drought,
            2: self.viz_watertemp,
            4: self.viz_recalGddp,
        }.get(x, 'Invalid input!')(**kwargs)

    def plot_with_widgets(self, **kwargs):
        layer_code = self.switch_layercode(kwargs['layer'])
        plant = widgets.Dropdown(options=sorted([int(i) for i in self.switch_filename_info(layer_code)[2]]),
                                 value=sorted([int(i) for i in self.switch_filename_info(layer_code)[2]])[0],
                                 description='Plant ID',
                                 disabled=False)
        scenario = widgets.SelectMultiple(options=self.switch_filename_info(layer_code)[1],
                                          value=[self.switch_filename_info(layer_code)[1][0]],
                                          description='Scenario', disabled=False)
        model = widgets.SelectMultiple(options=self.switch_filename_info(layer_code)[0],
                                       value=[self.switch_filename_info(layer_code)[0][0]],
                                       description='Model',
                                       disabled=False)
        aftDict = {item: kwargs.get(item) for item in
                   [i for i in kwargs.keys() if i not in ['plant_ids', 'scenario', 'model']]}
        befDict = {'plant_ids': plant, 'climate_scenario': scenario, 'climate_model': model, 'layer_code': layer_code}
        newDict = {**befDict, **aftDict}

        return self.switch_vizfunc, newDict

    def folder_switch(self, i):
        return {
            0: self.airtemp_folder,
            1: self.drought_folder,
            2: self.watertemp_folder,
            3: self.watertemp_folder
        }.get(i, 'No requested folder information!')

    def list_cs_cm_plantid(self, i):
        cs_list = []
        cm_list = []
        plantid_list = []

        if i == 2:
            basename_list = [i for i in os.listdir(self.folder_switch(i))
                             if i.startswith('TPP') and 'waterTemp_weekAvg' in i]
        elif i == 3:
            basename_list = [i for i in os.listdir(self.folder_switch(i))
                             if i.startswith('TPP') and 'waterTemperature_mergedV2' in i]
        else:
            basename_list = os.listdir(self.folder_switch(i))

        for basename in basename_list:
            name_info_list = self.split_filename(i, basename)
            if i in [0, 2, 3]:
                cs_list.append(name_info_list['climate_scenario'])
                cm_list.append(name_info_list['climate_model'])
                plantid_list.append(name_info_list['plant_id'])
                d = {'climate_scenario': list(set(cs_list)), 'climate_model': list(set(cm_list)),
                     'plant_id': list(set(plantid_list))}
            else:
                plantid_list.append(name_info_list['plant_id'])
                d = {'climate_scenario': None, 'climate_model': None, 'plant_id': list(set(plantid_list))}

        return d

    def std_airtemp(self, **kwargs):
        if 'split_name_param' not in kwargs or kwargs['split_name_param'] is None:
            kwargs['split_name_param'] = 0
        file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        if 'indicator' not in kwargs or kwargs['indicator'] is None:
            kwargs['indicator'] = self.gddp_indicator_list

        # clean air temp datasets
        file_ext = os.path.splitext(kwargs['file_path'])[-1]
        df_airtemp = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel,
        }.get(file_ext)(kwargs['file_path'])
        # df_airtemp = pd.read_csv(kwargs['file_path'], index_col=0)

        indicator_col = [i for i in df_airtemp.columns if 'indicator' in i][0]
        if indicator_col != 'indicatorName':
            df_airtemp.rename(columns={indicator_col: 'indicatorName'}, inplace=True)

        df_airtemp = df_airtemp.sort_values(by=['indicatorName', 'year'])
        df_airtemp = df_airtemp.loc[
            df_airtemp['indicatorName'].isin(kwargs['indicator']), ['data', 'indicatorName', 'year']]

        df_airtemp['data'] = df_airtemp['data'].apply(lambda row: ast.literal_eval(row))
        df_airtemp['n_day'] = df_airtemp['data'].apply(lambda row: len(row))

        month_list = {365: self.func_day_month_list(self.DDMM_DICT_365)[1],
                      366: self.func_day_month_list(self.DDMM_DICT_366)[1]}
        day_list = {365: self.func_day_month_list(self.DDMM_DICT_365)[0],
                    366: self.func_day_month_list(self.DDMM_DICT_366)[0]}
        df_airtemp.loc[:, 'month'] = df_airtemp['n_day'].map(month_list)
        df_airtemp.loc[:, 'day'] = df_airtemp['n_day'].map(day_list)

        df_airtemp = df_airtemp.set_index(['indicatorName', 'year', 'n_day']).apply(pd.Series.explode).reset_index()
        # print(df_airtemp)

        df_airtemp['model'] = self.split_filename(kwargs['split_name_param'], os.path.basename(kwargs['file_path']))[
            'climate_model']
        df_airtemp['scenario'] = self.split_filename(kwargs['split_name_param'], os.path.basename(kwargs['file_path']))[
            'climate_scenario']
        df_airtemp['plant_id'] = self.split_filename(kwargs['split_name_param'], os.path.basename(kwargs['file_path']))[
            'plant_id']

        old_names = list(set(df_airtemp['indicatorName'].values))
        old_max = [i for i in old_names if 'max' in i][0]
        old_min = [i for i in old_names if 'min' in i][0]
        old_pr = [i for i in old_names if 'pr' in i][0]
        indicator_names = {old_max: self.indicator_name_list[0], old_min: self.indicator_name_list[1],
                           old_pr: self.indicator_name_list[4]}
        df_airtemp['indicator'] = df_airtemp['indicatorName'].map(indicator_names)

        df_airtemp = df_airtemp[['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'data']]
        df_airtemp.columns = ['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'value']

        if 'save_output' in kwargs and kwargs['save_output']:
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
            fp_out = os.path.join(self.output_directory, file_name + '_restructure.csv')
            df_airtemp.to_csv(fp_out)
            print('Output is saved here: %s' % fp_out)

        return df_airtemp

    def std_drought(self, **kwargs):
        file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        # Clean drought datasets - spei
        df_drought = pd.read_csv(kwargs['file_path'], index_col=0)
        df_drought = df_drought.sort_values(by=['simulation_scenario', 'model', 'year', 'month'])

        df_drought['day'] = None
        n_day = self.DDMM_DICT_365
        df_drought.loc[:, 'day'] = df_drought['month'].map(n_day)
        df_drought = df_drought.explode('day')

        df_drought['indicator'] = 'spei'

        df_drought['plant_id'] = self.split_filename(1, os.path.basename(kwargs['file_path']))['plant_id']

        df_drought = df_drought[['plant_id', 'year', 'month', 'day', 'model', 'simulation_scenario', 'indicator', 'spei_12']]
        df_drought.columns = ['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'value']

        if 'save_output' in kwargs and kwargs['save_output']:
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
            fp_out = os.path.join(self.output_directory, file_name + '_restructure.csv')
            df_drought.to_csv(fp_out)
            print('Output is saved here: %s' % fp_out)

        return df_drought

    def std_watertemp(self, **kwargs):
        file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        # clean water temp datasets
        df_watertemp = pd.read_csv(kwargs['file_path'], index_col=0)

        df_watertemp['date'] = df_watertemp.date.astype(np.datetime64)
        df_watertemp = df_watertemp.sort_values(by='date')

        # print(df_watertemp.dtypes)

        df_watertemp['year'] = df_watertemp.date.dt.year
        df_watertemp['month'] = df_watertemp.date.dt.month
        df_watertemp['day'] = df_watertemp.date.dt.day

        df_watertemp['indicator'] = self.indicator_name_list[3]

        if 'weekAvg' in kwargs['file_path']:
            df_watertemp['model'] = self.split_filename(2, os.path.basename(kwargs['file_path']))['climate_model']
            df_watertemp['scenario'] = self.split_filename(2, os.path.basename(kwargs['file_path']))['climate_scenario']
            df_watertemp['plant_id'] = self.split_filename(2, os.path.basename(kwargs['file_path']))['plant_id']
        elif 'mergedV2' in kwargs['file_path']:
            df_watertemp['model'] = self.split_filename(3, os.path.basename(kwargs['file_path']))['climate_model']
            df_watertemp['scenario'] = self.split_filename(3, os.path.basename(kwargs['file_path']))['climate_scenario']
            df_watertemp['plant_id'] = self.split_filename(3, os.path.basename(kwargs['file_path']))['plant_id']
        else:
            df_watertemp['model'] = None
            df_watertemp['scenario'] = None

        df_watertemp = df_watertemp[['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'value']]

        if 'save_output' in kwargs and kwargs['save_output']:
            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
            fp_out = os.path.join(self.output_directory, file_name + '_restructure.csv')
            df_watertemp.to_csv(fp_out)
            print('Output is saved here: %s' % fp_out)

        return df_watertemp

    def std_comb(self, dfs, **kwargs):
        df = self.df
        for item in dfs:
            df = df.append(item)
        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs:
                file_name = kwargs['output_name']
            else:
                file_name = 'restructure_comb'

            if not os.path.exists(self.output_directory):
                os.mkdir(self.output_directory)
            fp_out = os.path.join(self.output_directory, file_name + '_restructure_comb.csv')
            df.to_csv(fp_out)
            print('Output is saved here: %s' % fp_out)
        return df

    def restructure_nexGddp_batch(self, **kwargs):
        '''
        Restructure NEX-GDDP calimate datasets with the same structure of ERA5 restructured datasets.
        Group by plant id and time span.
        :param kwargs:
            data_folder (optional): folder path-like string, default value is self.airtemp_folder.
            split_name_param (optional): specify which parse scheme to be used to get metadata information from the file name
            save_output (optional): boolean, default value is None.
            output_directory (optional): folder path-like string, default value is self.output_directory.
        :return: pandas dataframe, restructured NEX-GDDP climate datasets.
        '''

        if 'data_folder' not in kwargs or kwargs['data_folder'] is None:
            kwargs['data_folder'] = self.airtemp_folder
            kwargs['split_name_param'] = 0

        def f(con):
            def main(basename):
                if self.split_filename(kwargs['split_name_param'], basename)['plant_id'] == con[0] and '-'.join(
                        [self.split_filename(kwargs['split_name_param'], basename)[k] for k in
                         ['start_year', 'end_year']]) == con[1]:
                    return self.std_airtemp(file_path=os.path.join(kwargs['data_folder'], basename),
                                            split_name_param=kwargs['split_name_param'])

            dfs = pd.concat(list(map(main, [i for i in os.listdir(kwargs['data_folder']) if not i.endswith('.zip')])))

            if 'save_output' in kwargs and kwargs['save_output'] is not None:
                if 'output_directory' not in kwargs or kwargs['output_directory'] is None:
                    kwargs['output_directory'] = self.output_directory
                if not os.path.exists(kwargs['output_directory']):
                    os.mkdir(kwargs['output_directory'])

                fp_out = os.path.join(kwargs['output_directory'], f'PL_EBRD_TPP{con[0]}_GDDP_{con[1]}.csv')
                dfs.to_csv(fp_out)
                # print(f'Output is saved here: {fp_out}')

            return dfs

        # Group by plant_id and time span
        cons = list(itertools.product(self.switch_filename_info(kwargs['split_name_param'])[2],
                                      self.switch_filename_info(kwargs['split_name_param'])[3]))
        df_all = pd.concat(list(map(f, tqdm.tqdm(cons))))

        return df_all

    def cal_avgAirTemp(self, **kwargs):
        '''
        Calculate mean air temperature for restructured GDDP datasets.
        :param kwargs:
        :return:
        '''
        if 'file_path' in kwargs and kwargs['file_path'] is not None:
            file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]
            df = pd.read_csv(kwargs['file_path'], index_col=0)
        elif 'df' in kwargs and kwargs['df'] is not None:
            df = kwargs['df']

        ind_list = [i for i in df.columns if i not in ['indicator', 'value']]
        df = df.pivot(index=ind_list, columns='indicator', values='value')
        df['airTempAvg'] = (df['airTempMin'] + df['airTempMax']) / 2
        df = df.reset_index()
        df_out = pd.melt(df, id_vars=ind_list, value_vars=['airTempAvg', 'airTempMin', 'airTempMax', 'pr_nexGddp'],
                         value_name='value', var_name='indicator')

        if 'save_output' in kwargs and kwargs['save_output'] is not None:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name + '_withMeanAirTemp'
            else:
                output_name = 'GDDP_withMeanAirTemp'

            if 'output_path' in kwargs and kwargs['output_path'] is not None:
                fp_out = kwargs['output_path']
                if os.path.splitext(fp_out)[-1] != '.csv':
                    fp_out = os.path.splitext(fp_out)[0] + '.csv'
            else:
                if not os.path.exists(self.output_directory):
                    os.mkdir(self.output_directory)
                fp_out = os.path.join(self.output_directory, output_name + '.csv')

            df_out.to_csv(fp_out)
            # print(f'Output is saved here: {fp_out}')

        return df_out

    def cal_avgAirTemp_batch(self, **kwargs):
        return None

    def df_viz_airtemp(self, **kwargs):
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
            df_input = True
        else:
            df_input = False

        def f(x):
            if isinstance(x, str) or isinstance(x, int):
                out = [str(x)]
            else:
                out = x
            return out

        kwargs = {item: f(kwargs[item]) for item in [i for i in kwargs.keys() if i != 'df']}

        if not df_input:
            for plant_id in kwargs['plant_ids']:
                if 'climate_scenario' not in kwargs or kwargs['climate_scenario'] == ['all']:
                    kwargs['climate_scenario'] = self.switch_filename_info(0)[1]
                for cs in kwargs['climate_scenario']:
                    if cs == 'historical':
                        surfix = '1950_2005'
                    else:
                        surfix = '2030_2070'
                    if 'climate_model' not in kwargs or kwargs['climate_model'] == ['all']:
                        kwargs['climate_model'] = self.switch_filename_info(0)[0]
                    for cm in kwargs['climate_model']:
                        file_name = 'EBRD_TPP{}_{}_{}_{}'.format(plant_id, cs, cm, surfix)
                        file_path = os.path.join(self.airtemp_folder, file_name + '.csv')
                        df = df.append(self.std_airtemp(file_path=file_path))

        # Subset
        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            df = df[df['indicator'].isin(kwargs['indicator'])]
        if 'climate_scenario' in kwargs and kwargs['climate_scenario'] is not None:
            if kwargs['climate_scenario'] == 'all':
                kwargs['climate_scenario'] = self.switch_filename_info(0)[1]
            df = df[df['scenario'].isin(kwargs['climate_scenario'])]
        if 'climate_model' in kwargs and kwargs['climate_model'] is not None:
            if kwargs['climate_model'] == 'all':
                kwargs['climate_model'] = self.switch_filename_info(0)[0]
            df = df[df['model'].isin(kwargs['climate_model'])]

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)
        df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        df = df.sort_values(by='date')

        return df

    def df_viz_watertemp(self, **kwargs):
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
            df_input = True
        else:
            df_input = False

        def f(x):
            if isinstance(x, str) or isinstance(x, int):
                out = [str(x)]
            else:
                out = x
            return out

        kwargs = {item: f(kwargs[item]) for item in [i for i in kwargs.keys() if i != 'df']}

        if not df_input:
            for plant_id in kwargs['plant_ids']:
                if 'climate_scenario' not in kwargs or kwargs['climate_scenario'] == ['all']:
                    kwargs['climate_scenario'] = self.switch_filename_info(2)[1]
                for cs in kwargs['climate_scenario']:
                    if cs == 'historical':
                        prefix = 'waterTemperature_mergedV2'
                        surfix = '1965-2004'
                        file_name = 'TPP_{}_{}_{}'.format(plant_id, prefix, surfix)
                        file_path = os.path.join(self.watertemp_folder, file_name + '.csv')
                        df = df.append(self.std_watertemp(file_path=file_path))
                    else:
                        prefix = 'waterTemp_weekAvg_output'
                        surfix = '2030-2069'
                        if 'climate_model' not in kwargs or kwargs['climate_model'] == ['all']:
                            kwargs['climate_model'] = self.switch_filename_info(2)[0]
                        for cm in kwargs['climate_model']:
                            file_name = 'TPP_{}_{}_{}_{}_{}'.format(plant_id, prefix, cm, cs, surfix)
                            file_path = os.path.join(self.watertemp_folder, file_name + '.csv')
                            df = df.append(self.std_watertemp(file_path=file_path))

        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)

        # # Subset
        # if 'indicator' in kwargs and kwargs['indicator'] is not None:
        #     df = df[df['indicator'].isin(kwargs['indicator'])]
        # if 'climate_scenario' in kwargs and kwargs['climate_scenario'] is not None:
        #     if kwargs['climate_scenario'] == 'all':
        #         kwargs['climate_scenario'] = self.switch_filename_info(2)[1]
        #     df = df[df['model'].isin(kwargs['climate_scenario'])]
        # if 'climate_model' in kwargs and kwargs['climate_model'] is not None:
        #     if kwargs['climate_model'] == 'all':
        #         kwargs['climate_model'] = self.switch_filename_info(2)[0]
        #     df = df[df['model'].isin(kwargs['climate_model'])]

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)
        df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        df = df.sort_values(by='date')

        return df

    def df_viz_drought(self, **kwargs):
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
            df_input = True
        else:
            df_input = False

        def f(x):
            if isinstance(x, str) or isinstance(x, int):
                out = [str(x)]
            else:
                out = x
            return out

        kwargs = {item: f(kwargs[item]) for item in [i for i in kwargs.keys() if i != 'df']}

        if not df_input:
            surfix = 'cmip5'
            for plant_id in kwargs['plant_ids']:
                file_name = '{}_{}'.format(plant_id, surfix)
                file_path = os.path.join(self.drought_folder, file_name + '.csv')
                df = df.append(self.std_drought(file_path=file_path))

        # Subset
        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            df = df[df['indicator'].isin(kwargs['indicator'])]
        if 'climate_scenario' in kwargs and kwargs['climate_scenario'] is not None:
            if kwargs['climate_scenario'] == 'all':
                kwargs['climate_scenario'] = self.switch_filename_info(1)[1]
            df = df[df['scenario'].isin(kwargs['climate_scenario'])]
        if 'climate_model' in kwargs and kwargs['climate_model'] is not None:
            if kwargs['climate_model'] == 'all':
                kwargs['climate_model'] = self.switch_filename_info(1)[0]
            df = df[df['model'].isin(kwargs['climate_model'])]

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)
        df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        df = df.sort_values(by='date')

        return df

    def df_viz_wbtemp(self, **kwargs):
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
            df_input = True
        else:
            df_input = False

        def f(x):
            if isinstance(x, str) or isinstance(x, int):
                out = [str(x)]
            else:
                out = x
            return out

        kwargs = {item: f(kwargs[item]) for item in [i for i in kwargs.keys() if i != 'df']}

        if not df_input:
            surfix = 'ERA5_1980-2019_restructure_withWetBulbTemp'
            prefix = 'PL_EBRD'
            for plant_id in kwargs['plant_ids']:
                file_name = '{}_TPP{}_{}'.format(prefix, plant_id, surfix)
                file_path = os.path.join(self.wbtemp_folder, file_name + '.csv')
                df = df.append(pd.read_csv(file_path, index_col=0))

        # Subset
        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            df = df[df['indicator'].isin(kwargs['indicator'])]
        if 'climate_scenario' in kwargs and kwargs['climate_scenario'] is not None:
            if kwargs['climate_scenario'] == 'all':
                kwargs['climate_scenario'] = self.switch_filename_info(2)[1]
            df = df[df['scenario'].isin(kwargs['climate_scenario'])]
        if 'climate_model' in kwargs and kwargs['climate_model'] is not None:
            if kwargs['climate_model'] == 'all':
                kwargs['climate_model'] = self.switch_filename_info(2)[0]
            df = df[df['model'].isin(kwargs['climate_model'])]

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        if 'scenario' in df.columns and 'model' in df.columns:
            df.scenario = df.scenario.astype(str)
            df.model = df.model.astype(str)
            df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        df = df.sort_values(by='date')

        return df

    def df_viz_gddpRecal(self, **kwargs):
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
            df_input = True
        else:
            df_input = False

        def f(x):
            if isinstance(x, str) or isinstance(x, int):
                out = [str(x)]
            else:
                out = x
            return out

        kwargs = {item: f(kwargs[item]) for item in [i for i in kwargs.keys() if i != 'df']}

        if not df_input:
            for plant_id in kwargs['plant_ids']:
                df_list = []
                for surfix in ['1950-2005', '2030-2070']:
                    file_name = f'PL_EBRD_TPP{plant_id}_GDDP_{surfix}_withAirTempAvg_predictedWetBulbTemp'
                    file_path = os.path.join(self.gddp_recal_folder, file_name + '.csv')
                    df_list.append(pd.read_csv(file_path, index_col=0))
                df = pd.concat(df_list)

        # Subset
        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            df = df[df['indicator'].isin(kwargs['indicator'])]
        if 'climate_scenario' in kwargs and kwargs['climate_scenario'] is not None:
            if kwargs['climate_scenario'] == 'all':
                kwargs['climate_scenario'] = self.switch_filename_info(1)[1]
            df = df[df['scenario'].isin(kwargs['climate_scenario'])]
        if 'climate_model' in kwargs and kwargs['climate_model'] is not None:
            if kwargs['climate_model'] == 'all':
                kwargs['climate_model'] = self.switch_filename_info(1)[0]
            df = df[df['model'].isin(kwargs['climate_model'])]

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)
        df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        df = df.sort_values(by='date')

        return df

    def viz_watertemp(self, **kwargs):
        df = self.df_viz_watertemp(**kwargs)

        # Plot figures
        fig = px.scatter(df, x='date', y='value', color='scenario_model')
        fig.update_traces(mode='lines')
        fig.show()

    def viz_airtemp(self, **kwargs):
        df = self.df_viz_airtemp(**kwargs)

        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            # Plot figures
            fig = px.scatter(df, x='date', y='value', color='scenario_model')
            fig.update_traces(mode='lines')
            fig.show()
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.loc[df['indicator'] == 'airTempMin', 'date'],
                                     y=df.loc[df['indicator'] == 'airTempMin', 'value'],
                                     fill=None,
                                     name='airTempMin'
                                     ))
            fig.add_trace(go.Scatter(x=df.loc[df['indicator'] == 'airTempMax', 'date'],
                                     y=df.loc[df['indicator'] == 'airTempMax', 'value'],
                                     fill='tonexty',
                                     name='airTempMax'
                                     ))
            fig.update_traces(mode='lines')
            fig.show()

    def viz_drought(self, **kwargs):
        df = self.df_viz_drought(**kwargs)

        # Plot figures
        fig = px.scatter(df, x='date', y='value', color='scenario_model')
        fig.update_traces(mode='lines')
        fig.show()

    def viz_recalGddp(self, **kwargs):
        df = self.df_viz_gddpRecal(**kwargs)

        # Plot figures
        fig = px.scatter(df, x='date', y='value', color='scenario_model')
        fig.update_traces(mode='lines')
        fig.show()

    def viz_wbtemp(self, **kwargs):
        kwargs.update({'indicator': kwargs['indicator'] + kwargs['indicator_secondary']})
        df = self.df_viz_wbtemp(**kwargs)

        if 'secondary_yaxis' in kwargs and kwargs['secondary_yaxis'] is True and 'indicator_secondary' \
                in kwargs and len(kwargs['indicator_secondary']) > 0:
            fig = make_subplots(specs=[[{'secondary_y': True}]])

            def f(**kwargs):
                fig.add_trace(
                    go.Scatter(x=df.loc[df['indicator'] == kwargs['indicator'], 'date'],
                               y=df.loc[df['indicator'] == kwargs['indicator'], 'value'],
                               name=kwargs['indicator']),
                    secondary_y=kwargs['secondary_y'],
                )

            list(map(lambda kwargs: f(**kwargs), [{'indicator': i, 'secondary_y': True}
                                                  for i in kwargs['indicator_secondary']]))
            list(map(lambda kwargs: f(**kwargs), [{'indicator': i, 'secondary_y': False}
                                                  for i in list(set(kwargs['indicator']) -
                                                                set(kwargs['indicator_secondary']))]))
            fig.show()

        else:
            # Plot figures
            fig = px.scatter(df, x='date', y='value', color='indicator')
            fig.update_traces(mode='lines')
            fig.show()

    def plot_wbtemp_widgets(self, **kwargs):
        plant = widgets.Dropdown(options=[int(i) for i in range(1, 26)],
                                 value=1,
                                 description='Plant ID',
                                 disabled=False)
        indicator = widgets.SelectMultiple(options=ear5.Ear5().ERA5_INDICATOR_LIST + ['wet_bulb_temperature'],
                                           value=[ear5.Ear5().wbtemp_name, ear5.Ear5().airtemp_name],
                                           description='EAR5 Indicator',
                                           disabled=False)
        secondary_yaxis = widgets.Checkbox(value=True,
                                           description='Secondary y axis?',
                                           disabled=False,
                                           indent=False)
        indicator_secondary = widgets.SelectMultiple(options=ear5.Ear5().ERA5_INDICATOR_LIST + ['wet_bulb_temperature'],
                                                     value=['total_precipitation'],
                                                     description='EAR5 Indicator on secondary y axis',
                                                     disabled=not secondary_yaxis.value)

        aftDict = {item: kwargs.get(item) for item in [i for i in kwargs.keys() if i not in ['indicator', 'plant_ids']]}
        befDict = {'plant_ids': plant, 'indicator': indicator, 'secondary_yaxis': secondary_yaxis.value,
                   'indicator_secondary': indicator_secondary}
        newDict = {**befDict, **aftDict}

        return self.viz_wbtemp, newDict

    def switch_dfviz(self, x):
        return {'air temperature': self.df_viz_airtemp,
                'water temperature': self.df_viz_watertemp,
                'drought': self.df_viz_drought,
                'ear5': self.df_viz_wbtemp,
                'gddp recal': self.df_viz_gddpRecal,
                }.get(x, 'Invalid input!')

    def viz_multiple_layers(self, **kwargs):

        if isinstance(kwargs['layer'], str):
            kwargs['layer'] = [kwargs['layer']]

        aftDict = {item: kwargs.get(item) for item in [i for i in kwargs.keys() if i != 'layer']}

        df = pd.concat([self.switch_dfviz(x)(**aftDict) for x in kwargs['layer']])

        # Plot figures
        fig = px.scatter(df, x='date', y='value', color='indicator')
        fig.update_traces(mode='lines')
        fig.show()

    def plot_multi_widgets(self):
        layer = widgets.SelectMultiple(options=self.layer_name_list, value=['ear5', 'gddp recal'],
                                       description='Layer', disabled=False)
        indicator = widgets.SelectMultiple(options=self.indicator_name_list,
                                           value=['wet_bulb_temperature_predicted', ear5.Ear5().wbtemp_name],
                                           description='Indicator',
                                           disabled=False)
        plant = widgets.Dropdown(options=[int(i) for i in range(1, 26)],
                                 value=1,
                                 description='Plant ID',
                                 disabled=False)

        return self.viz_multiple_layers, layer, plant, indicator

    def viz_comb(self, **kwargs):
        # Read data from csv as pd.dataframe and reformat it using std_watertemp()
        df = self.df
        if 'df' in kwargs:
            df = df.append(kwargs['df'])
        else:
            for plant_id in kwargs['plant_ids']:
                if isinstance(kwargs['plant_ids'], int):
                    kwargs['plant_ids'] = [kwargs['plant_ids']]
                for v in kwargs['variables']:
                    if v == 'water temperature':
                        for cs in kwargs['climate_scenario']:
                            if cs == 'historical':
                                prefix = 'waterTemperature_mergedV2'
                                surfix = '1965-2004'
                                file_name = 'TPP_{}_{}_{}'.format(plant_id, prefix, surfix)
                                file_path = os.path.join(self.watertemp_folder, file_name + '.csv')
                                df = df.append(self.std_watertemp(file_path=file_path))
                            else:
                                prefix = 'waterTemp_weekAvg_output'
                                surfix = '2030-2069'
                                for cm in kwargs['climate_model']:
                                    file_name = 'TPP_{}_{}_{}_{}_{}'.format(plant_id, prefix, cm, cs, surfix)
                                    file_path = os.path.join(self.watertemp_folder, file_name + '.csv')
                                    df = df.append(self.std_watertemp(file_path=file_path))
                    elif v == 'air temperature':
                        break
                    else:
                        break

        # print(df)
        # print

        # Add fields
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
        df.scenario = df.scenario.astype(str)
        df.model = df.model.astype(str)
        df['scenario_model'] = df[['scenario', 'model']].agg('-'.join, axis=1)

        # Plot figures
        fig = px.scatter(df, x='date', y='value', color='scenario_model')
        fig.show()

        # 2 xaxis ???

        # Interactive widgets
        # select group by scenario or scenario-model

    def viz_violin(self, **kwargs):
        '''

        :param kwargs:
            'plant_id': string, the unique id of a plant
        :return: None
        '''
        layer_code = self.switch_layercode(kwargs['layer'])
        aftDict = {item: kwargs.get(item) for item in [i for i in kwargs.keys() if i != 'layer']}
        df = self.switch_dfviz(kwargs['layer'])(**aftDict)

        if layer_code == 3:
            df['scenario'] = self.switch_filename_info(layer_code)[1][0]

        # Plot
        fig = px.violin(df, y='value', x='scenario', box=True)

        fig.update_xaxes(title_text='scenario: baseline ({}), projection ({})'
                         .format(self.switch_filename_info(layer_code)[3][0],
                                 self.switch_filename_info(layer_code)[3][1]))
        fig.show()

    def viz_multi_violin(self, **kwargs):
        if isinstance(kwargs['layer'], str):
            kwargs['layer'] = [kwargs['layer']]

        # Combine datasets
        def main(x):
            df = self.switch_dfviz(x)(**kwargs)
            df['layer'] = x
            return df

        df = pd.concat([main(x) for x in kwargs['layer']])

        # Plot figures
        fig = px.violin(df, y='value', x='layer', box=True)
        # fig.update_xaxes(title_text=f'scenario: baseline ({basline_timeframe}), projection ({projection_timeframe})')
        fig.show()

    def plot_violin_widgets(self, **kwargs):
        if 'plant_ids' not in kwargs or kwargs['plant_ids'] is None:
            kwargs['plant_ids'] = [1]
        elif isinstance(kwargs['plant_ids'], str) or isinstance(kwargs['plant_ids'], int):
            kwargs['plant_ids'] = [kwargs['plant_ids']]
        layer = widgets.Dropdown(options=self.layer_name_list,
                                 value=kwargs['layer'],
                                 description='Layer',
                                 disabled=False)
        plant = widgets.SelectMultiple(options=[i for i in range(1, 26)],
                                       value=kwargs['plant_ids'],
                                       description='Plant ID',
                                       disabled=False)

        aftDict = {item: kwargs.get(item) for item in [i for i in kwargs.keys() if i not in ['layer', 'plant_ids']]}
        befDict = {'layer': layer, 'plant_ids': plant}
        newDict = {**befDict, **aftDict}

        return self.viz_violin, newDict

    def print_figure(self, output_directory=None, base_filename=None, dpi=None, format=None, prefix=None, surfix=None):

        return os.path.join(output_directory, base_filename + '.' + format)

    def save_table(self, output_directory=None, base_filename=None, format=None, prefix=None, surfix=None):

        return os.path.join(output_directory, base_filename + '.' + format)


def restructure_dataset_master(dataset, work_directory, output_folder_name, watertemp_folder=None, airtemp_folder=None):
    output_directory = os.path.join(work_directory, output_folder_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    tv = TsViz(output_directory=output_directory)

    if dataset == 'tpp water temp':
        tv.watertemp_folder = watertemp_folder
        futu_timespan = '2006-2069'  # change to auto definition.
        hist_timespan = '1965-2010'
        # for the basline period
        list(map(lambda plant_id: tv.std_watertemp(
            save_output=False,
            file_path=os.path.join(tv.watertemp_folder,
                                   f'TPP_{plant_id}_waterTemperature_mergedV2_{hist_timespan}.csv'))
                 .to_csv(
            os.path.join(tv.output_directory, f'PL_EBRD_TPP{plant_id}_waterTemperature-mergedV2_{futu_timespan}.csv')),
                 tqdm.tqdm(tv.switch_filename_info(2)[2], desc="Baseline period")))

        # for the projection period
        def main(plant_id):
            def f(args):
                cs, cm = args
                if cs == 'historical':
                    pass
                else:
                    return tv.std_watertemp(save_output=False, file_path=os.path.join(tv.watertemp_folder,
                                                                                      f'TPP_{plant_id}_waterTemp_weekAvg_output_{cm}_{cs}_{futu_timespan}.csv'))

            df_list = list(map(f, list(itertools.product(*[tv.switch_filename_info(2)[i] for i in (1, 0)]))))
            pd.concat(df_list).to_csv(
                os.path.join(tv.output_directory,
                             f'PL_EBRD_TPP{plant_id}_waterTemp-WeekAvg-output_{futu_timespan}.csv'))

        list(map(main, tqdm.tqdm(tv.switch_filename_info(2)[2], desc="Projection period")))

        print(f'Done restructuring tpp water temperature datasets. Find output here: {output_directory}\n')

    if dataset == 'tpp air temp':
        tv.airtemp_folder = airtemp_folder
        temp_folder = os.path.join(work_directory, 'temp')
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)
        tv.restructure_nexGddp_batch(data_folder=airtemp_folder, split_name_param='tpp_air_temp',
                                     output_directory=temp_folder, save_output=True)

        def f(file_path):
            file_name = os.path.splitext(os.path.basename(file_path))[0] + '_withAirTempAvg'
            output_path = os.path.join(tv.output_directory, file_name + '.csv')
            return tv.cal_avgAirTemp(file_path=file_path, save_output=True, output_path=output_path)

        df_list = list(map(f, tqdm.tqdm([os.path.join(temp_folder, i) for i in os.listdir(temp_folder)],
                                        desc="Baseline and projection period")))

        # Remove interim output
        list([os.remove(f) for f in glob.glob(os.path.join(temp_folder, '*.csv'))])

        print(f'Done restructuring tpp air temperature gddp datasets. Find output here: {output_directory}\n')


if __name__ == '__main__':
    tv = TsViz(output_directory=r'C:\Users\yan.cheng\PycharmProjects\EBRD\watertemp_output_temp_2030')

    # need to test

    # print(tv.std_airtemp(file_path=r'D:\Users\yan.cheng\PycharmProjects\pythonProject\EBRD\tpp climate\EBRD_TPP1_historical_CNRM-CM5_1950_2005.csv', save_output=True))

    # # Test std_comb()
    # # air Temp
    # dfs = tv.df
    # tpp_climate_folder = os.path.join(tv.work_directory, 'tpp climate')
    # for base_name in tqdm.tqdm(os.listdir(tpp_climate_folder)):
    #     dfs = dfs.append(tv.std_airtemp(file_path=os.path.join(tpp_climate_folder, base_name)))
    # fp_out = os.path.join(tv.output_directory, 'airTemp_comb.csv')
    # dfs.to_csv(fp_out)

    # # water Temp
    # dfs = list()
    # tpp_water_folder = os.path.join(tv.work_directory, 'tpp water temp')
    # for base_name in os.listdir(tpp_water_folder):
    #     dfs = dfs.append(tv.std_watertemp(file_path=os.path.join(tpp_water_folder, base_name)))
    # tv.std_comb(dfs=dfs, save_output=True, output_name='waterTemp_comb')
    #
    # # drought
    # dfs = list()
    # tpp_drought_folder = os.path.join(tv.work_directory, 'spei')
    # for base_name in os.listdir(tpp_drought_folder):
    #     dfs = dfs.append(tv.std_drought(file_path=os.path.join(tpp_drought_folder, base_name)))
    # tv.std_comb(dfs=dfs, save_output=True, output_name='drought_comb')

    # # Test viz_airtemp()
    # tv.viz_airtemp(plant_ids='1', climate_scenario=tv.switch_filename_info(0)[1][0],
    #                    climate_model=tv.switch_filename_info(0)[0][0], indicator='airTempMin')

    # # Test viz_drought()
    # tv.viz_drought(plant_ids='1', climate_scenario=tv.switch_filename_info(1)[1][0],
    #                climate_model=tv.switch_filename_info(1)[0][0], indicator='drought')

    # # Test viz_watertemp()
    # tv.viz_watertemp(plant_ids='1', climate_scenario=tv.switch_filename_info(2)[1][0],
    #                climate_model=tv.switch_filename_info(2)[0][0], indicator='waterTemp')

    # # Test viz_wbtemp()
    # tv.viz_wbtemp(plant_ids='1', indicator=ear5.Ear5().wbtemp_name)
    # tv.viz_violin(risk_type='air temperature', plant_id='12')
    # # Test viz_violin()
    # tv.viz_violin(layer='ear5', plant_ids='1', indicator=ear5.Ear5().wbtemp_name)

    # # Test viz_multiple_layers()
    # tv.viz_multiple_layers(layer=['air temperature', 'ear5'],
    #                        indicator=['airTempMin', ear5.Ear5().wbtemp_name],
    #                        plant_ids='1', climate_scenario=['historical'])

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Standardize tpp climate gddp datasets
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tv.restructure_nexGddp_batch(data_folder=os.path.join(tv.work_directory, 'tpp_climate_gddp'),
    #                              split_name_param='tpp_air_temp',
    #                              output_directory=r'C:\Users\yan.cheng\PycharmProjects\EBRD\tpp_climate_gddp_restructure',
    #                              save_output=True)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Calculate average air temperature
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tv.output_directory = r'C:\Users\yan.cheng\PycharmProjects\EBRD\tpp_climate_gddp_=restructure_withAirTempAvg'
    # if not os.path.exists(tv.output_directory):
    #     os.mkdir(tv.output_directory)
    # tpp_climate_restructure_folder = r'C:\Users\yan.cheng\PycharmProjects\EBRD\tpp_climate_gddp_restructure'
    #
    #
    # def f(file_path):
    #     file_name = os.path.splitext(os.path.basename(file_path))[0] + '_withAirTempAvg'
    #     output_path = os.path.join(tv.output_directory, file_name + '.csv')
    #     return tv.cal_avgAirTemp(file_path=file_path, save_output=True, output_path=output_path)
    #
    #
    # df_list = list(map(f, tqdm.tqdm([os.path.join(tpp_climate_restructure_folder, i)
    #                                  for i in os.listdir(tpp_climate_restructure_folder)])))
    # df_dict = list(zip(os.listdir(tpp_climate_restructure_folder), df_list))
    # print(df_dict)

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Standardize tpp water temp datasets
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # Test std_watertemp()
    # tv.output_directory = r'C:\Users\yan.cheng\PycharmProjects\EBRD\watertemp_output_temp_2030'
    # tv.watertemp_folder = r'C:\Users\yan.cheng\PycharmProjects\EBRD\tpp water temp 2030'
    # futu_timespan = '2006-2069'

    # # 1. Historical
    # list(map(lambda plant_id: tv.std_watertemp(
    #     save_output=False,
    #     file_path=os.path.join(tv.watertemp_folder, f'TPP_{plant_id}_waterTemperature_mergedV2_1965-2010.csv'))
    #          .to_csv(os.path.join(tv.output_directory, f'PL_EBRD_TPP{plant_id}_waterTemperature-mergedV2_1965-2010.csv')),
    #          tv.switch_filename_info(2)[2]))

    # # 2. Projection
    # def main(plant_id):
    #     def f(args):
    #         cs, cm = args
    #         if cs == 'historical':
    #             pass
    #         else:
    #             return tv.std_watertemp(save_output=False, file_path=os.path.join(tv.watertemp_folder,
    #                                                                               f'TPP_{plant_id}_waterTemp_weekAvg_output_{cm}_{cs}_{futu_timespan}.csv'))
    #
    #     df_list = list(map(f, list(itertools.product(*[tv.switch_filename_info(2)[i] for i in (1, 0)]))))
    #     pd.concat(df_list).to_csv(
    #         os.path.join(tv.output_directory, f'PL_EBRD_TPP{plant_id}_waterTemp-WeekAvg-output_{futu_timespan}.csv'))
    #
    #
    # list(map(main, tv.switch_filename_info(2)[2]))
