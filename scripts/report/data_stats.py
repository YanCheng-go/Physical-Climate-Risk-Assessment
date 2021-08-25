'''
Input data exploration after preprocessing.
    - Summary statistic of inputs
'''

import os
import glob
import sys
import itertools

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from scripts.utils import utils
from scripts.data import data_configs


class InputStats:
    def __init__(self, name, **kwargs):
        # Default settings
        WORK_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
        OUTPUT_FN = None

        # Defined by users
        self.work_directory = utils.kwargs_generator(kwargs, 'work_directory', WORK_DIRECTORY)
        self.name = name
        self.output_fn = utils.kwargs_generator(kwargs, 'output_fn', OUTPUT_FN)
        self.year_suffix = utils.kwargs_generator(kwargs, 'year_suffix', '')
        self.water_stress_year_suffix = utils.kwargs_generator(kwargs, 'water_stress_year_suffix', '')
        wbt_models_gddp = ['GFDL-ESM2M', 'NorESM1-M', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM']
        self.historical_years = utils.kwargs_generator(kwargs, 'historical_years', [1965, 2004])
        self.projection_years = utils.kwargs_generator(kwargs, 'projection_years', [2010, 2049])
        self.query_str = {
            'spei_neg2': '(((indicator == "spei_12") & (value <= -2)) | (indicator != "spei_12")) '
                         '& ((year >= {} & year <= {}) | (year>={} & year<={}))'.format(self.historical_years[0],
                                                                                        self.historical_years[1],
                                                                                        self.projection_years[0],
                                                                                        self.projection_years[1]),
            'spei': '(year >= {} & year <= {}) | (year >= {} & year <= {})'.format(self.historical_years[0],
                                                                                   self.historical_years[1],
                                                                                   self.projection_years[0],
                                                                                   self.projection_years[1]),
            'gddp': '(year >= {} & year <= {}) | (year >= {} & year <= {})'.format(self.historical_years[0],
                                                                                   self.historical_years[1],
                                                                                   self.projection_years[0],
                                                                                   self.projection_years[1]),
            'uuwt': '(year >= {} & year <= {}) | (year >= {} & year <= {})'.format(self.historical_years[0],
                                                                                   self.historical_years[1],
                                                                                   self.projection_years[0],
                                                                                   self.projection_years[1]),
            'era5': '(year >= 1980 & year <= 2010)',
            'gddp_wbt': '((year >= 1980 & year <= 2005) | (year >= {} & year <= {})) & ((model == "GFDL-ESM2M") '
                        '| (model == "NorESM1-M") | (model == "IPSL-CM5A-LR") | (model == "MIROC-ESM-CHEM"))'.format(
                self.projection_years[0], self.projection_years[1]),
        }.get(self.name, 'Invalid input!')
        self.find_fp_list_args = {
            'spei_neg2': ['*', 'cmip5'],
            'spei': ['*', 'cmip5'],
            'gddp': ['PL_EBRD', '*', 'GDDP', '*'],
            'uuwt': ['PL_EBRD', '*', '*', '*'],
            'era5': ['PL_EBRD', '*', 'ERA5', '*'],
            'gddp_wbt': {
                '2010-2049': ['PL_EBRD', '*', 'GDDP', ['1980-2005*', '2010-2049*']],
                '2030-2069': ['PL_EBRD', '*', 'GDDP', ['1980-2005*', '2030-2070*']],
            }.get(f'{self.projection_years[0]}-{self.projection_years[1]}'),
        }.get(self.name, 'Invalid input!')
        self.dataset = {
            'spei_neg2': 'SPEI',
            'spei': 'SPEI',
            'gddp': 'GDDP',
            'uuwt': 'UUWT',
            'era5': 'ERA5',
            'gddp_wbt': 'GDDP_WBT',
        }.get(self.name, 'Invalid input!')
        self.data_folder = {
            'spei_neg2': 'spei',
            'spei': 'spei',
            'gddp': 'tpp_climate_gddp_restructure_all_withAirTempAvg',
            'uuwt': 'watertemp_output_temp_all',
            'era5': 'era5_wetbulbtemp',
            'gddp_wbt': 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols',
        }.get(self.name, 'Invalid input!')
        self.prep_df_args = {
            'spei_neg2': ['spei'],
            'spei': ['spei'],
            'gddp': [None],
            'uuwt': ['uuwt'],
            'era5': ['era5'],
            'gddp_wbt': ['gddp_wbt']
        }.get(self.name, 'Invalid input!')

    def config_kwargs(self, dataset):
        return {
            'data_folder': data_configs.FOLDER_NAME.get(dataset),
            'timespan_hist': data_configs.TIMESPAN.get(dataset)[0],
            'timespan_futu': data_configs.TIMESPAN.get(dataset)[1],
            'file_name_prefix': data_configs.FILE_NAME_PREFIX.get(dataset),
            'file_name_suffix': data_configs.FILE_NAME_SURFIX.get(dataset),
        }

    @staticmethod
    def parse_fn(dataset, *args):
        def spei_fn(*args):
            return f'{args[0]}_{args[1]}.csv'

        def gddp_fn(*args):
            return f'{args[0]}_TPP{args[1]}_{args[2]}_{args[3]}.csv'

        def uuwt_fn(*args):
            return f'{args[0]}_TPP{args[1]}_{args[2]}_{args[3]}.csv'

        def era5_fn(*args):
            return f'{args[0]}_TPP{args[1]}_{args[2]}_{args[3]}.csv'

        def gddp_wbt_fn(*args):
            return [f'{args[0]}_TPP{args[1]}_{args[2]}_{args[3][0]}.csv',
                    f'{args[0]}_TPP{args[1]}_{args[2]}_{args[3][1]}.csv']

        return {
            'SPEI': spei_fn,
            'GDDP': gddp_fn,
            'UUWT': uuwt_fn,
            'ERA5': era5_fn,
            'GDDP_WBT': gddp_wbt_fn
        }.get(dataset, 'Invalid input!')(*args)

    def find_fp_list(self, *args, **kwargs):
        fn_ = self.parse_fn(kwargs['dataset'], *args)
        if isinstance(fn_, list):
            fn_list = fn_
        else:
            fn_list = [fn_]
        fp_list = []
        for fn in fn_list:
            fp_str = os.path.join(self.work_directory, kwargs['data_folder'], fn)
            for f in glob.glob(fp_str):
                fp_list.append(f)
        return fp_list

    @staticmethod
    def prep_df(fp_list, *args):
        if args[0] == 'spei':
            # To be moved to data_prep.py
            df_list = []
            for fp in fp_list:
                df = pd.read_csv(fp, index_col=0)
                plant_id = os.path.basename(fp).split('_')[0]
                df['plant_id'] = int(plant_id)
                df_list.append(df)
            df_all = pd.concat(df_list)
            df_all['day'] = int(1)
            df_all.rename(columns={'simulation_scenario': 'scenario'}, inplace=True)
            value_vars_cols = [i for i in df_all.columns if
                               i not in ['plant_id', 'year', 'month', 'day', 'model', 'scenario']]
            df_all = pd.melt(df_all, id_vars=['plant_id', 'year', 'month', 'day', 'model', 'scenario'],
                             value_vars=value_vars_cols, value_name='value', var_name='indicator')
            out = df_all[['plant_id', 'year', 'month', 'day', 'model', 'scenario', 'indicator', 'value']]
        elif args[0] == 'uuwt':
            out = pd.concat([pd.read_csv(fp, index_col=0) for fp in fp_list])
            out['model'].replace([np.nan], ['NAN'], inplace=True)
            out['scenario'].replace(['rcp4p5', 'rcp8p5'], ['rcp45', 'rcp85'], inplace=True)
        elif args[0] == 'era5':
            out = pd.concat([pd.read_csv(fp, index_col=0) for fp in fp_list])
            out['model'] = 'NAN'
            out['scenario'] = 'historical'
        elif args[0] == 'gddp_wbt':
            out = pd.concat([pd.read_csv(fp, index_col=0) for fp in fp_list])
            out.drop(['date'], axis=1, inplace=True)
        else:
            out = pd.concat([pd.read_csv(fp, index_col=0) for fp in fp_list])
        # df.replace([-np.inf, np.inf, np.nan], [-999999, 999999, 0], inplace=True)
        return out

    @staticmethod
    def filter_df(df, query_str):
        out = df.query(query_str)
        return out

    def cal_stats(self, df, groupby=['plant_id', 'indicator', 'scenario', 'model'], value_column='value',
                  output_fn=None):
        # summary_stats = df.groupby(groupby)[value_column].describe().reset_index()
        summary_stats = df.groupby(groupby)[value_column].apply(
            lambda x: x.describe([.25, .5, .75, .9, .99])).reset_index()
        summary_stats = summary_stats.pivot(index=['plant_id', 'indicator', 'scenario', 'model'], columns='level_4',
                                            values='value').reset_index()
        if output_fn is not None:
            file_ext = os.path.splitext(output_fn)[-1]
            if file_ext == '':
                file_ext = '.csv'
                output_fn += file_ext
            out_fp = os.path.join(self.work_directory, 'reports', 'data', output_fn)
            export_switcher = {
                '.csv': summary_stats.to_csv,
                '.xlsx': summary_stats.to_excel
            }.get(file_ext, 'Invalid file extension!')(out_fp)
        return summary_stats

    def main(self, output_fn=None):
        fp_list = self.find_fp_list(*self.find_fp_list_args, dataset=self.dataset, data_folder=self.data_folder)
        df = self.prep_df(fp_list, *self.prep_df_args)
        df_sub = self.filter_df(df, self.query_str)
        summary_stats = self.cal_stats(df_sub, output_fn=output_fn)
        return summary_stats

    @staticmethod
    def calculate_change_from_baseline(fp_in=None, df=None, stat='50%'):
        """Calculate the change of median as compared to baseline condition for the output of the main function"""
        df = pd.read_csv(fp_in, index_col=0) if df is None else df
        out_df = df.groupby(['plant_id', 'indicator', 'scenario'])[stat].apply(
            lambda x: x.describe([.05, .5, .95])).reset_index().rename(columns={'level_3': 'statistic'})
        return out_df

    def var_chg_batch(self, wt_fp, at_fp, dt_fp, ws_fp, wbt_futu_fp, wbt_hist_fp, stat='50%'):
        """Calculate changes for all input climatic variables"""
        wt_df = pd.read_csv(wt_fp, index_col=0)
        wt_df_sub = wt_df[wt_df['indicator'] == 'waterTemp']
        wt_df_sub[stat] = wt_df_sub[stat] - 273.15
        wt_out = self.calculate_change_from_baseline(df=wt_df_sub, stat=stat)
        wt_out = wt_out.pivot(index=['plant_id', 'indicator', 'statistic'], values=[stat],
                              columns=['scenario']).reset_index()
        wt_out.columns = ['plant_id', 'indicator', 'statistic', 'historical', 'rcp45', 'rcp85']
        wt_out['chg_rcp45'] = (wt_out['rcp45'] - wt_out['historical']) / wt_out['historical']
        wt_out['chg_rcp85'] = (wt_out['rcp85'] - wt_out['historical']) / wt_out['historical']

        wbt_futu_df = pd.read_csv(wbt_futu_fp, index_col=0)
        wbt_futu_df_sub = wbt_futu_df[(wbt_futu_df['indicator'] == 'wet_bulb_temperature_predicted')
                                      & (wbt_futu_df['scenario'] != 'historical')]
        wbt_hist_df = pd.read_csv(wbt_hist_fp, index_col=0)
        wbt_hist_df_sub = wbt_hist_df[wbt_hist_df['indicator'] == 'wet_bulb_temperature']
        wbt_df_sub = pd.concat([wbt_hist_df_sub, wbt_futu_df_sub])
        wbt_df_sub['indicator'] = wbt_df_sub['indicator'].replace(['wet_bulb_temperature_predicted'],
                                                                  ['wet_bulb_temperature'])
        wbt_df_sub[stat] = wbt_df_sub[stat] - 273.15
        wbt_out = self.calculate_change_from_baseline(df=wbt_df_sub, stat=stat)
        wbt_out = wbt_out.pivot(index=['plant_id', 'indicator', 'statistic'], values=[stat],
                                columns=['scenario']).reset_index()
        wbt_out.columns = ['plant_id', 'indicator', 'statistic', 'historical', 'rcp45', 'rcp85']
        wbt_out['chg_rcp45'] = (wbt_out['rcp45'] - wbt_out['historical']) / wbt_out['historical']
        wbt_out['chg_rcp85'] = (wbt_out['rcp85'] - wbt_out['historical']) / wbt_out['historical']

        at_df = pd.read_csv(at_fp, index_col=0)
        at_df_sub = at_df[(at_df['indicator'].isin(['airTempMax', 'airTempMin']))]
        at_df_sub = at_df_sub.pivot(index=['plant_id', 'scenario', 'model'], columns=['indicator'], values=[stat])[
            stat].reset_index()
        at_df_sub['airTempAvg'] = (at_df_sub['airTempMax'] + at_df_sub['airTempMin']) / 2
        at_df_sub.drop(['airTempMin', 'airTempMax'], axis=1, inplace=True)
        at_df_sub.columns = ['plant_id', 'scenario', 'model', stat]
        at_df_sub['indicator'] = 'AirTempAvg'
        at_df_sub[stat] = at_df_sub[stat] - 273.15
        at_out = self.calculate_change_from_baseline(df=at_df_sub, stat=stat)
        at_out = at_out.pivot(index=['plant_id', 'indicator', 'statistic'], values=[stat],
                              columns=['scenario']).reset_index()
        at_out.columns = ['plant_id', 'indicator', 'statistic', 'historical', 'rcp45', 'rcp85']
        at_out['chg_rcp45'] = (at_out['rcp45'] - at_out['historical']) / at_out['historical']
        at_out['chg_rcp85'] = (at_out['rcp85'] - at_out['historical']) / at_out['historical']

        dt_df = pd.read_csv(dt_fp, index_col=0)
        dt_df_sub = dt_df[dt_df['indicator'] == 'spei_12']
        dt_out = self.calculate_change_from_baseline(df=dt_df_sub, stat=stat)
        dt_out = dt_out.pivot(index=['plant_id', 'indicator', 'statistic'], values=[stat],
                              columns=['scenario']).reset_index()
        dt_out.columns = ['plant_id', 'indicator', 'statistic', 'historical', 'rcp45', 'rcp85']
        dt_out['chg_rcp45'] = (dt_out['rcp45'] - dt_out['historical']) / dt_out['historical']
        dt_out['chg_rcp85'] = (dt_out['rcp85'] - dt_out['historical']) / dt_out['historical']

        out_all = pd.concat([wt_out, at_out, wbt_out, dt_out])
        out_all['statistic'] = out_all['statistic'].replace(['50%', '5%', '95%', 'mean'], ['med', 'q5', 'q95', 'avg'])
        out_all = pd.melt(out_all, id_vars=['plant_id', 'indicator', 'statistic'],
                          value_vars=['chg_rcp45', 'chg_rcp85'], value_name='value', var_name='scenario')
        out_all['scenario'] = out_all['scenario'].replace(['chg_rcp45', 'chg_rcp85'], ['RCP4.5', 'RCP8.5'])
        out_all.to_csv(
            os.path.join(self.work_directory, 'reports', 'data', f'change_wt-at-dt-wbt{self.year_suffix}.csv'))

        ws_df = pd.read_csv(ws_fp, index_col=0)
        ws_df['indicator'] = 'water_stress_srr'
        ws_df['statistic'] = ws_df.apply(lambda x: x['variable'].split('_')[0], axis=1)
        ws_df.to_csv(
            os.path.join(self.work_directory, 'reports', 'data', f'change_ws-srr{self.water_stress_year_suffix}.csv'))

    def find_percentile_change(self, input_fp, thd_val=None, save_output=None):
        indicator = {
            'gddp_wbt': 'wet_bulb_temperature_predicted',
            'gddp': 'airTempAvg',
            'uuwt': 'waterTemp',
        }.get(self.name)
        plant_list = {
            'gddp_wbt': [1, 3, 5, 7, 10, 12, 13, 14, 15, 19, 25],
            'gddp': [5, 6, 8, 9, 10, 11, 12],
            'uuwt': [i for i in range(1, 26) if i not in [4, 6, 8, 9, 10, 11]],
        }.get(self.name)

        historical_scenario = 'historical'

        def read_hist_thd():
            stat_name = {
                'gddp': '90%',
                'uuwt': '90%',
                'gddp_wbt': '99%',
            }.get(self.name)
            df_thd_all = pd.read_csv(input_fp, index_col=0)
            df_thd_sub = df_thd_all[df_thd_all['scenario'] == 'historical'][
                ['plant_id', 'indicator', 'scenario', 'model', stat_name]]
            df_thd = df_thd_sub[df_thd_sub['plant_id'].isin(plant_list)]
            return df_thd

        def read_futu_df():
            fp_list = self.find_fp_list(*self.find_fp_list_args, dataset=self.dataset, data_folder=self.data_folder)
            df = self.prep_df(fp_list, *self.prep_df_args)
            df = self.filter_df(df, self.query_str)
            df = df[df['indicator'] == indicator]
            return df

        if thd_val is None:
            df_thd = read_hist_thd()
            df_thd.columns = [i if i in ['plant_id', 'scenario', 'model', 'indicator'] else 'value' for i in
                              df_thd.columns]
            model_list = [i for i in df_thd.model.unique()]

        df = read_futu_df()
        futu_model_list = [i for i in df.model.unique() if 'NAN' not in i]
        futu_scenario_list = [i for i in df.scenario.unique() if 'historical' not in i]

        out = []
        for plant_id, scenario, model in list(itertools.product(plant_list, futu_scenario_list, futu_model_list)):
            if thd_val is None:
                if model_list != ['NAN']:
                    thd_sub = df_thd[(df_thd['plant_id'] == plant_id) & (df_thd['model'] == model)]
                else:
                    thd_sub = df_thd[(df_thd['plant_id'] == plant_id)]
                val = thd_sub.value.values[0]
            else:
                val = thd_val
            df_sub = df[(df['plant_id'] == plant_id) & (df['scenario'] == scenario) & (df['model'] == model)]
            series = df_sub.value.values
            futu_percentile = sum(series < val) / len(series)
            out.append((plant_id, scenario, model, val, futu_percentile))

        if thd_val is not None:
            hist_scenario_list = [historical_scenario]
            hist_model_list = [i for i in df[df['scenario'] == historical_scenario].model.unique()]
            for plant_id, scenario, model in list(itertools.product(plant_list, hist_scenario_list, hist_model_list)):
                df_sub = df[(df['plant_id'] == plant_id) & (df['scenario'] == scenario) & (df['model'] == model)]
                series = df_sub.value.values
                hist_percentile = sum(series < thd_val) / len(series)
                out.append((plant_id, scenario, model, val, hist_percentile))

        df_out = pd.DataFrame(out, columns=['plant_id', 'scenario', 'model', 'threshold', 'percentile'])

        df_out['percentile_median'] = df_out.groupby(['plant_id', 'scenario'])['percentile'].transform(np.median)

        if save_output is True:
            out_fp = {
                'gddp_wbt': os.path.join(self.work_directory, 'reports', 'data',
                                         f'wet-bulb-temperature_p99_{self.projection_years[0]}-{self.projection_years[1]}.csv'),
                'gddp': os.path.join(self.work_directory, 'reports', 'data',
                                     f'air-temperature_p90_{self.projection_years[0]}-{self.projection_years[1]}.csv'),
                'uuwt': os.path.join(self.work_directory, 'reports', 'data',
                                     f'water-temperature_p90_{self.projection_years[0]}-{self.projection_years[1]}.csv'),
            }.get(self.name)
            if thd_val is not None:
                out_fp = out_fp.replace('water-temperature_p90', f'water-temperature_{thd_val}')
            df_out.to_csv(out_fp)
            print(f'Output saved here: {out_fp}')
        return df_out


if __name__ == '__main__':
    name = 'gddp_wbt'

    historical_years = {
        'gddp': [1950, 2005],
        'spei': [1965, 2004],
        'uuwt': [1965, 2004],
        'spei_neg2': [1965, 2004],
    }.get(name)
    projection_years = [2010, 2049]
    year_suffix = f'_{projection_years[0]}-{projection_years[1]}'
    water_stress_year_suffix = '_2030'
    if projection_years == [2030, 2069]:
        year_suffix = ''
        water_stress_year_suffix = ''

    gddp_wbt_historical_years = [1980, 2005]
    era5_historical_years = [1980, 2010]
    output_fn = {
        'gddp_wbt': f'wet-bulb-temperature_gddp_{gddp_wbt_historical_years[0]}-{gddp_wbt_historical_years[1]}_{projection_years[0]}-{projection_years[1]}',
        'spei_neg2': f'spei-neg2_spei_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}',
        'spei': f'spei_spei_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}',
        'uuwt': f'water-temperature_uuwt_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}',
        'gddp': f'air-temperature_gddp_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}',
        'era5': f'wet-bulb-temperature_era5_{era5_historical_years[0]}-{era5_historical_years[1]}',
    }.get(name)
    insta = InputStats(name=name, output_fn=output_fn, historical_years=historical_years,
                       projection_years=projection_years, year_suffix=year_suffix,
                       water_stress_year_suffix=water_stress_year_suffix)

    # -----------------------------------------------------------------
    # Use-case 1: Calculate statistics of input datasets (need to test)
    # -----------------------------------------------------------------
    summary_stats = insta.main(output_fn=output_fn)

    # ---------------------------------------------------------------------------------------------------------
    # Use-case 2: Calculate the change of median as compared to baseline condition for the output of use-case 1
    # Before try out use-case 2, please run use-case 1 for all datasets
    # (i.e., ['gddp_wbt', 'spei_neg2', 'uuwt', 'gddp', 'era5', 'spei']) (need to test)
    # ---------------------------------------------------------------------------------------------------------
    # # Individual process
    # insta.calculate_change_from_baseline(fp_in=r'C:\Users\yan.cheng\PycharmProjects\EBRD\reports\data\spei_spei-neg2_cal5-2.csv')

    # # Batch process
    # data_folder = os.path.join(insta.work_directory, 'reports', 'data')
    # wt_fp = os.path.join(data_folder,
    #                      f'water-temperature_uuwt_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}.csv')
    # at_fp = os.path.join(data_folder,
    #                      f'air-temperature_gddp_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}.csv')
    # dt_fp = os.path.join(data_folder,
    #                      f'spei-neg2_spei_{historical_years[0]}-{historical_years[1]}_{projection_years[0]}-{projection_years[1]}.csv')
    # wbt_hist_fp = os.path.join(data_folder,
    #                            f'wet-bulb-temperature_era5_{era5_historical_years[0]}-{era5_historical_years[1]}.csv')
    # wbt_futu_fp = os.path.join(data_folder,
    #                            f'wet-bulb-temperature_gddp_{gddp_wbt_historical_years[0]}-{gddp_wbt_historical_years[1]}_{projection_years[0]}-{projection_years[1]}.csv')
    # ws_fp = os.path.join(insta.work_directory, 'data', 'processed', 'water_stress',
    #                      f'water-stress_srr{water_stress_year_suffix}.csv')
    # insta.var_chg_batch(wt_fp=wt_fp, at_fp=at_fp, dt_fp=dt_fp, ws_fp=ws_fp, wbt_futu_fp=wbt_futu_fp,
    #                     wbt_hist_fp=wbt_hist_fp, stat='50%')

    # -----------------------------------------------------------------------------------------
    # Use-case 3: How thresholds retrieved from historical datasets change in the future period
    # Before try out use-case 3, please complete use-case 1 and 2. (need to test)
    # -----------------------------------------------------------------------------------------
    # output = insta.find_percentile_change(name=name, input_fp=output_fp save_output=True, thd_val=None)
    # print(output)
