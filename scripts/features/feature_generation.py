"""
Feature generation or recalculation.
"""
import numpy as np
import pandas as pd
import os
import itertools
import tqdm

from scripts.features import correct_bias as corr_bias
from scripts import utils
from scripts import default
from scripts.data import data_configs

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class Era5:
    def __init__(self, **kwargs):
        self.project_folder = utils.kwargs_generator(kwargs, 'project_folder',
                                                     os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))
        self.era5_folder = utils.kwargs_generator(kwargs, 'data_folder', 'era5_wetbulbtemp')


class NexGddp(Era5):
    """
    Feature generation for NEX-GDDP datasets.
    """

    def __init__(self, **kwargs):
        """
        :keyword project_folder: folder path-like string, work directory.
        :keyword gddp_folder: folder path-like string, the folder path of restructured GDDP datasets with average air temperatures.
        :keyword bc_timespan_bias_corrected: string, '{start year}-{end-year}', indicating the historical period to conduct bias correction, variable placeholder.
        :keyword prj_timespan_bias_corrected: string, '{start year}-{end-year}', indicating the projection period to conduct bias correction, variable placeholder.
        """
        super().__init__()
        self.project_folder = utils.kwargs_generator(kwargs, 'project_folder',
                                                     os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))
        self.gddp_folder = utils.kwargs_generator(kwargs, 'gddp_folder', 'tpp_climate_gddp_restructure_withAirTempAvg')
        self.bc_timespan_bias_corrected = None
        self.prj_timespan_bias_corrected = None

    def correct_bias(self, plant_id, indicator, model, prj_scenario, bc_start_year=1980, bc_end_year=2005,
                     prj_start_year=2030, prj_end_year=2070, correct_bc=True, correct_prj=True):
        """
        Harmonize era5 and gddp indicators, i.e., precipitation and mean air temperature.

        :param plant_id: int/str, plant id.
        :param indicator: str, 'precipitation' or 'air temperature'.
        :param model: str, climate model.
        :param prj_scenario: str, climate scenario.
        :param bc_start_year: int, year-like integer, start year of backcast dataset and observation dataset.
        :param bc_end_year: int, year-like integer, end year of backcast dataset and observation dataset.
        :param prj_start_year: int, year-like integer, start year of projection dataset.
        :param prj_end_year: int, year-like integer, end year of projection dataset.
        :param correct_bc: boolean, conduct for the baseline period.
        :param correct_prj: boolean, correct for the projection period.
        :return: pandas dataframe, columns = ['date', 'value', 'indicator'].
        """

        indicator_name_gddp, indicator_name_era5 = {
            'precipitation': ['pr_nexGddp', 'total_precipitation'],
            'air temperature': ['airTempAvg', 'mean_2m_air_temperature'],
        }.get(indicator)
        time_span = data_configs.RAWDATA_TIMESPAN['GDDP']
        self.bc_timespan_bias_corrected = f'{bc_start_year}-{bc_end_year}'
        self.prj_timespan_bias_corrected = f'{prj_start_year}-{prj_end_year}'

        bc_file_name = f'PL_EBRD_TPP{str(plant_id)}_GDDP_{time_span[0]}_withAirTempAvg.csv'
        df_bc = pd.read_csv(os.path.join(self.project_folder, self.gddp_folder, bc_file_name), index_col=0)
        df_bc = df_bc[(df_bc['model'] == model) & (df_bc['year'] >= bc_start_year)
                      & (df_bc['year'] <= bc_end_year) & (df_bc['indicator'] == indicator_name_gddp)]
        df_bc['date'] = pd.to_datetime(df_bc[['year', 'month', 'day']])
        df_bc['value'] = default.pr_to_m(df_bc['value']) if indicator == 'precipitation' else df_bc['value']
        df_bc[
            'indicator'] = f'{indicator_name_gddp}_unitConverted' if indicator == 'precipitation' else indicator_name_gddp
        df_bc = df_bc.sort_values(by='value')

        prj_file_name = f'PL_EBRD_TPP{str(plant_id)}_GDDP_{time_span[1]}_withAirTempAvg.csv'
        df_prj = pd.read_csv(os.path.join(self.project_folder, self.gddp_folder, prj_file_name), index_col=0)
        df_prj = df_prj[(df_prj['model'] == model) & (df_prj['year'] >= prj_start_year)
                        & (df_prj['year'] <= prj_end_year) & (df_prj['indicator'] == indicator_name_gddp)
                        & (df_prj['scenario'] == prj_scenario)]
        df_prj['date'] = pd.to_datetime(df_prj[['year', 'month', 'day']])
        df_prj['value'] = default.pr_to_m(df_prj['value']) if indicator == 'precipitation' else df_prj['value']
        df_prj['indicator'] = f'{indicator_name_gddp}_unitConverted' if indicator == 'precipitation' else indicator_name_gddp
        df_prj = df_prj.sort_values(by='value')

        df_obs = pd.read_csv(os.path.join(
            self.project_folder, self.era5_folder, f'PL_EBRD_TPP{str(plant_id)}_ERA5_1980-2019_'
                                                   f'restructure_withWetBulbTemp.csv'), index_col=0)
        df_obs = df_obs[(df_obs['year'] >= bc_start_year) & (df_obs['year'] <= bc_end_year)
                        & (df_obs['indicator'] == indicator_name_era5)]
        df_obs['date'] = pd.to_datetime(df_obs[['year', 'month', 'day']])
        df_obs = df_obs.sort_values(by='value')

        df_obs = df_obs[df_obs.date.isin(df_bc.date)]  # Remove data on Feb 29
        # print(len(df_obs.value) == len(df_bc.value))

        global_max = np.max(np.concatenate((df_obs.value, df_bc.value, df_prj.value), axis=None))
        global_min = np.min(np.concatenate((df_obs.value, df_bc.value, df_prj.value), axis=None))
        cdfn_max = np.max([len(df_prj.value), len(df_obs.value)])

        if correct_prj is True:
            prj_corrected = corr_bias.correct_bias_func(in_data=df_prj.value, obs=df_obs.value, bc=df_bc.value,
                                                        cdfn=cdfn_max, global_max=global_max, global_min=global_min,
                                                        reverse_zeros=True)
            prj_corrected_df = pd.concat([pd.Series(df_prj['date'].values, name='date'),
                                          pd.Series(prj_corrected, name='value')], axis=1)
            prj_corrected_df['indicator'] = f'{indicator_name_gddp}_biasCorrectedVsEra5'
            out_prj = prj_corrected_df.sort_values(by='date')
            out_prj_all = df_prj[[i for i in df_prj.columns if i not in ['value', 'indicator']]] \
                .merge(out_prj, on='date')
            out_prj_all = df_prj.append(out_prj_all).sort_values(by=['indicator', 'date'])
        else:
            out_prj = None
            out_prj_all = None

        if correct_bc is True:
            bc_corrected = corr_bias.correct_bias_func(in_data=df_bc.value, obs=df_obs.value, bc=df_bc.value,
                                                       cdfn=cdfn_max, global_max=global_max, global_min=global_min,
                                                       reverse_zeros=True)
            bc_corrected_df = pd.concat([pd.Series(df_bc['date'].values, name='date'),
                                         pd.Series(bc_corrected, name='value')], axis=1)
            bc_corrected_df['indicator'] = f'{indicator_name_gddp}_biasCorrectedVsEra5'
            out_bc = bc_corrected_df.sort_values(by='date')
            out_bc_all = df_bc[[i for i in df_bc.columns if i not in ['value', 'indicator']]] \
                .merge(out_bc, on='date')
            out_bc_all = df_bc.append(out_bc_all).sort_values(by=['indicator', 'date'])
        else:
            out_bc = None
            out_bc_all = None

        return out_prj, out_bc, out_prj_all, out_bc_all,
        # df_obs.sort_values(by='date'), df_bc.sort_values(by='date'), df_prj.sort_values(by='date')

    def correct_bias_batch(self, save_output, output_folder_name=None, bc_start_year=1980, bc_end_year=2005,
                           prj_start_year=2030, prj_end_year=2070, correct_bc=True, correct_prj=True):
        """
        Correct bias for all model-scenario combinations for all plants.

        :param save_output: boolean, save output as csv files in local disk.
        :param output_folder_name: string, output folder name.
        :param bc_start_year: int, start year of the backcast period.
        :param bc_end_year: int, end year of the backcast period.
        :param prj_start_year: int, start year of the projection period.
        :param prj_end_year: int, end year of the projection period.
        :param correct_bc: boolean, whether to correction bias for the backcast period.
        :param correct_prj: boolean, whether to correct bias for the projection period.
        :return: None
        """

        model_list = ['inmcm4', 'MRI-CGCM3', 'CCSM4', 'MPI-ESM-LR', 'CNRM-CM5', 'NorESM1-M', 'IPSL-CM5A-LR',
                      'MIROC-ESM-CHEM', 'GFDL-ESM2M']
        indicator_list = ['precipitation', 'air temperature']
        plant_id_list = [str(i) for i in range(1, 26)]
        prj_scenario_list = ['rcp45', 'rcp85']
        params = list(itertools.product(indicator_list, model_list, prj_scenario_list, [bc_start_year], [bc_end_year],
                                        [prj_start_year], [prj_end_year]))

        if output_folder_name is None:
            output_folder = os.path.join(self.project_folder, f'{self.gddp_folder}_biasCorrected')
        else:
            output_folder = os.path.join(self.project_folder, f'{output_folder_name}')

        def main(plant_id):
            args = list(zip(*params))
            df_list = list(
                map(self.correct_bias, itertools.repeat(plant_id, len(args[0])), args[0], args[1], args[2], args[3],
                    args[4], args[5], args[6]))
            if correct_prj is True:
                df_prj = pd.concat([sub_list[2] for sub_list in df_list])
            else:
                df_prj = None
            if correct_bc is True:
                df_bc = pd.concat([sub_list[3] for sub_list in df_list[1::2]])
            else:
                df_bc = None

            if save_output is True:
                bc_file_name = f'PL_EBRD_TPP{str(plant_id)}_GDDP_{self.bc_timespan_bias_corrected}_withAirTempAvg.csv'
                prj_file_name = f'PL_EBRD_TPP{str(plant_id)}_GDDP_{self.prj_timespan_bias_corrected}_withAirTempAvg.csv'
                if not os.path.exists(output_folder):
                    os.mkdir(output_folder)
                if df_prj is not None:
                    with open(os.path.join(output_folder, f'{os.path.splitext(prj_file_name)[0]}_biasCorrected.csv'),
                              'wb') as out_fp:
                        df_prj.to_csv(out_fp)
                if df_bc is not None:
                    with open(os.path.join(output_folder, f'{os.path.splitext(bc_file_name)[0]}_biasCorrected.csv'),
                              'wb') as out_fp:
                        df_bc.to_csv(out_fp)

        list(map(main, tqdm.tqdm(plant_id_list)))

        print(f'Done bias correction. Please find output here: {output_folder}')


if __name__ == '__main__':
    gddp_gf = NexGddp(gddp_folder='tpp_climate_gddp_restructure_all_withAirTempAvg')
