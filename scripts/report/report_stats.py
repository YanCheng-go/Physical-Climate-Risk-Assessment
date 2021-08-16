"""
Analyse the output of thermal_assessment() in power_plant_physical_physical_climate_risk_assessment.py
"""

import glob
import os
import sys

import pandas as pd
import numpy as np
import itertools

import datetime

from scripts.utils import utils


class PrepReport:
    """Preprocess the output of thermal assessment"""

    def __init__(self, **kwargs):
        """
        :keyword work_directory: folder path-like string, work directory.
        :keyword interim_folder: folder path-like string, where the output of power_plant_physical_climate_risk_assessment.py saved.
        :keyword processed_folder: folder path-like string, where the post-processed files to be stored.
        """

        self.date_code = datetime.datetime.now().strftime("%Y%m%d")
        self.work_directory = utils.kwargs_generator(kwargs, 'work_directory',
                                                     os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))
        self.report_folder = 'final assessment'
        self.report_prefix = 'final-assessment'
        self.report_suffix = ''
        self.thds_prefix = 'temp-var'
        self.interim_folder = utils.kwargs_generator(kwargs, 'interim_folder', 'final assessment\\interim')
        self.processed_folder = utils.kwargs_generator(kwargs, 'processed_folder', 'final assessment\\processed')
        self.temp_output_folder = 'output_temp'
        for i in [self.interim_folder, self.processed_folder, self.report_folder]:
            if not os.path.exists(os.path.join(self.work_directory, i)):
                os.mkdir(os.path.join(self.work_directory, i))

    @staticmethod
    def parse_thds_txt(fp):
        """
        Return refactored txt files, which includes the following information:
            - shutdown & desired water temperature defined in water temperature assessment for once-through plants.
            - wet-bulb temperature (99th percentile), and desired outlet water temperature in water temperature assessment for recirculating plants.
            - shutdown & desired air temperature defined in air temperature assessment.
        Please refer to the 'thresholds' sheet in the vulnerability factor table for associated calculation of aforementioned thresholds/arguments.

        :param fp: file path-like string, file path of the txt file to be refactored.
        :return: a tuple of two Pandas DataFrame, i.e., thresholds defined in air temperature assessment and water temperature assessment for each plant.
        """

        def parse_at_line(l):
            """
            parse air temperature txt file.

            :param l: each line in the text file.
            :return: dictionary, keys -> 'model', 'thred_name', 'thred_val', 'indicator', 'statistic'
            """

            l = l.split('\n')[0]
            plant_id = int(l.split(' | ')[0].split(': ')[-1])
            model = l.split(' | ')[1].split(': ')[-1]
            thred_vul = l.split(' | ')[-1].split(': ')[-1]
            thred_name = l.split(' | ')[-1].split(': ')[0]
            indicator = thred_name.split('-')[0]
            statistic = thred_name.split('-')[1]
            return {'plant_id': plant_id, 'model': model, 'thred_name': thred_name, 'thred_val': thred_vul,
                    'indicator': indicator, 'statistic': statistic}

        def parse_wt_line(l):
            """
            parse water temperature txt file.

            :param l: each line in the text file.
            :return: dictionary, keys -> 'plant_id', 'thred_name', 'thred_val'
            """

            l = l.split('\n')[0]
            plant_id = int(l.split(' | ')[0].split(': ')[-1])
            thred_vul = l.split(' | ')[-1].split(': ')[-1]
            thred_name = l.split(' | ')[-1].split(': ')[0]
            return {'plant_id': plant_id, 'thred_name': thred_name, 'thred_val': thred_vul}

        fp = glob.glob(fp)[0]
        file1 = open(fp, 'r')
        Lines = file1.readlines()

        # Air temperature module
        lines_at = [item for idx, item in enumerate(Lines) if 'Air Temperature' in item]
        if len(lines_at) != 0:
            df_at = pd.DataFrame(list(map(parse_at_line, lines_at)))
            df_at['thred_val'] = df_at['thred_val'].astype('float')
            df_at_groupby = df_at.groupby(['plant_id', 'thred_name', 'indicator', 'statistic']).median().reset_index()
            df_at_out = \
            df_at_groupby[['plant_id', 'thred_name', 'thred_val']].pivot(columns='thred_name', index=['plant_id'])[
                'thred_val'].reset_index()
        else:
            df_at_out = None

        # Water temperature module
        lines_wt = [item for idx, item in enumerate(Lines) if
                    ('Water Temperature' in item) or ('Wet-bulb Temperature' in item)]
        if len(lines_wt) != 0:
            df_wt = pd.DataFrame(list(map(parse_wt_line, lines_wt)))
            df_wt['thred_val'] = df_wt['thred_val'].astype('float')
            df_wt_out = df_wt.pivot(columns='thred_name', index=['plant_id'])['thred_val'].reset_index()
        else:
            df_wt_out = None
        return df_at_out, df_wt_out

    @staticmethod
    def read_files(fp):
        """
        Read one or multiple files depending on the format of file path string assigned. Data will be concatenated in the case of multiples files.

        :param fp: file path-like string, support regex, must include the file extension in the string.
        :return: Pandas DataFrame
        """

        file_ext = os.path.splitext(fp)[-1]
        fp_list = glob.glob(fp)
        reader_switcher = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel
        }
        df = pd.concat([reader_switcher.get(file_ext)(f, index_col=0) for f in fp_list])
        return df

    @staticmethod
    def attach_thds(df_report, df_thds_at=None, df_thds_wt=None):
        df_out = df_report.copy()
        for df_thds in [df_thds_at, df_thds_wt]:
            if df_thds is not None:
                df_thds.rename(columns={'plant_id': 'Group'}, inplace=True)
                df_out = df_out.merge(df_thds, on='Group', how='outer')
        return df_out

    @staticmethod
    def cal_rankings(df):
        """ Calculate the rankings of the generation loss induced by different physical risks."""

        val_columns = ['Baseline Value', 'Projection Value']
        idx_columns = [i for i in df.columns if i not in val_columns]
        df = pd.melt(df, id_vars=idx_columns, value_vars=val_columns, var_name='variable', value_name='value')
        # df.reset_index(inplace=True)
        # df.rename(columns={'index': 'Row Index'}, inplace=True)
        df.replace([-np.inf, np.inf, np.nan], [-999999, 999999, 0], inplace=True)
        df_groupby = df.groupby(['Statistic Type', 'Run Code', 'Group', 'variable'])
        df['gen_sum'] = df_groupby['value'].transform(sum)
        df['gen_per'] = df['value'] / df['gen_sum']
        df['gen_rank_min'] = df.groupby(['Statistic Type', 'Run Code', 'Group', 'variable'])['gen_per'].rank(
            ascending=False, method='min')
        df['gen_rank_max'] = df.groupby(['Statistic Type', 'Run Code', 'Group', 'variable'])['gen_per'].rank(
            ascending=False, method='max')
        return df

    def feature_engineer(self, vul_id, thd_id, scenario_id, module, date, groupby, run_code, save_output=False):
        """return final assessment tables with thresholds and rankings"""

        report_file_str = f'{self.report_prefix}_vul{vul_id}_thd{thd_id}_rcp{scenario_id}_{module}_{date}_{groupby}.xlsx'
        thds_file_str = f'{self.thds_prefix}_vul{vul_id}_thd{thd_id}_rcp{scenario_id}_{module}_{date}_{groupby}_{run_code}.txt'
        fp_report_excel = os.path.join(self.work_directory, self.interim_folder, report_file_str)
        fp_thds_txt = os.path.join(self.work_directory, self.interim_folder, thds_file_str)

        df_report = self.read_files(fp_report_excel)
        try:
            df_thds_at, df_thds_wt = self.parse_thds_txt(fp_thds_txt)
        except:
            df_thds_at, df_thds_wt = None, None
        df = self.attach_thds(df_report, df_thds_at, df_thds_wt)
        df_withStats = self.cal_rankings(df)

        if save_output is True:
            ouput_fn = os.path.basename(fp_report_excel)
            fp_out = os.path.join(self.work_directory, self.temp_output_folder, ouput_fn)
            df_withStats.to_excel(fp_out, engine='openpyxl')
            print(f'Output saved here: {fp_out}')
        return df_withStats

    def main(self, vul_id_list, thd_id_list, scenario_id_list, module_list=['*'], date_list=['*'],
             groupby_list=['ByPlant', 'ByCountry'], run_code_list=['*'], output_fn=None):
        """Calculate rankings and merge"""

        def f(args):
            vul_id, thd_id, scenario_id, module, date, groupby, run_code = args
            df = self.feature_engineer(vul_id, thd_id, scenario_id, module, date, groupby, run_code)
            df['vul_id'] = vul_id
            df['thd_id'] = thd_id
            df['scenario_id'] = scenario_id
            return [groupby, df]

        out_list = list(map(f, list(
            itertools.product(vul_id_list, thd_id_list, scenario_id_list, module_list, date_list, groupby_list,
                              run_code_list))))
        df_con_list = [pd.concat([i[1] for i in out_list if groupby in i[0]]) for groupby in groupby_list]
        if output_fn is not None:
            try:
                output_fn = os.path.splitext(output_fn)[0]
            except:
                output_fn = output_fn
            output_basename_list = [output_fn + f'_{groupby}.xlsx' for groupby in groupby_list]
            fp_out_list = [os.path.join(self.work_directory, self.processed_folder, basename) for basename in
                           output_basename_list]
            list([df.to_excel(fp_out, engine='openpyxl') for df, fp_out in list(zip(df_con_list, fp_out_list))])
            print(f'Output saved here: {fp_out_list}')
        return df_con_list

    @staticmethod
    def merge(df_list=None, fp_list=None):
        """Please run for ByPlant and ByCounty separately"""

        if df_list is None and fp_list is not None:
            reader_switcher = {
                '.csv': pd.read_csv,
                '.xlsx': pd.read_excel
            }
            df_list = []
            for fp in fp_list:
                file_ext = os.path.splitext(fp)[-1]
                df = reader_switcher.get(file_ext)(fp, index_col=0)
                df['vul_id'], df['thd_id'], df['scenario_id'] = os.path.basename(
                    fp).split('_')[1:5]
                df['scenario_id'].replace('rcp', '', inplace=True)
                df['vul_id'].replace('vul', '', inplace=True)
                df['thd_id'].replace('thd', '', inplace=True)
                df_list.append(df)
            df = pd.concat(df_list)
        elif df_list is not None:
            df = pd.concat(df_list)
        return df

    def merge_batch(self, vul_id_list, thd_id_list, scenario_id_list, module_list=['*'], date_list=['*'],
                    groupby_list=['ByPlant', 'ByCountry'], run_code_list=['*'], output_fn=None):
        """
        Merge results of different parameter settings, i.e., vul, thd, and scenario, into a single one.

        :param vul_id_list: list, reference number id of vulnerability scenarios, i.e., 19 -> no regulatory limits, 21 -> regulatory limits
        :param thd_id_list: list, reference number id of threshold settings.
        :param scenario_id_list: list, reference number id of climate scenarios, '45' -> RCP4.5, '85' -> RCP8.5
        :param module_list: list, '*' -> all modules
        :param date_list: list, a list of date indicating execute the function on output generated in which days.
        :param groupby_list: list, a subset of ['ByPlant', 'ByCountry']
        :param run_code_list: list, '*' -> all runcodes
        :param output_fn: file name-like string, output file name with no file extension.
        :return: pandas DataFrame,
        """

        def f(args):
            vul_id, thd_id, scenario_id, module, date, groupby, run_code = args
            report_file_str = f'{self.report_prefix}_vul{vul_id}_thd{thd_id}_rcp{scenario_id}_{module}_{date}_{groupby}.xlsx'
            fp_report_excel = os.path.join(self.work_directory, self.interim_folder, report_file_str)
            df = self.read_files(fp_report_excel)
            df['vul_id'] = vul_id
            df['thd_id'] = thd_id
            df['scenario_id'] = scenario_id
            return [groupby, df]

        out_list = list(map(f, list(
            itertools.product(vul_id_list, thd_id_list, scenario_id_list, module_list, date_list, groupby_list,
                              run_code_list))))
        df_con_list = [pd.concat([i[1] for i in out_list if groupby in i[0]]) for groupby in groupby_list]
        if output_fn is not None:
            try:
                output_fn = os.path.splitext(output_fn)[0]
            except:
                output_fn = output_fn
            output_basename_list = [output_fn + f'_{groupby}.xlsx' for groupby in groupby_list]
            fp_out_list = [os.path.join(self.work_directory, self.processed_folder, basename) for basename in
                           output_basename_list]
            list([df.to_excel(fp_out, engine='openpyxl') for df, fp_out in list(zip(df_con_list, fp_out_list))])
            print(f'Output saved here: {fp_out_list}')
        return df_con_list

    def update_modules(self):
        """Update final assessment of individual model in the merged final assessment file and the test report"""
        pass


class MiscAnal:
    """Miscellaneous Analyses for the processed output of thermal assessments"""

    def __init__(self):
        pass

    @staticmethod
    def cal_portfolio_stats(df, groupby=['Risk Type', 'Statistic Type', 'Run Code', 'variable'], value_name='value'):
        """retrieve sum stats of all units/plants"""

        out = df.groupby(groupby)[value_name].sum().reset_index()
        return out

    def cal_portfolio_stats_batch(self, fp, output_fn=None):
        file_ext = os.path.splitext(fp)[-1]
        fp_list = glob.glob(fp)
        reader_switcher = {
            '.csv': pd.read_csv,
            '.xlsx': pd.read_excel
        }
        df = pd.concat([reader_switcher.get(file_ext)(fp, index_col=0) for fp in fp_list])

        out = self.cal_portfolio_stats(df)

        if output_fn is not None:
            file_ext = os.path.splitext(output_fn)[-1]
            if file_ext == '':
                file_ext = '.csv'
                output_fn += file_ext
            out_fp = os.path.join(self.work_directory, self.report_folder, output_fn)
            export_switcher = {
                '.csv': out.to_csv,
                '.xlsx': out.to_excel
            }.get(file_ext, 'Invalid input!')(out_fp)


if __name__ == '__main__':
    vul_id_list = [3]
    thd_id_list = [21, 19]
    scenario_id_list = ['45', '85']
    module_list = ['*']
    date_list = ['20210519']
    groupby_list = ['ByPlant']
    run_code_list = ['*']
    interim_folder = 'final assessment\\interim'
    suffix = ''
    suffix_date = '-'.join(date_list)
    pr = PrepReport(interim_folder=interim_folder)

    # Need to test
    # pr.main(vul_id_list, thd_id_list, scenario_id_list, module_list=module_list, date_list=date_list, groupby_list=groupby_list,
    #         output_fn='test_20210425')

    # ------------------------------------------------------------------------------------------------
    # Use-case 1
    # Merge initial output of the final assessment
    # -------------------------------------------------------------------------------------------------
    pr.merge_batch(vul_id_list, thd_id_list, scenario_id_list, module_list=module_list, date_list=date_list,
                   groupby_list=groupby_list,
                   output_fn=f'final-assessment-merge_{suffix_date}{suffix}')

    # -------------------------------------------------------------------------------------------------
    # Use case 2
    # Calculate rankings for the updated merged final assessment which was updated mannually previously (need to test)
    # -------------------------------------------------------------------------------------------------
    # # Manual updated merged file
    # df = pd.read_excel(
    #     r'C:\Users\yan.cheng\PycharmProjects\EBRD\final assessment\processed\final-assessment-merge_20210507_ByPlant.xlsx',
    #     index_col=0)
    # # # Attach thresholds
    #
    # # # Calculate rankings and export
    # df_withStats = pr.cal_rankings(df=df)
    # df_withStats.to_excel(
    #     r'C:\Users\yan.cheng\PycharmProjects\EBRD\reports\assessment\test_20210507_ByPlant_report.xlsx')
