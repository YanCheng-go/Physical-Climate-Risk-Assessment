"""
Post-processing final assessment
merged -> master

use merged final output from report_stats.py as the input of this process
"""

import os
import pandas as pd
import numpy as np
import sys

from scripts.utils import utils


class MasterReport:
    def __init__(self, in_fn, out_fn=None, **kwargs):
        """
        :param in_fn: file name-like string,  with the extension, merged final assessment excel file.
        :param out_fn: file name-like string,  output excel file name, the default name is 'master'.
        :keyword work_directory: folder path-like string, work directory.
        """

        self.work_directory = utils.kwargs_generator(kwargs, 'work_directory',
                                                     os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)))
        self.in_fn = in_fn
        self.in_fp = os.path.join(self.work_directory, 'final assessment', 'processed', self.in_fn)
        self.out_fn = 'master' if out_fn is None else out_fn
        self.out_fp = os.path.join(self.work_directory, 'reports', 'assessment', self.out_fn + '.xlsx')

    def generate_master_report(self, year_suffix='', year_suffix_water_stress=''):
        in_df_raw = pd.read_excel(self.in_fp, index_col=0)
        in_df = in_df_raw[((in_df_raw['thd_id'] == 19) & (
            in_df_raw['Risk Type'].isin(['water temperature', 'air temperature', 'flood', 'drought']))) | (
                                  in_df_raw['thd_id'] == 21)]
        group_cols = ['Power Plant Name', 'Group', 'Country', 'Capacity', 'Statistic Type', 'vul_id', 'thd_id',
                      'scenario_id']
        value_cols = ['Baseline Value', 'Projection Value']
        out_df = in_df.groupby(group_cols)[value_cols].sum().reset_index().rename(
            columns={'Baseline Value': 'Total baseline value', 'Projection Value': 'Total projected value'})
        out_df['thd_id'] = out_df['thd_id'].replace([19, 21], ['natural', 'regulatory'])
        out_df['scenario_id'] = out_df['scenario_id'].replace([45, 85], ['RCP4.5', 'RCP8.5'])
        out_df = out_df.pivot(index=[i for i in group_cols if i != 'thd_id'], columns=['thd_id'],
                              values=['Total baseline value', 'Total projected value']).reset_index()
        out_df = out_df.drop('vul_id', axis=1)
        out_df.columns = ['Power Plant Name', 'Plant ID', 'Country', 'Capacity', 'Statistic Type', 'Scenario'] + [
            'Baseline natural generation losses (MWh)', 'Baseline total generation losses (MWh)',
            'Projected natural generation losses (MWh)', 'Projected total generation losses (MWh)']

        # Feature engineering
        regroup_info = pd.read_excel(
            os.path.join(self.work_directory, 'tpp info', 'tpp_regroup.xlsx'), engine='openpyxl')
        regroup_info.rename(columns={'Group': 'Plant ID'}, inplace=True)
        out_df = out_df.merge(regroup_info[['Plant ID', 'Cooling', 'Fuel-Turbine']], on='Plant ID', how='outer')

        out_df['Baseline regulatory generation losses (MWh)'] = out_df['Baseline total generation losses (MWh)'] - \
                                                                out_df['Baseline natural generation losses (MWh)']
        out_df['Projected regulatory generation losses (MWh)'] = out_df['Projected total generation losses (MWh)'] - \
                                                                 out_df['Projected natural generation losses (MWh)']

        variable_change_df = pd.read_csv(
            os.path.join(self.work_directory, 'reports', 'data', f'change_wt-at-dt-wbt{year_suffix}.csv'), index_col=0)
        variable_change_df.rename(
            columns={'plant_id': 'Plant ID', 'scenario': 'Scenario', 'statistic': 'Statistic Type'}, inplace=True)
        wt_change_df = variable_change_df[variable_change_df['indicator'] == 'waterTemp'].rename(
            columns={'value': '% change of water temp.'}).drop('indicator', axis=1)
        wbt_change_df = variable_change_df[variable_change_df['indicator'] == 'wet_bulb_temperature'].rename(
            columns={'value': '% change of wet-bulb air temp.'}).drop('indicator', axis=1)
        dt_change_df = variable_change_df[variable_change_df['indicator'] == 'spei_12'].rename(
            columns={'value': '% change of SPEI (<=-2)'}).drop('indicator', axis=1)
        at_change_df = variable_change_df[variable_change_df['indicator'] == 'AirTempAvg'].rename(
            columns={'value': '% change of air temp.'}).drop('indicator', axis=1)
        ws_change_df = pd.read_csv(
            os.path.join(self.work_directory, 'reports', 'data', f'change_ws-srr{year_suffix_water_stress}.csv'),
            index_col=0).drop(['indicator', 'variable'], axis=1)
        ws_change_df.rename(
            columns={'value': '% change of water availability', 'plant_id': 'Plant ID', 'scenario': 'Scenario',
                     'statistic': 'Statistic Type'}, inplace=True)
        out_df = out_df.merge(wt_change_df, on=['Plant ID', 'Statistic Type', 'Scenario'], how='left') \
            .merge(wbt_change_df, on=['Plant ID', 'Statistic Type', 'Scenario'], how='left') \
            .merge(dt_change_df, on=['Plant ID', 'Statistic Type', 'Scenario'], how='left') \
            .merge(at_change_df, on=['Plant ID', 'Statistic Type', 'Scenario'], how='left') \
            .merge(ws_change_df, on=['Plant ID', 'Statistic Type'], how='left')

        out_df = out_df[['Power Plant Name', 'Plant ID', 'Country', 'Fuel-Turbine', 'Cooling', 'Capacity',
                         'Statistic Type', 'Scenario', '% change of water temp.', '% change of wet-bulb air temp.',
                         '% change of air temp.', '% change of SPEI (<=-2)', '% change of water availability',
                         'Baseline natural generation losses (MWh)', 'Baseline total generation losses (MWh)',
                         'Baseline regulatory generation losses (MWh)', 'Projected natural generation losses (MWh)',
                         'Projected total generation losses (MWh)', 'Projected regulatory generation losses (MWh)']]

        out_df.to_excel(self.out_fp, engine='openpyxl')
        print(f'Find the output here: {self.out_fp}')


if __name__ == '__main__':
    in_fn = 'final-assessment-merge_20210519_ByPlant.xlsx'
    mr = MasterReport(in_fn=in_fn, out_fn='master')
    mr.generate_master_report(year_suffix='_2010-2049', year_suffix_water_stress='_2030')
