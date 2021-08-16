"""Visualize trajectories of climatic variables and frequency of climate extremes"""

import datetime
import os
import glob

import pandas as pd
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from scripts.visualizations import viz_configs as vc

# Only for gddp datasets
extreme_indicator = 'airTempMax'
trajectory_indicator = 'airTempAvg'

hist_start_year, hist_end_year = 1965, 2004
futu_start_year, futu_end_year = 2010, 2049
freq = {0: 'M', 1: 'Y', }.get(0)  # monthly average
apply_rolling = {
    1: True,
    0: False
}.get(1)
window = 12  # Moving average
work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))

at_plants = [5, 6, 8, 9, 10, 11, 12]
wt_plants = [i for i in range(1, 26) if i not in [4, 6, 8, 9, 10, 11]]
wbt_plants = [1, 3, 5, 7, 10, 12, 13, 14, 15, 19, 25]
dt_plants = [i for i in range(1, 26) if i not in [11, 9, 8, 7, 6, 4]]


def prep_data(dataset):
    # Read data in the form of pandas dataframe
    data_directory = {
        'gddp': os.path.join(work_directory, 'tpp_climate_gddp_restructure_all_withAirTempAvg'),
        'uuwt': os.path.join(work_directory, 'watertemp_output_temp_all'),
        # 'era5_gddp-wbt': {
        #     'historical': os.path.join(work_directory, 'ear5_wetbulbtemp'),
        #     'future': os.path.join(work_directory, 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols'),
        # }
    }.get(dataset)
    fp_list = glob.glob(os.path.join(data_directory, '*.csv'))
    df = pd.concat([pd.read_csv(fp, index_col=0) for fp in fp_list])

    # Data transformation or feature engineering
    if dataset == 'gddp':
        in_df = df[(df['indicator'].isin(['airTempAvg', 'airTempMax', 'airTempMin', 'pr_nexGddp']))
                   & (((df['year'] >= hist_start_year) & (df['year'] <= hist_end_year))
                      | ((df['year'] >= futu_start_year) & (df['year'] <= futu_end_year)))]
        in_df.loc[in_df['indicator'] == 'airTempAvg', ['value']] = in_df[in_df['indicator'] == 'airTempAvg'][
                                                                       'value'] - 273.15
        in_df.loc[in_df['indicator'] == 'airTempMax', ['value']] = in_df[in_df['indicator'] == 'airTempMax'][
                                                                       'value'] - 273.15
        in_df.loc[in_df['indicator'] == 'airTempMin', ['value']] = in_df[in_df['indicator'] == 'airTempMin'][
                                                                       'value'] - 273.15
        in_df['date'] = pd.to_datetime(in_df[['year', 'month', 'day']])
        in_df['plant-id_model'] = in_df['plant_id'].astype(str) + in_df['model'].astype(str)
        in_df.sort_values(by=['date', 'scenario'], inplace=True)
    elif dataset == 'uuwt':
        in_df = df[(df['indicator'].isin(['waterTemp']))
                   & (((df['year'] >= hist_start_year) & (df['year'] <= hist_end_year) & (
                df['scenario'] == 'historical'))
                      | ((df['year'] >= futu_start_year) & (df['year'] <= futu_end_year) & (
                    df['scenario'].isin(['rcp4p5', 'rcp8p5']))))]
        in_df.loc[in_df['indicator'] == 'waterTemp', ['value']] = in_df[in_df['indicator'] == 'waterTemp'][
                                                                      'value'] - 273.15
        in_df['date'] = pd.to_datetime(in_df[['year', 'month', 'day']])
        in_df['model'].fillna('NAN', inplace=True)
        in_df['plant-id_model'] = in_df['plant_id'].astype(str) + in_df['model'].astype(str)
        in_df['scenario'].replace(['rcp4p5', 'rcp8p5'], ['rcp45', 'rcp85'], inplace=True)
        in_df.sort_values(by=['date', 'scenario', 'plant_id'], inplace=True)

    # elif dataset == 'era5_gddp-wbt':
    # To be updated

    # Aggregation
    def temp_agg_sum(in_df, freq):
        """Aggregation along a temporal dimension"""
        in_df_agg = in_df.groupby(['plant_id', 'scenario', 'model', 'indicator', 'plant-id_model']).apply(
            lambda df_sub: df_sub.groupby(pd.PeriodIndex(df_sub['date'], freq=freq))[
                'value'].sum().reset_index()).reset_index()[:]
        in_df_agg['date'] = pd.to_datetime(in_df_agg['date'].astype(str))
        return in_df_agg

    def temp_agg_avg(in_df, freq):
        """Aggregation along a temporal dimension"""
        in_df_agg = in_df.groupby(['plant_id', 'scenario', 'model', 'indicator', 'plant-id_model']).apply(
            lambda df_sub: df_sub.groupby(pd.PeriodIndex(df_sub['date'], freq=freq))[
                'value'].mean().reset_index()).reset_index()[:]
        in_df_agg['date'] = pd.to_datetime(in_df_agg['date'].astype(str))
        return in_df_agg

    def temp_moving_avg(in_df, window):
        """Moving average"""
        in_df_agg = in_df.set_index('date')
        in_df_agg = in_df_agg.groupby(['plant_id', 'scenario', 'model', 'indicator', 'plant-id_model'])[
                        'value'].rolling(
            window=window).mean().reset_index()[:]
        return in_df_agg

    def temp_moving_min(in_df, window):
        """Moving comparison"""
        in_df_agg = in_df.set_index('date')
        in_df_agg = in_df_agg.groupby(['plant_id', 'scenario', 'model', 'indicator', 'plant-id_model'])[
                        'value'].rolling(
            window=window).min().reset_index()[:]
        return in_df_agg

    in_df_agg = temp_agg_avg(in_df=in_df, freq=freq)

    if dataset == 'gddp':
        in_df_agg_prAvg = temp_moving_avg(in_df_agg[in_df_agg['indicator'] == 'pr_nexGddp'], window=window)
        in_df_agg_prAvg['aggregation'] = 'avg'
        in_df_agg_prMin = temp_moving_min(in_df_agg[in_df_agg['indicator'] == 'pr_nexGddp'], window=window)
        in_df_agg_prMin['aggregation'] = 'min'
        in_df_agg_airTempAvg = temp_moving_avg(in_df_agg[in_df_agg['indicator'] == 'airTempAvg'], window=window)
        in_df_agg_airTempAvg['aggregation'] = 'avg'
        in_df_agg_airTempMax = temp_moving_avg(in_df_agg[in_df_agg['indicator'] == 'airTempMax'], window=window)
        in_df_agg_airTempMax['aggregation'] = 'avg'
        in_df_agg = pd.concat([in_df_agg_prAvg, in_df_agg_prMin, in_df_agg_airTempAvg, in_df_agg_airTempMax])
    elif dataset == 'uuwt':
        in_df_agg = temp_moving_avg(in_df_agg[in_df_agg['indicator'] == 'waterTemp'], window=window)
        in_df_agg['aggregation'] = 'avg'

    # Reorganize
    in_df_agg.sort_values(by=['date', 'scenario'], inplace=True)

    # Max and min values
    df_range_max = in_df_agg.groupby(['indicator', 'aggregation'])['value'].max().reset_index()
    df_range_min = in_df_agg.groupby(['indicator', 'aggregation'])['value'].min().reset_index()

    return in_df, in_df_agg, df_range_max, df_range_min


in_df_gddp, in_df_agg_gddp, df_range_max_gddp, df_range_min_gddp = prep_data(dataset='gddp')
in_df_uuwt, in_df_agg_uuwt, df_range_max_uuwt, df_range_min_uuwt = prep_data(dataset='uuwt')

# visualization configurations
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
PROPS = {'linewidth': 1.5}
PROPS_BG = {'linewidth': 0.5}


def viz_watertemp_timeseries(in_df_agg, df_range_min, df_range_max):
    """Visualize water temperature time series"""
    # Layout
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    y1_min = df_range_min[
        (df_range_min['indicator'] == 'waterTemp') & (df_range_max['aggregation'] == 'avg')].value.values
    y1_max = df_range_max[
        (df_range_max['indicator'] == 'waterTemp') & (df_range_max['aggregation'] == 'avg')].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    for i in [0, 1]:
        axes[i].set_ylim(y1_range)
        # axes[i].set_xticklabels([])
        axes[i].legend().set_visible(False)
    axes[0].set_ylabel('Water temperature ($^\circ$C)')
    axes[1].set_yticklabels([])
    axes[1].set_ylabel('')
    axes[0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
    axes[1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
    axes[0].set_xlabel(f'Baseline period')
    axes[1].set_xlabel(f'2030 Projection period')

    # custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
    #                 Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
    #                 Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    # leg = axes[1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def viz_watertemp_timeseries_per_plant(in_df_agg, df_range_min, df_range_max, plant_id, save_output=False):
    """Visualize time series of water temperature for each plant"""
    in_df_agg = in_df_agg[in_df_agg['plant_id'] == plant_id]

    # Layout
    fig, axes = plt.subplots(nrows=1, ncols=2)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'waterTemp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    y1_min = df_range_min[
        (df_range_min['indicator'] == 'waterTemp') & (df_range_max['aggregation'] == 'avg')].value.values
    y1_max = df_range_max[
        (df_range_max['indicator'] == 'waterTemp') & (df_range_max['aggregation'] == 'avg')].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    for i in [0, 1]:
        axes[i].set_ylim(y1_range)
        # axes[i].set_xticklabels([])
        axes[i].legend().set_visible(False)
    axes[0].set_ylabel('Water temperature ($^\circ$C)')
    axes[1].set_yticklabels([])
    axes[1].set_ylabel([])
    axes[0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
    axes[1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
    axes[0].set_xlabel(f'Baseline period ({hist_start_year}-{hist_end_year})')
    axes[1].set_xlabel(f'Projection period ({futu_start_year}-{futu_end_year})')

    custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    leg = axes[1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'Plant ID: {plant_id}')
    if save_output is True:
        fig.savefig(os.path.join(work_directory, 'reports', 'data', f'water-temperature_timeseries_{plant_id}.svg'),
                    format='svg')
    else:
        plt.show()


def viz_watertemp_plant_batch(in_df_agg, plant_id_list, save_output):
    """Visualize time series of climatic variables for each plant (batch run)"""
    return list(map(
        lambda x: viz_watertemp_timeseries_per_plant(in_df_agg=in_df_agg[in_df_agg['plant_id'] == x],
                                                     plant_id=x, save_output=save_output), plant_id_list))


# in_df_agg = in_df_agg
# plant_id_list = [i for i in in_df_agg['plant_id'].unique()]
# viz_watertemp_plant_batch(in_df_agg=in_df_agg, plant_id_list=plant_id_list, save_output=True)


def viz_climatic_variables_timeseries(in_df_agg, df_range_min, df_range_max):
    """Visualize timeseries of precipitation and air temperature from NEX-GDDP datasets for all plants and all models"""
    # Layout
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # PR
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[0][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # PR-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[1][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[1][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # AT-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[1][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    for i in [0, 1]:
        # axes[0][i].invert_yaxis()
        axes[i][0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
        axes[i][1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
        if freq == 'M':
            if apply_rolling is True and window == 12:
                axes[0][i].set_ylim([0, 0.00006])
                axes[1][i].set_ylim([-2, 22])
            else:
                axes[0][i].set_ylim([0, 0.00018])
                axes[1][i].set_ylim([-30, 40])
        elif freq == 'Y':
            axes[0][i].set_ylim([0, 0.00006])
            axes[1][i].set_ylim([-2, 22])
        axes[0][i].set_xlabel('')
        axes[1][i].set_xlabel('')
        axes[0][i].set_xticklabels([])
        axes[1][i].set_xticklabels([])
        axes[i][1].set_ylabel('')
        axes[i][1].set_yticklabels([])
        axes[0][i].legend().set_visible(False)
        axes[1][i].legend().set_visible(False)
    axes[0][0].set_ylabel('Precipitation (kg/($m^{2}$s))')
    axes[1][0].set_ylabel('Air temperature ($^\circ$C)')
    # axes[1][0].set_xlabel(f'Baseline period ({hist_start_year}-{hist_end_year})')
    # axes[1][1].set_xlabel(f'Projection period ({futu_start_year}-{futu_end_year})')

    y1_min = df_range_min[
        (df_range_min['indicator'] == 'pr_nexGddp') & (df_range_max['aggregation'] == 'avg')].value.values
    y1_max = df_range_max[
        (df_range_max['indicator'] == 'pr_nexGddp') & (df_range_max['aggregation'] == 'avg')].value.values
    y2_min = df_range_min[df_range_min['indicator'] == trajectory_indicator].value.values
    y2_max = df_range_max[df_range_max['indicator'] == trajectory_indicator].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    y2_range = [y2_min - (y2_max - y2_min) * 0.1, y2_max + (y2_max - y2_min) * 0.1]

    for i in [0, 1]:
        axes[0][i].set_ylim(y1_range)
        axes[1][i].set_ylim(y2_range)

    custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    leg = axes[0][1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def viz_climatic_variables_timeseries_v2(in_df_agg, df_range_min, df_range_max):
    """Visualize timeseries of precipitation and air temperature from NEX-GDDP datasets for all plants and all models"""
    # Layout
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # PR
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical'])) & (
                in_df_agg['aggregation'] == 'min')],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg['aggregation'] == 'min')],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[0][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # PR-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical'])) & (
                    in_df_agg['aggregation'] == 'min')],
        x='date', y='value', estimator=np.median, ax=axes[0][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg['aggregation'] == 'min')],
        x='date', y='value', estimator=np.median, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempMax') & (in_df_agg['scenario'].isin(['historical'])) & (
                    in_df_agg['aggregation'] == 'avg')],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[1][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempMax') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg['aggregation'] == 'avg')],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[1][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # AT-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempMax') & (in_df_agg['scenario'].isin(['historical'])) & (
                    in_df_agg['aggregation'] == 'avg')],
        x='date', y='value', estimator=np.median, ax=axes[1][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempMax') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg['aggregation'] == 'avg')],
        x='date', y='value', estimator=np.median, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    for i in [0, 1]:
        # axes[0][i].invert_yaxis()
        axes[i][0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
        axes[i][1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
        if freq == 'M':
            if apply_rolling is True and window == 12:
                axes[0][i].set_ylim([0, 0.00006])
                axes[1][i].set_ylim([-2, 22])
            else:
                axes[0][i].set_ylim([0, 0.00018])
                axes[1][i].set_ylim([-30, 40])
        elif freq == 'Y':
            axes[0][i].set_ylim([0, 0.00006])
            axes[1][i].set_ylim([-2, 22])
        axes[0][i].set_xlabel('')
        axes[1][i].set_xlabel('')
        axes[0][i].set_xticklabels([])
        axes[1][i].set_xticklabels([])
        axes[i][1].set_ylabel('')
        axes[i][1].set_yticklabels([])
        axes[0][i].legend().set_visible(False)
        axes[1][i].legend().set_visible(False)
    axes[0][0].set_ylabel('Precipitation (kg/($m^{2}$s))')
    axes[1][0].set_ylabel('Air temperature ($^\circ$C)')
    # axes[1][0].set_xlabel(f'Baseline period ({hist_start_year}-{hist_end_year})')
    # axes[1][1].set_xlabel(f'Projection period ({futu_start_year}-{futu_end_year})')

    y1_min = df_range_min[
        (df_range_min['indicator'] == 'pr_nexGddp') & (df_range_max['aggregation'] == 'min')].value.values
    y1_max = df_range_max[
        (df_range_max['indicator'] == 'pr_nexGddp') & (df_range_max['aggregation'] == 'min')].value.values
    y2_min = df_range_min[df_range_min['indicator'] == extreme_indicator].value.values
    y2_max = df_range_max[df_range_max['indicator'] == extreme_indicator].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    y2_range = [y2_min - (y2_max - y2_min) * 0.1, y2_max + (y2_max - y2_min) * 0.1]

    for i in [0, 1]:
        axes[0][i].set_ylim(y1_range)
        axes[1][i].set_ylim(y2_range)

    # custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
    #                 Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
    #                 Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    # leg = axes[0][1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def viz_climatic_variables_timeseries_model_median(in_df_agg):
    """Visualize timeseries of precipitation and air temperature from NEX-GDDP datasets for all plants (model median)"""

    in_df_agg_med = in_df_agg.groupby(['plant_id', 'scenario', 'indicator', 'date'])['value'].median().reset_index()
    df_range_max = in_df_agg_med.groupby('indicator')['value'].max().reset_index()
    df_range_min = in_df_agg_med.groupby('indicator')['value'].min().reset_index()

    fig, axes = plt.subplots(nrows=2, ncols=2)
    # PR
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'pr_nexGddp') & (in_df_agg_med['scenario'].isin(['historical']))],
        x='date', y='value', units='plant_id', estimator=None, ax=axes[0][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'pr_nexGddp') & (in_df_agg_med['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant_id', estimator=None, hue='scenario', ax=axes[0][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # PR-median
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'pr_nexGddp') & (in_df_agg_med['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'pr_nexGddp') & (in_df_agg_med['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'airTempAvg') & (in_df_agg_med['scenario'].isin(['historical']))],
        x='date', y='value', units='plant_id', estimator=None, ax=axes[1][0], color=vc.WRI_COLOR_SCHEME['gray'])
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg_med['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant_id', estimator=None, hue='scenario', ax=axes[1][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # AT-median
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'airTempAvg') & (in_df_agg_med['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[1][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_med[
            (in_df_agg_med['indicator'] == 'airTempAvg') & (in_df_agg_med['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    for i in [0, 1]:
        # axes[0][i].invert_yaxis()
        axes[i][0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
        axes[i][1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
        if freq == 'M':
            if apply_rolling is True and window == 12:
                axes[0][i].set_ylim([0, 0.00006])
                axes[1][i].set_ylim([-2, 22])
            else:
                axes[0][i].set_ylim([0, 0.00018])
                axes[1][i].set_ylim([-30, 40])
        elif freq == 'Y':
            axes[0][i].set_ylim([0, 0.00006])
            axes[1][i].set_ylim([-2, 22])
        axes[0][i].set_xlabel('')
        axes[1][i].set_xlabel('')
        axes[0][i].set_xticklabels([])
        axes[1][i].set_xticklabels([])
        axes[i][1].set_ylabel('')
        axes[i][1].set_yticklabels([])
        axes[0][i].legend().set_visible(False)
        axes[1][i].legend().set_visible(False)
    axes[0][0].set_ylabel('Precipitation (kg/($m^{2}$s))')
    axes[1][0].set_ylabel('Air temperature ($^\circ$C)')
    # axes[1][0].set_xlabel(f'Baseline period ({hist_start_year}-{hist_end_year})')
    # axes[1][1].set_xlabel(f'Projection period ({futu_start_year}-{futu_end_year})')

    y1_min = df_range_min[df_range_min['indicator'] == 'pr_nexGddp'].value.values
    y1_max = df_range_max[df_range_max['indicator'] == 'pr_nexGddp'].value.values
    y2_min = df_range_min[df_range_min['indicator'] == trajectory_indicator].value.values
    y2_max = df_range_max[df_range_max['indicator'] == trajectory_indicator].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    y2_range = [y2_min - (y2_max - y2_min) * 0.1, y2_max + (y2_max - y2_min) * 0.1]

    for i in [0, 1]:
        axes[0][i].set_ylim(y1_range)
        axes[1][i].set_ylim(y2_range)

    custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    leg = axes[0][1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def viz_climatic_variables_timeseries_per_plant(in_df_agg, plant_id, save_output=False):
    """Visualize time series of climatic variables for each plant"""
    in_df_agg = in_df_agg[in_df_agg['plant_id'] == plant_id]

    # Layout
    fig, axes = plt.subplots(nrows=2, ncols=2)
    # PR
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['gray']])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # PR-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'pr_nexGddp') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[1][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['gray']])
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']])
    # AT-median
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[1][0], hue='scenario', palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg[(in_df_agg['indicator'] == 'airTempAvg') & (in_df_agg['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    for i in [0, 1]:
        # axes[0][i].invert_yaxis()
        axes[i][0].set_xlim([datetime.date(hist_start_year, 1, 1), datetime.date(hist_end_year, 12, 1)])
        axes[i][1].set_xlim([datetime.date(futu_start_year, 1, 1), datetime.date(futu_end_year, 12, 1)])
        if freq == 'M':
            if apply_rolling is True and window == 12:
                axes[0][i].set_ylim([0, 0.00006])
                axes[1][i].set_ylim([-2, 22])
            else:
                axes[0][i].set_ylim([0, 0.00018])
                axes[1][i].set_ylim([-30, 40])
        elif freq == 'Y':
            axes[0][i].set_ylim([0, 0.00006])
            axes[1][i].set_ylim([-2, 22])
        axes[0][i].set_xlabel('')
        axes[1][i].set_xlabel('')
        axes[0][i].set_xticklabels([])
        axes[1][i].set_xticklabels([])
        axes[i][1].set_ylabel('')
        axes[i][1].set_yticklabels([])
        axes[0][i].legend().set_visible(False)
        axes[1][i].legend().set_visible(False)
    axes[0][0].set_ylabel('Precipitation (kg/($m_{2}$s))')
    axes[1][0].set_ylabel('Air temperature ($^\circ$C)')
    # axes[1][0].set_xlabel(f'Baseline period ({hist_start_year}-{hist_end_year})')
    # axes[1][1].set_xlabel(f'Projection period ({futu_start_year}-{futu_end_year})')

    custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    leg = axes[0][1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.suptitle(f'Plant ID: {plant_id}')
    if save_output is True:
        fig.savefig(os.path.join(work_directory, 'reports', 'data', f'timeseries_{plant_id}.svg'), format='svg')
    else:
        plt.show()


def viz_plant_batch(in_df_agg, plant_id_list, save_output):
    """Visualize time series of climatic variables for each plant (batch run)"""
    return list(map(
        lambda x: viz_climatic_variables_timeseries_per_plant(in_df_agg=in_df_agg[in_df_agg['plant_id'] == x],
                                                              plant_id=x, save_output=save_output), plant_id_list))


# in_df_agg = in_df_agg
# viz_plant_batch(in_df_agg=in_df_agg, plant_id_list=range(1, 26), save_output=True)


def all_in_one():
    # Layout
    fig, axes = plt.subplots(nrows=5, ncols=2)

    # Air temperature and precipitation
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[0][0],
        color=vc.WRI_COLOR_SCHEME['gray'], **PROPS_BG)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[0][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']], **PROPS_BG)
    # PR-median
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[0][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[0][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempAvg') & (in_df_agg_gddp['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[1][0],
        color=vc.WRI_COLOR_SCHEME['gray'], **PROPS_BG)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempAvg') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[1][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']], **PROPS_BG)
    # AT-median
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempAvg') & (in_df_agg_gddp['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[1][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempAvg') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[1][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    axes[0][0].set_ylabel('Precipitation')
    axes[1][0].set_ylabel('Avg. air temp.')
    y1_min = df_range_min_gddp[
        (df_range_min_gddp['indicator'] == 'pr_nexGddp') & (df_range_min_gddp['aggregation'] == 'avg')].value.values
    y1_max = df_range_max_gddp[
        (df_range_max_gddp['indicator'] == 'pr_nexGddp') & (df_range_max_gddp['aggregation'] == 'avg')].value.values
    y2_min = df_range_min_gddp[df_range_min_gddp['indicator'] == trajectory_indicator].value.values
    y2_max = df_range_max_gddp[df_range_max_gddp['indicator'] == trajectory_indicator].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    y2_range = [y2_min - (y2_max - y2_min) * 0.1, y2_max + (y2_max - y2_min) * 0.1]
    for i in [0, 1]:
        axes[0][i].set_ylim(y1_range)
        axes[1][i].set_ylim(y2_range)

    # Air temperature and precipitation (extreme conditions)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['historical'])) & (
                    in_df_agg_gddp['aggregation'] == 'min')],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[2][0],
        color=vc.WRI_COLOR_SCHEME['gray'], **PROPS_BG)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg_gddp['aggregation'] == 'min')],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[2][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']], **PROPS_BG)
    # PR-median
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['historical'])) & (
                    in_df_agg_gddp['aggregation'] == 'min')],
        x='date', y='value', estimator=np.median, ax=axes[2][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'pr_nexGddp') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg_gddp['aggregation'] == 'min')],
        x='date', y='value', estimator=np.median, ax=axes[2][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)
    # AT
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempMax') & (in_df_agg_gddp['scenario'].isin(['historical'])) & (
                    in_df_agg_gddp['aggregation'] == 'avg')],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[3][0],
        color=vc.WRI_COLOR_SCHEME['gray'], **PROPS_BG)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempMax') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg_gddp['aggregation'] == 'avg')],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[3][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']], **PROPS_BG)
    # AT-median
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempMax') & (in_df_agg_gddp['scenario'].isin(['historical'])) & (
                    in_df_agg_gddp['aggregation'] == 'avg')],
        x='date', y='value', estimator=np.median, ax=axes[3][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_gddp[
            (in_df_agg_gddp['indicator'] == 'airTempMax') & (in_df_agg_gddp['scenario'].isin(['rcp45', 'rcp85'])) & (
                    in_df_agg_gddp['aggregation'] == 'avg')],
        x='date', y='value', estimator=np.median, ax=axes[3][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    axes[2][0].set_ylabel('Min. precipitation')
    axes[3][0].set_ylabel('Max. air temp.')
    y1_min = df_range_min_gddp[
        (df_range_min_gddp['indicator'] == 'pr_nexGddp') & (df_range_min_gddp['aggregation'] == 'min')].value.values
    y1_max = df_range_max_gddp[
        (df_range_max_gddp['indicator'] == 'pr_nexGddp') & (df_range_max_gddp['aggregation'] == 'min')].value.values
    y2_min = df_range_min_gddp[df_range_min_gddp['indicator'] == extreme_indicator].value.values
    y2_max = df_range_max_gddp[df_range_max_gddp['indicator'] == extreme_indicator].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    y2_range = [y2_min - (y2_max - y2_min) * 0.1, y2_max + (y2_max - y2_min) * 0.1]
    for i in [0, 1]:
        axes[2][i].set_ylim(y1_range)
        axes[3][i].set_ylim(y2_range)

    # Water temperature
    sns.lineplot(
        data=in_df_agg_uuwt[
            (in_df_agg_uuwt['indicator'] == 'waterTemp') & (in_df_agg_uuwt['scenario'].isin(['historical']))],
        x='date', y='value', units='plant-id_model', estimator=None, ax=axes[4][0], color=vc.WRI_COLOR_SCHEME['gray'],
        **PROPS_BG)
    sns.lineplot(
        data=in_df_agg_uuwt[
            (in_df_agg_uuwt['indicator'] == 'waterTemp') & (in_df_agg_uuwt['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', units='plant-id_model', estimator=None, hue='scenario', ax=axes[4][1],
        palette=[vc.WRI_COLOR_SCHEME['gray'], vc.WRI_COLOR_SCHEME['gray']], **PROPS_BG)
    # median
    sns.lineplot(
        data=in_df_agg_uuwt[
            (in_df_agg_uuwt['indicator'] == 'waterTemp') & (in_df_agg_uuwt['scenario'].isin(['historical']))],
        x='date', y='value', estimator=np.median, ax=axes[4][0], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['black']],
        **PROPS)
    sns.lineplot(
        data=in_df_agg_uuwt[
            (in_df_agg_uuwt['indicator'] == 'waterTemp') & (in_df_agg_uuwt['scenario'].isin(['rcp45', 'rcp85']))],
        x='date', y='value', estimator=np.median, ax=axes[4][1], hue='scenario',
        palette=[vc.WRI_COLOR_SCHEME['blue'], vc.WRI_COLOR_SCHEME['blue-dark']], **PROPS)

    # Visualization assessories
    axes[4][0].set_ylabel('Water temp.')
    y1_min = df_range_min_uuwt[
        (df_range_min_uuwt['indicator'] == 'waterTemp') & (df_range_min_uuwt['aggregation'] == 'avg')].value.values
    y1_max = df_range_max_uuwt[
        (df_range_max_uuwt['indicator'] == 'waterTemp') & (df_range_max_uuwt['aggregation'] == 'avg')].value.values
    y1_range = [y1_min - (y1_max - y1_min) * 0.1, y1_max + (y1_max - y1_min) * 0.1]
    for i in [0, 1]:
        axes[4][i].set_ylim(y1_range)

    for c, r in list(itertools.product(range(0, 2), range(0, 5))):
        # axes[0][i].invert_yaxis()
        axes[r][0].set_xlim([datetime.date(hist_start_year, 12, 1), datetime.date(hist_end_year, 12, 1)])
        axes[r][1].set_xlim([datetime.date(futu_start_year, 12, 1), datetime.date(futu_end_year, 12, 1)])
        axes[r][c].set_xlabel('')
        axes[r][1].set_ylabel('')
        axes[r][1].set_yticklabels([])
        axes[r][c].legend().set_visible(False)
        if r != 4:
            axes[r][c].set_xticklabels([])

    axes[4][0].set_xlabel('Baseline period')
    axes[4][1].set_xlabel('2030 Projection period')
    custom_lines = [Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['black'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue'], lw=2),
                    Line2D([0], [0], color=vc.WRI_COLOR_SCHEME['blue-dark'], lw=2)]
    leg = axes[0][1].legend(custom_lines, ['Historical', 'RCP4.5', 'RCP8.5'], ncol=1, frameon=False, fontsize=11)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    print('Done with visualizing climate trajectories. Please adjust figure dimentions and '
          'save it to ../reports/assessment.')


def viz_climate_extremes_histogram(in_df, save_output=None):
    """Visualize histograms of climate variables"""
    thd = 36.6
    models = ['inmcm4', 'MPI-ESM-LR', 'MRI-CGCM3', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M', 'GFDL-ESM2M', 'CCSM4',
              'CNRM-CM5']
    fig, axes = plt.subplots(nrows=1, ncols=2)

    def main(args):
        plant_id, model = args
        data = in_df[
            (in_df['indicator'] == extreme_indicator) & (in_df['plant_id'] == plant_id) & (in_df['model'] == model) & (
                        in_df['scenario'] != 'historical')]
        sns.histplot(data, x='value', element='step', stat='count', hue='scenario', bins=np.arange(-5, 40, 1),
                     ax=axes[1], color='gray')
        axes[1].vlines(thd, 0, 1000, color='black', ls=':')
        # Historical
        data_hist = in_df[
            (in_df['indicator'] == extreme_indicator) & (in_df['plant_id'] == plant_id) & (in_df['model'] == model) & (
                        in_df['scenario'] == 'historical')]
        sns.histplot(data_hist, x='value', element='step', stat='count', bins=np.arange(-5, 40, 1), ax=axes[0],
                     color='gray')
        # hp_heights = [[h.xy[0], h.get_width(), h.get_height()] for h in sns.histplot(data=data_hist, bins=np.arange(-5, 40, 1)).patches]
        # thd_height = [hp_heights[i][2] for i in range(len(hp_heights)) if hp_heights[i][0] == thd][0]
        axes[0].vlines(thd, 0, 1000, color='black', ls=':')

    list(map(main, list(itertools.product(range(1, 26), models))))

    if save_output is True:
        fig.savefig(os.path.join(work_directory, 'reports', 'data', 'histogram_climate-extremes_gddp.svg'),
                    format='svg')
    else:
        plt.show()


def viz_climate_extremes_frequency(in_df, save_output=None):
    """Visualize the frequency in days of extreme weathers (i.e., above 36.6 degC)"""
    thd = 36.6
    in_df_sub = in_df[in_df['indicator'] == extreme_indicator]
    in_df_sub['extreme'] = in_df_sub['value'].apply(lambda x: x >= thd)
    in_df_sub_agg = in_df_sub.groupby(['plant_id', 'scenario', 'model', 'indicator', 'plant-id_model'])[
        'extreme'].sum().reset_index()
    extreme_med = in_df_sub_agg.groupby(['scenario', 'indicator'])['extreme'].median().reset_index()

    fig, ax = plt.subplots()
    g1 = sns.stripplot(x="scenario", y="extreme", color='black', size=5, alpha=0.3, data=in_df_sub_agg, ax=ax)
    ax.scatter(y=extreme_med['extreme'], x=extreme_med['scenario'], marker='+', s=80, c=vc.WRI_COLOR_SCHEME['blue'])
    g1.set_xlabel('')
    g1.set_xticklabels([f'Baseline ({hist_start_year}-{hist_end_year})',
                        f'RCP4.5 ({futu_start_year}-{futu_end_year})',
                        f'RCP8.5 ({futu_start_year}-{futu_end_year})'])
    g1.set_ylabel(f'Number of days above {thd} $^\circ$C')
    g1.set_xlim([-0.2, 2.2])
    g1.set_ylim([0, in_df_sub_agg['extreme'].max() * 1.01])
    plt.subplots_adjust(wspace=0, hspace=0)

    if save_output is True:
        output_directory = os.path.join(work_directory, 'reports', 'data')
        if not output_directory:
            os.mkdir(output_directory)
        fig.savefig(os.path.join(output_directory, 'climate_extreme_frequency.svg', format='svg', dpi=300))
    else:
        plt.show()


def viz_prob_cdf(variable, thd=None, indicator=None, save_output=None, xlabel='value'):
    """
    Visualize histograms of climate variables.
    :param variable:
    :param thd: thresholds in degree Celsius in the case of temperature thresholds
    :param indicator:
    :param save_output:
    :param xlabel:
    :return:
    """

    in_df = {
        'air temperature': in_df_gddp,
        # 'wet-bulb temperature': in_df_wbt,
        'water temperature': in_df_uuwt,
    }.get(variable)
    model_list = [i for i in in_df.model.unique()]
    indicator = extreme_indicator if indicator is None else indicator
    thd_fp = {
        'air temperature': os.path.join(work_directory, 'reports', 'data', 'air-temperature_p90_2010-2049.csv'),
        'wet-bulb temperature': os.path.join(work_directory, 'reports', 'data',
                                             'wet-bulb-temperature_p99_2010-2049.csv'),
        'water temperature': os.path.join(work_directory, 'reports', 'data', 'water-temperature_p90_2010-2049.csv'),
    }.get(variable)
    plant_list = {
        'air temperature': at_plants,
        'wet-bulb temperature': wbt_plants,
        'water temperature': wt_plants,
    }.get(variable)

    if thd is None:
        thd_df = pd.read_csv(thd_fp, index_col=0)
        thd_df['threshold'] = thd_df['threshold'] - 273.15

    fig, axes = plt.subplots(nrows=1, ncols=1)

    def main(args):
        plant_id, model = args
        data = in_df[(in_df['indicator'] == indicator) & (in_df['plant_id'] == plant_id) & (in_df['model'] == model)]
        g1 = sns.ecdfplot(data, x='value', stat='proportion', hue='scenario', ax=axes)
        in_thd = thd if thd is not None else thd_df[thd_df['plant_id'] == plant_id].threshold.values[0]
        axes.vlines(in_thd, 0, 1, color='black', ls=':', label='Desired air temp.')
        return g1

    g = list(map(main, list(itertools.product(plant_list, model_list))))
    g[-1].set(xlabel=xlabel, ylabel='Probability')
    g[-1].legend(labels=['Historical', 'RCP4.5', 'RCP8.5'])
    plt.show()

    if save_output is True:
        fig.savefig(os.path.join(work_directory, 'reports', 'data', 'climate-shifts-probability.svg'),
                    format='svg')


if __name__ == '__main__':
    all_in_one()
    # viz_climate_extremes_histogram(in_df=in_df_gddp)
    # viz_prob_cdf(variable='air temperature', xlabel='Air temp.', thd=295 - 273.15, indicator='airTempAvg')
