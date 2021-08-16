"""
Visualize exceedance probability of design air temperature and design water temperature in future periods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scripts.visualizations import viz_configs as vc

work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
output_directory = os.path.join(work_directory, 'reports', 'assessment')

plt.rcParams["figure.figsize"] = (10, 5)


def prep_data(variable):
    """
    Preprocess datasets

    :param variable: string, climate variable, any value in ['desired air temperature', 'desired water temperature', 'wet-bulb temperature threshold', 'discharge water limit']
    :return: in_df, plant_seq, bottom, in_df -> pandas dataframe, plant_seq -> a list of plant id, bottom -> historical exceedance probability of each plant.
    """

    data_directory = os.path.join(work_directory, 'reports', 'data')
    fp = {
        'desired air temperature': os.path.join(data_directory, 'air-temperature_p90_2010-2049.csv'),
        'wet-bulb temperature threshold': os.path.join(data_directory, 'wet-bulb-temperature_p99_2010-2049.csv'),
        'desired water temperature': os.path.join(data_directory, 'water-temperature_p90_2010-2049.csv'),
        'discharge water limit': os.path.join(data_directory, 'water-temperature_305.15_2010-2049.csv'),
    }.get(variable)  # to be set as parameters # output of data_stats.py # to be updated

    df = pd.read_csv(fp, index_col=0)
    df['threshold_median'] = df.groupby(['plant_id', 'scenario'])['threshold'].transform(np.median)
    df.drop(['model', 'percentile', 'threshold'], axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.sort_values(by=['percentile_median', 'plant_id', 'scenario'], inplace=True)

    df_typ = df.groupby(['scenario'])[['percentile_median', 'threshold_median']].median().reset_index()
    df_typ['plant_id'] = 0
    in_df = pd.concat([df, df_typ[[i for i in df.columns]]])
    in_df.sort_values(by=['scenario', 'percentile_median', 'plant_id'], inplace=True)

    n_plant = len(list(in_df.plant_id.unique()))

    if variable != 'discharge water limit':
        bottom = {
            'desired air temperature': [0.1] * n_plant,
            'wet-bulb temperature threshold': [0.01] * n_plant,
            'desired water temperature': [0.1] * n_plant
        }.get(variable)
        in_df['increase'] = in_df['percentile_median'] <= 1 - bottom[0]
        in_df.sort_values(by=['increase', 'scenario', 'percentile_median', 'plant_id'],
                          ascending=[False, True, True, True],
                          inplace=True)
        plant_seq = list(in_df[in_df['scenario'] == 'rcp45']['plant_id'].values)
    else:
        in_df_pivot = in_df.pivot(columns='scenario', index='plant_id', values='percentile_median').reset_index()
        in_df_pivot['rcp85'] = in_df_pivot['rcp85'] - in_df_pivot['historical'] <= 0
        in_df_pivot['rcp45'] = in_df_pivot['rcp45'] - in_df_pivot['historical'] <= 0
        in_df_increase = pd.melt(in_df_pivot, id_vars='plant_id', value_vars=['rcp45', 'rcp85'], value_name='increase',
                                 var_name='scenario')
        in_df = in_df.merge(in_df_increase, on=['plant_id', 'scenario'], how='left')
        in_df.sort_values(by=['increase', 'scenario', 'percentile_median', 'plant_id'],
                          ascending=[False, True, True, True], inplace=True)

        plant_seq = list(in_df[in_df['scenario'] == 'rcp45']['plant_id'].values)
        bottom = [1 - in_df[(in_df['scenario'] == 'historical') & (in_df['plant_id'] == i)].percentile_median.values for
                  i in plant_seq]

    return in_df, plant_seq, bottom


def lollipop_plot(variable_list, save_fig=None, save_table=False):
    """
    Visualize exceedance probability using lollipop charts.

    :param variable_list: list, a list of variables to be visualized, a subset of ['desired air temperature', 'desired water temperature', 'wet-bulb temperature threshold', 'discharge water limit']
    :param save_fig: boolean, whether to save figures.
    :param save_table: boolean, whether to save input data tables.
    :return: None
    """

    markersize = 5
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    color = {
        'rcp45': vc.WRI_COLOR_SCHEME['blue'],
        'rcp85': vc.WRI_COLOR_SCHEME['blue-dark'],
        'med': vc.WRI_COLOR_SCHEME['orange'],
    }

    n_cols = len(variable_list)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols)

    def viz_main(ax, variable):
        in_df, plant_seq, bottom_list = prep_data(variable)
        med_idx = plant_seq.index(0)
        plant_seq = [i for i in plant_seq if i != 0]
        in_df = in_df[in_df['plant_id'] != 0]
        bottom_list = [b for idx, b in enumerate(bottom_list) if idx != med_idx]
        in_df['increase'].replace([True, False], [10, 11], inplace=True)
        x = range(len(plant_seq))

        if save_table is True:
            in_df['Exceedance Probability'] = 1 - in_df['percentile_median']
            df_hist = pd.DataFrame(list(zip(plant_seq, ['historical'] * len(plant_seq), bottom_list)),
                                   columns=['plant_id', 'scenario', 'Exceedance Probability'])
            df_exp = df_hist.append(in_df[['plant_id', 'scenario', 'Exceedance Probability']])
            df_exp.columns = ['Reference ID Number of Each Plant', 'Scenario', 'Exceedance Probability']
            df_exp['Variable'] = variable.replace('desired', 'design')
            table_report_folder = os.path.join(work_directory, 'reports', 'tables')
            if not os.path.exists(table_report_folder):
                os.mkdir(table_report_folder)
            df_exp.to_csv(os.path.join(table_report_folder,
                                       'probability_changes_{}.csv'.format('-'.join(variable.split(' ')))), index=False)

        plot_title = {
            'desired air temperature': 'design air temperature',
            'wet-bulb temperature threshold': 'design wet-bulb temperature threshold',
            'desired water temperature': 'design water temperature',
            'discharge water limit': 'discharge water limit',
        }.get(variable)

        # Bars
        for idx, plant_id in enumerate(plant_seq):
            height = 1 - \
                     in_df[(in_df['scenario'] == 'rcp85') & (in_df['plant_id'] == plant_id)].percentile_median.values[
                         0] - (1 - in_df[
                (in_df['scenario'] == 'rcp45') & (in_df['plant_id'] == plant_id)].percentile_median.values[0])
            min_ = min(
                1 - in_df[(in_df['scenario'] == 'rcp85') & (in_df['plant_id'] == plant_id)].percentile_median.values[0],
                1 - in_df[(in_df['scenario'] == 'rcp45') & (in_df['plant_id'] == plant_id)].percentile_median.values[0])
            max_ = max(
                1 - in_df[(in_df['scenario'] == 'rcp85') & (in_df['plant_id'] == plant_id)].percentile_median.values[0],
                1 - in_df[(in_df['scenario'] == 'rcp45') & (in_df['plant_id'] == plant_id)].percentile_median.values[0])
            markerline, stemlines, baseline = ax.stem([x[idx]], [max_], bottom=min_, markerfmt='o', label='')  # RCP8.5
            plt.setp(stemlines, 'color', vc.WRI_COLOR_SCHEME['blue-dark'])
            plt.setp(stemlines, 'linewidth', 2)
            plt.setp(baseline, 'alpha', 0.0)
            plt.setp(markerline, 'color', vc.WRI_COLOR_SCHEME['blue-dark'])
            plt.setp(markerline, 'markersize', markersize)

        # Lollipops
        for scenario in ['rcp85', 'rcp45']:
            for idx, plant_id in enumerate(plant_seq):
                percentile_median = 1 - in_df[
                    (in_df['scenario'] == scenario) & (in_df['plant_id'] == plant_id)].percentile_median.values[0]
                bottom = bottom_list[idx]
                marker = in_df[(in_df['scenario'] == scenario) & (in_df['plant_id'] == plant_id)].increase.values[0]
                label = None
                if plant_id == 0:
                    label = 'RCP4.5 median' if scenario == 'rcp45' else 'RCP8.5 median'
                elif plant_id == [i for i in plant_seq if i != 0][0]:
                    label = 'RCP4.5' if scenario == 'rcp45' else 'RCP8.5'
                markerline, stemlines, baseline = ax.stem([x[idx]], [percentile_median], bottom=bottom,
                                                          markerfmt='o', label=label)  # RCP8.5
                color_code = scenario if plant_id != 0 else 'med'
                # plt.setp(markerline, 'marker', marker)
                plt.setp(stemlines, 'color', color.get(color_code))
                plt.setp(baseline, 'alpha', 0.0)
                plt.setp(markerline, 'color', color.get(color_code))
                plt.setp(markerline, 'markersize', markersize)
                plt.setp(stemlines, 'linestyle', 'dotted')
                # ax.annotate(str(plant_id), xy=(idx, percentile_median), xytext=(idx, percentile_median+0.01))

        # Baseline markers
        ax.scatter(x, bottom_list, color=vc.WRI_COLOR_SCHEME['black'], marker='o', s=markersize ** 2,
                   label='Historical')

        ax.set_xlabel('Reference ID number of each plant')
        ax.set_xlim([-1, len(plant_seq)])
        ax.set_ylim([0, 0.3])
        ax.set_title(plot_title.capitalize())
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in ax.get_yticks()])
        ax.set_xticks(range(-1, len(plant_seq)))
        ax.set_xticklabels([''] + [i if i != 0 else 'med' for i in plant_seq])
        return ax

    if n_cols == 1:
        variable = variable_list[0]
        viz_main(ax=axes, variable=variable)
        legend_loc = 'upper right' if variable != 'discharge water limit' else 'best'
        axes.set_ylabel(f'Exceedance probability')
        plt.legend(frameon=False, loc=legend_loc)
        plt.show()
        if save_fig is True:
            fig_name = '-'.join(variable.split(' '))
            fig.savefig(os.path.join(output_directory, f'prob-chg_{fig_name}.svg'))
    else:
        list(map(lambda x: viz_main(ax=axes[variable_list.index(x)], variable=x), variable_list))
        axes[0].set_ylabel(f'Exceedance probability')
        plt.legend(frameon=False, loc='upper right')
        plt.subplots_adjust(wspace=0.01, hspace=0)
        fig.tight_layout()
        plt.show()
        if save_fig is True:
            fig_name = 'paper'
            fig.savefig(os.path.join(output_directory, f'prob-chg_{fig_name}.svg'))
            print('Done with visualizing exceedence probability of design water and air temperature. '
                  'Figure is saved here {}'.format(os.path.join(output_directory, f'prob-chg_{fig_name}.svg')))


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # User-defined parameter -> plot which variable?

    variable_list = ['desired air temperature', 'desired water temperature',
                     # 'wet-bulb temperature threshold',
                     # 'discharge water limit'
                     ]

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Use case 1: individual plot

    # list(map(lambda x: lollipop_plot(variable_list=[x], save_fig=True), variable_list))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Use case 2: all in one

    lollipop_plot(variable_list=variable_list, save_fig=False, save_table=True)
