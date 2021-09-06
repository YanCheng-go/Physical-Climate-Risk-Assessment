"""
Boxplot for visualizing portfolio-level statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scripts.visualizations import viz_configs as vc
import itertools


def main(work_directory, report_excel_fp, year_id, thd_id, group, vul_id, save_fig, fig_suffix, title_suffix='',
         fig_format='svg'):
    """
    :param work_directory: folder path-like string, work directory,
    :param report_excel_fp: file path-like string, where the post-processed final assessment excel saved
    :param year_id: int, reference number id of assessment over different projection period, herein we only assessed over the period 2030 (from 2010-2049)
    :param thd_id: int, reference number if of threshold settings.
    :param group: string, either Fuel-Turbine or 'Cooling', denoting the way of grouping, i.e., by fuel-turbin type or cooling teches.
    :param vul_id: int, reference number id of vulnerability factors.
    :param save_fig: boolean, whether to save figures in local disk or not.
    :param fig_suffix: string, suffix in figure name to differentiate figures.
    :param title_suffix: string, suffix in figure title, default value is ''
    :param fig_format: string, 'svg', 'png', figure format.
    :return: None
    """

    # Read and subset data
    df = pd.read_excel(report_excel_fp, engine='openpyxl', index_col=0)
    df = df[(df['Statistic Type'] == 'med') & (df['vul_id'] == vul_id) & (df['thd_id'] == thd_id)]
    df = df[df['Risk Type'] != 'flood']
    if thd_id == 19:
        df = df[df['Risk Type'] != 'water stress']
    # Feature engineer
    df_regroup = pd.read_excel(os.path.join(work_directory, 'tpp info', 'tpp_regroup.xlsx'),
                               engine='openpyxl')
    df = df.merge(df_regroup[['Group', 'Fuel-Turbine', 'Cooling']], on=['Group'], how='outer')

    if group == 'Fuel-Turbine':
        df = df[df['Fuel-Turbine'].isin(['CCGT', 'Coal-Steam'])]

    # Subset
    df_hist = df[df['scenario_id'] == 45][
        ['Power Plant Name', 'Group', 'Risk Type', 'base/ideal', 'Fuel-Turbine', 'Cooling']]
    df_hist.sort_values(by=['Fuel-Turbine', 'Group', 'Risk Type'], inplace=True)
    df_rcp45 = df[df['scenario_id'] == 45][
        ['Power Plant Name', 'Group', 'Risk Type', 'proj/ideal', 'Fuel-Turbine', 'Cooling']]
    df_rcp45.sort_values(by=['Fuel-Turbine', 'Group', 'Risk Type'], inplace=True)
    df_rcp85 = df[df['scenario_id'] == 85][
        ['Power Plant Name', 'Group', 'Risk Type', 'proj/ideal', 'Fuel-Turbine', 'Cooling']]
    df_rcp85.sort_values(by=['Fuel-Turbine', 'Group', 'Risk Type'], inplace=True)

    # Feature engineer
    df_hist_sum = df_hist.groupby('Group')['base/ideal'].sum().reset_index()
    df_hist_sum.columns = ['Group', 'Total Percentage']
    df_hist_in = df_hist_sum.merge(df_hist[['Group', 'Fuel-Turbine', 'Cooling']], on='Group', how='inner', copy=False)
    df_hist_in.drop_duplicates(subset=None, keep='first', inplace=True)
    # df_hist_in.to_csv(r'C:\Users\yan.cheng\PycharmProjects\EBRD\reports\assessment\boxplot_hist.csv', index=False)

    df_rcp45_sum = df_rcp45.groupby('Group')['proj/ideal'].sum().reset_index()
    df_rcp45_sum.columns = ['Group', 'Total Percentage']
    df_rcp45_in = df_rcp45_sum.merge(df_rcp45[['Group', 'Fuel-Turbine', 'Cooling']], on='Group', how='inner',
                                     copy=False)
    df_rcp45_in.drop_duplicates(subset=None, keep='first', inplace=True)
    # df_rcp45_in.to_csv(r'C:\Users\yan.cheng\PycharmProjects\EBRD\reports\assessment\boxplot_rcp45.csv', index=False)

    df_rcp85_sum = df_rcp85.groupby('Group')['proj/ideal'].sum().reset_index()
    df_rcp85_sum.columns = ['Group', 'Total Percentage']
    df_rcp85_in = df_rcp85_sum.merge(df_rcp85[['Group', 'Fuel-Turbine', 'Cooling']], on='Group', how='inner',
                                     copy=False)
    df_rcp85_in.drop_duplicates(subset=None, keep='first', inplace=True)

    # df_rcp85_in.to_csv(r'C:\Users\yan.cheng\PycharmProjects\EBRD\reports\assessment\boxplot_rcp85.csv', index=False)

    def extract_median(df, group):
        return df.groupby(group)['Total Percentage'].median().reset_index()

    print('Baseline:\n{}'.format(extract_median(df=df_hist_in, group=group)))
    print('RCP4.5:\n{}'.format(extract_median(df=df_rcp45_in, group=group)))
    print('RCP8.5:\n{}'.format(extract_median(df=df_rcp85_in, group=group)))

    def boxplot_stats(df, group):
        df_out = df.groupby([group])["Total Percentage"].describe(percentiles=[0.05, 0.50, 0.95])
        # df_out['iqr'] = df_out['75%'] - df_out['25%']
        # df_out['up'] = df_out['75%'] + 1.5 * df_out['iqr']
        # df_out['low'] = df_out['25%'] - 1.5 * df_out['iqr']
        return df_out[['50%', '5%', '95%']]

    print('Baseline:\n{}'.format(boxplot_stats(df=df_hist_in, group=group)))
    print('RCP4.5:\n{}'.format(boxplot_stats(df=df_rcp45_in, group=group)))
    print('RCP8.5:\n{}'.format(boxplot_stats(df=df_rcp85_in, group=group)))

    # Customize layout parameters
    PROPS = {
        'boxprops': {'facecolor': vc.WRI_COLOR_SCHEME['blue'], 'edgecolor': 'black', 'alpha': 1},
        'medianprops': {'linewidth': 1.5, 'color': 'red'},
        'whiskerprops': {'color': 'black'},
        'capprops': {'color': 'black'}
    }
    meanprops = {"marker": "+",
                 "markeredgecolor": "black",
                 "markersize": "5"}
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 10

    # Draw plots
    f, axes = plt.subplots(1, 3)
    bp_order = ['Once-through', 'Recirculating', 'Dry'] if group == 'Cooling' else None
    g1 = sns.boxplot(y="Total Percentage", x=group, data=df_hist_in, orient='v', ax=axes[0], showmeans=True,
                     meanprops=meanprops, showfliers=False, linewidth=1, order=bp_order, **PROPS)
    g2 = sns.boxplot(y="Total Percentage", x=group, data=df_rcp45_in, orient='v', ax=axes[1], showmeans=True,
                     meanprops=meanprops, showfliers=False, linewidth=1, order=bp_order, **PROPS)
    g3 = sns.boxplot(y="Total Percentage", x=group, data=df_rcp85_in, orient='v', ax=axes[2], showmeans=True,
                     meanprops=meanprops, showfliers=False, linewidth=1, order=bp_order, **PROPS)
    axes[1].legend([], [], frameon=False)
    axes[2].legend([], [], frameon=False)
    axes[0].set_title(f'Baseline{title_suffix}')
    axes[1].set_title(f'{year_id}\nRCP4.5{title_suffix}')
    axes[2].set_title(f'{year_id}\nRCP8.5{title_suffix}')

    if group == 'Cooling':
        if thd_id == 21:
            if year_id == 2030:
                axes[0].set_ylim([0, 0.14])
                axes[1].set_ylim([0, 0.14])
                axes[2].set_ylim([0, 0.14])
                for i in range(0, 3):
                    axes[i].set_xticklabels(['Once-\nthrough', 'Recirculating', 'Dry'])
            else:
                axes[0].set_ylim([0, 0.17])
                axes[1].set_ylim([0, 0.17])
                axes[2].set_ylim([0, 0.17])
                for i in range(0, 3):
                    axes[i].set_xticklabels(['Once-\nthrough', 'Recirculating', 'Dry'])
        else:
            if year_id == 2030:
                axes[0].set_ylim([0, 0.01])
                axes[1].set_ylim([0, 0.01])
                axes[2].set_ylim([0, 0.01])
                for i in range(0, 3):
                    axes[i].set_xticklabels(['Once-\nthrough', 'Recirculating', 'Dry'])
            else:
                axes[0].set_ylim([0, 0.02])
                axes[1].set_ylim([0, 0.02])
                axes[2].set_ylim([0, 0.02])
                for i in range(0, 3):
                    axes[i].set_xticklabels(['Once-\nthrough', 'Recirculating', 'Dry'])
    else:
        if thd_id == 21:
            if year_id == 2030:
                axes[0].set_ylim([0, 0.04])
                axes[1].set_ylim([0, 0.04])
                axes[2].set_ylim([0, 0.04])
            else:
                axes[0].set_ylim([0, 0.25])
                axes[1].set_ylim([0, 0.25])
                axes[2].set_ylim([0, 0.25])
        else:
            if year_id == 2030:
                axes[0].set_ylim([0, 0.012])
                axes[1].set_ylim([0, 0.012])
                axes[2].set_ylim([0, 0.012])
            else:
                axes[0].set_ylim([0, 0.02])
                axes[1].set_ylim([0, 0.02])
                axes[2].set_ylim([0, 0.02])

    axes[0].set_ylabel('Estimated plant-level percent annual generation losses')
    axes[0].set_yticklabels(['{:,.1%}'.format(x) for x in axes[0].get_yticks()])
    for i in range(0, 3):
        if group == 'Fuel-Turbine':
            axes[i].xaxis.set_tick_params(rotation=0)
        elif group == 'Cooling':
            axes[i].xaxis.set_tick_params(rotation=0)
        axes[i].set_xlabel('')
        axes[i].margins(x=2)
        if group == 'Fuel-Turbine':
            axes[i].set_xlim(-1, axes[i].get_xlim()[-1] + 0.5)
        else:
            axes[i].set_xlim(-0.5, axes[i].get_xlim()[-1])
    for i in range(1, 3):
        # axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_ylabel('')
    f.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_fig is True:
        fig_folder = os.path.join(work_directory, 'reports', 'assessment')
        fig_name = group.lower()
        plt.savefig(os.path.join(fig_folder, f'{fig_name}{fig_suffix}.{fig_format}'))
        print(f'Done with ploting boxplots. Find output here: {fig_folder}')
    else:
        plt.show()


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # User-defined parameters
    date = '20210519'
    year_id = 2030
    vul_id = 3
    thd_id_list = 19, 21
    group_list = ['Fuel-Turbine', 'Cooling']
    save_fig = True
    fig_format = 'svg'

    # For testing
    # date = '20210519'
    # year_id = {
    #     '20210519': 2030,
    #     '20210518': 2050,
    # }.get(date)

    # User-adjustable parameters
    fig_suffix_dict = {
        thd_id_list[1]: '_withRegLims',
        thd_id_list[0]: '_noRegLims',
    }
    report_fn = f'final-assessment-merge_{date}_ByPlant.xlsx'
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    report_excel_fp = os.path.join(os.path.join(work_directory, 'final assessment', 'processed'), report_fn)

    for thd_id, group in itertools.product(thd_id_list, group_list):
        fig_suffix = fig_suffix_dict.get(thd_id)
        title_suffix = '' if thd_id == 21 else ''
        main(work_directory=work_directory, report_excel_fp=report_excel_fp, year_id=year_id, vul_id=vul_id,
             thd_id=thd_id, group=group, save_fig=save_fig, fig_suffix=fig_suffix, title_suffix=title_suffix,
             fig_format=fig_format)
