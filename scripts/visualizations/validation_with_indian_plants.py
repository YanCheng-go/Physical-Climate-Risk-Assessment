"""Validation using daily generations of Indian plants"""

import os
import glob

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import datetime

import tqdm

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from scripts.visualizations import viz_configs as vc

pd.options.mode.chained_assignment = None  # default='warn'

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

plant_id = {
    'URAN CCPP': 9106,
    'VALUTHUR CCPP': 9107,
}


def prep_data(plant_name):
    def extract_cap():
        # Installed capacities
        return list(gen_df_all[(gen_df_all.PLANT == f'{plant_name}') & (gen_df_all.TYPE == 'cap_mw') & (
                gen_df_all.UNIT == 'All Units')].VALUE.unique())[0]

    def cal_at_p90():
        return np.percentile(at_df_all.data, q=90)

    def subset_df():
        # Subset
        inop_df_sub = inop_df_all[(inop_df_all.PLANT == f'{plant_name}') & (inop_df_all.DATE >= '2013-01-01') & (
                inop_df_all.DATE <= '2016-12-31')]
        gen_df_sub = gen_df_all[
            (gen_df_all.PLANT == f'{plant_name}') & (gen_df_all.UNIT == 'All Units') & (gen_df_all.TYPE == 'ag_gwh') &
            (gen_df_all.DATE >= '2013-01-01') & (gen_df_all.DATE <= '2016-12-31')]
        at_df_sub = at_df_all[(at_df_all.date <= '2016-12-31') & (at_df_all.date >= '2013-01-01')]
        return at_df_sub, inop_df_sub, gen_df_sub

    def op_dates():
        # Operation dates
        inop_dates = list(inop_df_sub.DATE.unique())
        zero_dates = list(gen_df_sub[pd.isnull(gen_df_sub.VALUE) | (gen_df_sub.VALUE == 0)].DATE.unique())
        drop_dates = list(set(inop_dates + zero_dates))
        op_dates = [i for i in gen_df_sub.DATE.unique() if i not in drop_dates]
        # len(op_dates)
        return op_dates

    # Read datasets
    # Power generation
    gen_df = pd.read_csv(os.path.join(work_directory, r'data\external\india_generation', 'dgr2.csv'), index_col=0)
    gen_df['Plant-Unit-Type'] = gen_df['Plant-Unit-Type'].str.replace('  ', ' ')
    gen_df['Plant-Unit-Type'] = gen_df['Plant-Unit-Type'].str.replace('MANERI BHALI - I HPS', 'MANERI BHALI _ I HPS')
    gen_df['Plant-Unit-Type'] = gen_df['Plant-Unit-Type'].str.replace('MANERI BHALI - II HPS', 'MANERI BHALI _ II HPS')
    gen_df['Plant-Unit-Type'] = gen_df['Plant-Unit-Type'].str.replace('RAMAGUNDEM - B TPS', 'RAMAGUNDEM _ B TPS')
    gen_df[['PLANT', 'UNIT', 'TYPE']] = gen_df['Plant-Unit-Type'].str.split(' - ', expand=True)[[0, 1, 2]]
    gen_df.PLANT = gen_df.PLANT.str.replace('MANERI BHALI _ I HPS', 'MANERI BHALI - I HPS')
    gen_df.PLANT = gen_df.PLANT.str.replace('MANERI BHALI _ II HPS', 'MANERI BHALI - II HPS')
    gen_df.PLANT = gen_df.PLANT.str.replace('RAMAGUNDEM _ B TPS', 'RAMAGUNDEM - B TPS')
    gen_df = pd.melt(gen_df, id_vars=['Plant-Unit-Type', 'PLANT', 'UNIT', 'TYPE'],
                     value_vars=[i for i in gen_df.columns if i not in ['Plant-Unit-Type', 'PLANT', 'UNIT', 'TYPE']],
                     var_name='DATE', value_name='VALUE')
    gen_df.DATE = pd.to_datetime(gen_df.DATE, format='%Y%m%d')
    gen_df['MONTH'] = gen_df.DATE.dt.month
    gen_df_all = gen_df.copy()
    gen_df_all.VALUE = gen_df_all.VALUE.replace(['*', 'L', 'S', 'P'], [0, 0, 0, 0])
    gen_df_all.VALUE = gen_df_all.VALUE.astype(float)

    # Outages
    inop_df = pd.read_csv(os.path.join(work_directory, r'data\external\india_generation', 'dgr10.csv'), index_col=0)
    inop_df.PLANT = inop_df.PLANT.str.replace('  ', ' ')
    inop_df.DATE = pd.to_datetime(inop_df.DATE, format='%Y%m%d')
    inop_df['MONTH'] = inop_df.DATE.dt.month
    inop_df_all = inop_df.copy()

    # Air temperatures from ERA5
    at_df = pd.read_csv(os.path.join(work_directory, r'india_data\era5_4_airtemp',
                                     f'PL_EBRD_TPP{plant_id.get(plant_name)}_historical_NAN_1980_2019.csv'),
                        index_col=0)
    at_df.date = pd.to_datetime(at_df.date, format='%Y-%m-%d')
    at_df = at_df[(at_df.indicator == 'tasavg')]
    at_df_all = at_df.copy()

    cap_mw_all = extract_cap()
    at_p90 = cal_at_p90()
    at_df_sub, inop_df_sub, gen_df_sub = subset_df()
    op_dates = op_dates()

    return cap_mw_all, at_p90, at_df_sub, inop_df_sub, gen_df_sub, op_dates


def dbscan_cluster(X, eps, min_samples):
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto').fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    # The Silhouette Coefficient is calculated using
    # the mean intra-cluster distance (a) and the mean nearest-cluster distance (b) for each sample.
    # A silhouette score ranges from -1 to 1,
    # with -1 being the worst score possible and 1 being the best score.
    # Silhouette scores of 0 suggest overlapping clusters.

    # Visualization
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6, label='')
        plt.legend()

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

    return db


def viz_regr(df, group_id, thd=None):
    fig, ax = plt.subplots()
    df[df.regroup == group_id].plot.scatter(x='AT', y='VALUE', ax=ax, s=10, c='blue')
    y_max, x_max = df[df.regroup == group_id].VALUE.max(), df[df.regroup == group_id].AT.max()
    y_min, x_min = df[df.regroup == group_id].VALUE.min(), df[df.regroup == group_id].AT.min()
    m, b = np.polyfit(df[df.regroup == group_id].AT, df[df.regroup == group_id].VALUE, 1)
    plt.plot(df[df.regroup == group_id].AT, m * (df[df.regroup == group_id].AT) + b)
    if thd is not None:
        plt.vlines(thd, y_min, y_max)
        ax.annotate('%.3f' % thd, xy=(thd, (y_max - y_min) * 0.5 + y_min))
    ax.annotate('y = %.3f * x + %.3f' % (m, b), xy=(x_max - 0.2 * (x_max - x_min), y_max))
    ax.set_ylabel('Actual daily generation (GWh)', color='blue')
    ax.set_xlabel('Air temperature (Kelvin)', color='red')
    plt.show()
    return m, b


def calculate_reduced_capacity(df, cap_mw_all, group_id_list, m_b_list, thd=None):
    def f(df, group_id, m, b, thd=None):
        df_in = df[df.regroup == group_id]
        df_in.loc[:, ['regr']] = df_in.AT * m + b
        n_days = len(df_in.AT)

        if thd is not None:
            max_gen = df_in[df_in.AT >= thd].regr.max()
            df_in.loc[:, ['gen_loss']] = max_gen - df_in.regr
            gen_loss_all = df_in[df_in.AT >= thd].gen_loss.sum()
        else:
            max_gen = df_in.regr.max()
            df_in.loc[:, ['gen_loss']] = max_gen - df_in.regr
            gen_loss_all = df_in.gen_loss.sum()

        max_gen_all = max_gen * n_days

        return max_gen_all, gen_loss_all, n_days

    out = []
    for group_id, (m, b) in list(zip(group_id_list, m_b_list)):
        out.append(f(df, group_id, m, b, thd))
    df = pd.DataFrame(out, columns=['mg', 'gl', 'd'])
    max_gen_all, gen_loss_all, n_days = df.apply(sum)

    return gen_loss_all * 1000 / (cap_mw_all * n_days * 24)


def efficiency_reduction(a, b, thd=None):
    return ((a * thd + b) - (a * (thd + 1) + b)) / (a * thd + b)


def viz_main(output_name=None, fig_extension='svg'):
    plant_name_list = ['URAN CCPP', 'VALUTHUR CCPP']
    colors = {-1: 'black', 1: 'green', 2: 'orange'}
    year_max = 2017

    eff_list = []
    sum_list = []
    df_list = []

    fig, axes = plt.subplots(nrows=1, ncols=2)
    for idx, plant_name in enumerate(plant_name_list):
        cap_mw_all, at_p90, at_df_sub, inop_df_sub, gen_df_sub, op_dates = prep_data(plant_name)

        #  All years, remove zero and inoperable days
        at_df_in = at_df_sub.copy()
        at_df_in.columns = ['DATE', 'AT', 'INDICATOR']
        df_merge = gen_df_sub[gen_df_sub.DATE.isin(op_dates)].merge(at_df_in[['DATE', 'AT']], on='DATE', how='inner')

        # Compute DBSCAN
        X = df_merge[df_merge.DATE.dt.year < year_max][['AT', 'VALUE']]
        X = StandardScaler().fit_transform(X)
        db = DBSCAN(eps=0.2, min_samples=6, algorithm='auto').fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        df = df_merge[df_merge.DATE.dt.year < year_max][['DATE', 'AT', 'VALUE']]
        df['AT'] = df['AT'] - 273.15
        df['label'] = db.labels_
        df['regroup'] = df['label'].replace(dict(zip(range(0, 5), [1, 2, -1, -1, -1])))

        df_exp = df.copy()
        df_exp.columns = ['Date', 'Air temperaure (degC)', 'Actual generation (GWh)', 'label', 'Cluster id']
        df_exp['Plant name'] = plant_name
        df_list.append(df_exp[['Plant name', 'Date', 'Air temperaure (degC)', 'Actual generation (GWh)', 'Cluster id']])

        # Points
        g1 = sns.scatterplot(data=df, x="AT", y="VALUE", hue='regroup', size='regroup', palette=list(colors.values()),
                             alpha=0.5, ax=axes[idx], marker='o', sizes=dict(zip([1, 2, -1], [25, 25, 8])))

        # Regression lines
        m_list = []
        b_list = []
        for group_id, col in zip([1, 2], [colors.get(1), colors.get(2)]):
            m, b = np.polyfit(df[df.regroup == group_id].AT, df[df.regroup == group_id].VALUE, 1)
            m_list.append(m)
            b_list.append(b)
            line, = axes[idx].plot(df[df.regroup == group_id].AT, m * df[df.regroup == group_id].AT + b, color=col,
                                   linewidth=2)

        reg_eqs = {1: '{} = {:.3f}{} + {:.2f}'.format('$y_{1}$', m_list[0], '$x_{1}$', b_list[0]),
                   2: '{} = {:.3f}{} + {:.2f}'.format('$y_{2}$', m_list[1], '$x_{2}$', b_list[1])}

        # Thresholds
        thd = at_p90 - 273.15
        if thd is not None:
            axes[idx].axvline(x=thd, color='gray', linestyle='--', label='')

        ref_ids = ['(a)', '(b)']
        axes[0].set_xlabel('Air temperature ($^\circ$C)')
        axes[1].set_xlabel('Air temperature ($^\circ$C)')
        axes[idx].text(0.05, 0.95, ref_ids[idx], ha='center', va='center', transform=axes[idx].transAxes,
                       fontdict=dict(fontsize=15, fontfamily='Arial'))

        # Legend
        custom_lines = [Line2D([0], [0], color=colors.get(1), lw=2),
                        Line2D([0], [0], color=colors.get(2), lw=2)]
        l1 = axes[idx].legend(custom_lines, [reg_eqs.get(1), reg_eqs.get(2)],
                              frameon=True, framealpha=1,
                              # borderpad=0.4,
                              edgecolor=(0, 0, 0, 0), facecolor="white")
        l1.get_frame().set_linewidth(0.0)

        axes[1].set_ylabel('')
        axes[0].set_ylabel('Actual daily generation (GWh)')
        axes[idx].spines['right'].set_visible(False)
        axes[idx].spines['top'].set_visible(False)
        axes[idx].yaxis.set_ticks_position('left')
        axes[idx].xaxis.set_ticks_position('bottom')

        # Statistics
        print(f'{plant_name}: ')
        for idx, (m, b) in enumerate(list(zip(m_list, b_list))):
            print(f'Linear regression model: y = {m}x + {b}')
            eff_ = efficiency_reduction(m, b, thd=thd)
            print('% Efficiency reduction: ', eff_)
            sum_ = calculate_reduced_capacity(df, cap_mw_all, [idx + 1], list(zip([m], [b])), thd=None)
            print('% Generation losses (all; class-level): ', sum_)
            sum_ = calculate_reduced_capacity(df, cap_mw_all, [idx + 1], list(zip([m], [b])), thd=thd)
            print('% Generation losses (>p90; class-level): ', sum_)
            eff_list.append(eff_)
            sum_list.append(sum_)

        sum_ = calculate_reduced_capacity(df, cap_mw_all, [1, 2], list(zip(m_list, b_list)), thd=None)
        print('% Generation losses (>p90; plant-level): ', sum_)

    custom_points = [Line2D([0], [0], color=colors.get(1), marker='o', linestyle=''),
                     Line2D([0], [0], color=colors.get(2), marker='o', linestyle='')]
    l1 = fig.legend(custom_points, ['Cluster 1', 'Cluster 2'], ncol=1, frameon=False,
                    # framealpha=1, edgecolor=(0, 0, 0, 0), facecolor="white", borderpad=2,
                    # handletextpad=0, handlelength=1, borderaxespad=1, borderpad=0.1, columnspacing=1,
                    handletextpad=0,
                    loc='lower right', bbox_to_anchor=(1, 0.17))
    l1.get_frame().set_linewidth(0.0)

    plt.tight_layout()
    if output_name is not None:
        output_fp = os.path.join(work_directory, 'reports', 'assessment', output_name + '.' + fig_extension)
        plt.savefig(output_fp, dpi=300)
        # Export figure source data
        src_data_fp = os.path.join(work_directory, 'reports', 'assessment', output_name + '.csv')
        pd.concat(df_list).to_csv(src_data_fp, index=False)
        print(f'Output saved here: {output_fp} and {src_data_fp}')
    else:
        plt.show()
    return eff_list, sum_list


def viz_comparison(base_, output_name=None, fig_extension='svg'):
    # Bar chart with uncertainties
    # Actual and model
    # All values are actual values
    eff_list, sum_list = viz_main(output_name='indian_gen-at_reg', fig_extension=fig_extension)
    # sum_list = [0.0005721924229584155, 0.000385293867257428, 0.00004929799740869499, 0.00023582818733163504]
    # eff_list = [0.009759285260488205, 0.013505995947599767, 0.0003957802486856967, 0.005670796928410134]
    eff_avg = np.mean(eff_list)
    sum_avg = np.mean(sum_list)
    eff_std = np.std(eff_list)
    sum_std = np.std(sum_list)
    base_avg = np.mean(base_)
    base_std = np.std(base_)

    # Portfolio-level results
    port_avg = [0.00351, 0.00496]
    port_loms = [0.00326, 0.00258]
    port_upms = [0.00406, 0.00734]
    port_x = [1, 2]
    port_lables = ['Model estimates', 'Observations']

    plant_avg = [base_avg, sum_avg]
    plant_loms = [base_avg - base_std, sum_avg - sum_std]
    plant_upms = [base_avg + base_std, sum_avg + sum_std]
    plant_x = [1, 2]
    plant_lables = ['Proposed framework', 'Indian case']

    labels = [port_lables, plant_lables]
    x_pos = [port_x, plant_x]
    yerr_loms = [port_loms, plant_loms]
    yerr_upms = [port_upms, plant_upms]
    y = [port_avg, plant_avg]
    colors = ['green', 'orange']

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
    for idx in [0, 1]:
        axes[idx].barh(x_pos[idx][0], y[idx][0], color=colors[0], label=labels[idx][0])
        axes[idx].barh(x_pos[idx][1], y[idx][1], color=colors[1], label=labels[idx][1])
        axes[idx].hlines(y=x_pos[idx], xmin=yerr_loms[idx], xmax=yerr_upms[idx],
                         colors=vc.WRI_COLOR_SCHEME['black'],
                         label='')
        for i in range(len(yerr_loms[idx])):
            axes[idx].plot([yerr_loms[idx][i], yerr_upms[idx][i]], [x_pos[idx][i], x_pos[idx][i]], "|", color='black')

        axes[idx].set_yticks([0, 1, 2, 3])
        axes[idx].set_yticklabels(['', '', '', ''])
        axes[idx].set_xticklabels(['{:,.2%}'.format(x) for x in axes[idx].get_xticks()])
        axes[idx].invert_yaxis()
        axes[idx].set_yticks([], minor=False)

    axes[0].set_ylabel('(a) Portfolio level', size=10)
    axes[1].set_ylabel('(b) Plant level', size=10)
    axes[1].set_xlabel('Share of physical climate hazard-induced generation losses', size=10)
    axes[0].legend(frameon=False)

    if output_name is not None:
        output_fp = os.path.join(work_directory, 'reports', 'assessment', output_name + '.' + fig_extension)
        plt.savefig(output_fp, dpi=300)
        print(f'Output saved here: {output_fp}')
    else:
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # viz_main(output_name=None)
    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    report_folder_name = 'reports'
    report_folder = os.path.join(work_directory, report_folder_name)
    if not os.path.exists(report_folder):
        os.mkdir(report_folder)
    assessment_report_folder = os.path.join(report_folder, 'assessment')
    if not os.path.exists(assessment_report_folder):
        os.mkdir(assessment_report_folder)

    try:
        fp = glob.glob(os.path.join(work_directory, 'final assessment', 'india', 'final-assessment*_ByPlant.xlsx'))[0]
        plant_name_list = ['URAN CCPP', 'VALUTHUR CCPP']
        df = pd.read_excel(fp, index_col=0, engine='openpyxl')
        base_ = [
            df[(df['Power Plant Name'] == plant_name_list[0]) & (df['Statistic Type'] == 'med')]['base/ideal'].values[
                0],
            df[(df['Power Plant Name'] == plant_name_list[1]) & (df['Statistic Type'] == 'med')]['base/ideal'].values[
                0]]
        # base_ = [0.0005150140213639, 0.0005905356211205]
        viz_comparison(base_=base_, output_name='model_vs_obs', fig_extension='png')
    except:
        fp = glob.glob(os.path.join(work_directory, 'final assessment', 'india', 'final-assessment*_ByPlant.xlsx'))[0]
        plant_name_list = ['URAN CCPP', 'VALUTHUR CCPP']
        df = pd.read_excel(fp, index_col=0, engine='openpyxl')
        base_ = [
            df[(df['Power Plant Name'] == plant_name_list[0]) & (df['Statistic Type'] == 'med')]['base/ideal'].values[
                0],
            df[(df['Power Plant Name'] == plant_name_list[1]) & (df['Statistic Type'] == 'med')]['base/ideal'].values[
                0]]
        # base_ = [0.0005150140213639, 0.0005905356211205]
        viz_comparison(base_=base_, output_name='model_vs_obs', fig_extension='png')
