"""
Visualize portfolio-level result of hydro and thermal physical risks as a waterfall chart.

1. Organize final assessment reports for waterfall plots.
2. Use the merged final assessment report as the input of waterfall visualization.
"""

import pandas as pd
import os
import plotly.graph_objects as go
import numpy as np
from scripts.visualizations import viz_configs as vc
import plotly.io as pio
from plotly.subplots import make_subplots


def main(work_directory, in_fp, thd_id_reg, thd_id_noreg, fn_suffix='', stat='med', hydro_fp=None, save_fig=False):
    """
    Visualize portfolio-level result of hydro and thermal physical risks as a waterfall chart.

    :param work_directory: folder path-like string, work directory.
    :param in_fp: file path like string, input file path, where the output of report_states() saved.
    :param thd_id_reg: int, threshold reference id of the case with regulatory limits.
    :param thd_id_noreg: int, threshold reference id of the case without regulatory limits.
    :param fn_suffix: string, the suffix of a input file name.
    :param stat: string, reference id of statistic metric, 'med' -> median, 'q5' -> 5th percentile, 'q95' -> 95th percentile, 'avg' -> mean, 'min' -> minimum, 'max' -> maximum.
    :param hydro_fp: file path-like string, hydro plant assessment result.
    :param save_fig: boolean, whether to save the output waterfall figure.
    :return: None
    """

    # Read and subset data
    df = pd.read_excel(in_fp, index_col=0, engine='openpyxl')
    df = df.groupby(['thd_id', 'scenario_id', 'Statistic Type', 'Risk Type'])[
        ['Baseline Value', 'Projection Value']].apply(sum).reset_index()

    def prep_df(df, stat):
        """
        Prepare input dataframe for waterfall visualization.

        :param df: pandas dataframe, output of report_states() in the form of pandas dataframe.
        :param stat: stat: string, reference id of statistic metric, 'med' -> median, 'q5' -> 5th percentile, 'q95' -> 95th percentile, 'avg' -> mean, 'min' -> minimum, 'max' -> maximum.
        :return: pandas dataframe.
        """

        in_df = df[(((df['thd_id'] == thd_id_noreg) & (
            df['Risk Type'].isin(['water temperature', 'air temperature', 'drought', 'flood']))) | (
                            df['thd_id'] == thd_id_reg))]
        in_df.loc[in_df['Risk Type'] == 'water stress', ['Statistic Type']] = \
            in_df[in_df['Risk Type'] == 'water stress']['Statistic Type'].replace(['q5', 'q95'], ['q95', 'q5'])
        in_df = in_df[in_df['Statistic Type'] == stat]
        in_df = in_df.pivot(values=['Baseline Value', 'Projection Value'],
                            index=['thd_id', 'Statistic Type', 'Risk Type'],
                            columns=['scenario_id'])
        in_df.columns = ['Baseline Value', 'Baseline Value (duplicate)', 'Projection Value (RCP4.5)',
                         'Projection Value (RCP8.5)']
        in_df.reset_index(inplace=True)
        in_df = in_df.drop('Baseline Value (duplicate)', axis=1)

        # Feature engineering
        in_df['Difference (RCP4.5)'] = in_df['Projection Value (RCP4.5)'] - in_df['Baseline Value']
        in_df['Difference (RCP8.5)'] = in_df['Projection Value (RCP8.5)'] - in_df['Baseline Value']
        in_df['total_baseline'] = in_df.groupby('thd_id')['Baseline Value'].transform('sum')
        in_df['total_rcp45'] = in_df.groupby('thd_id')['Projection Value (RCP4.5)'].transform('sum')
        in_df['total_rcp85'] = in_df.groupby('thd_id')['Projection Value (RCP8.5)'].transform('sum')
        in_df['measure'] = 'relative'
        in_df = pd.melt(in_df, id_vars=['thd_id', 'Statistic Type', 'Risk Type', 'measure'],
                        value_vars=['Baseline Value', 'Projection Value (RCP4.5)', 'Projection Value (RCP8.5)',
                                    'Difference (RCP4.5)', 'Difference (RCP8.5)', 'total_baseline', 'total_rcp45',
                                    'total_rcp85'])
        # base_ws = in_df[(in_df['Risk Type'] == 'water stress') & (in_df['variable'] == 'Baseline Value')]
        # in_df.drop(base_ws.index, axis=0, inplace=True)

        return in_df

    in_df = prep_df(df=df, stat=stat)
    # Add extra fields for waterfall plots
    in_df['x2'] = in_df['Risk Type'].replace(
        ['water stress', 'air temperature', 'flood', 'drought', 'water temperature'], ['WS', 'AT', 'FL', 'DT', 'WT'])
    in_df['x1'] = in_df['variable'].replace(['Baseline Value', 'Projection Value (RCP4.5)', 'Projection Value (RCP8.5)',
                                             'Difference (RCP4.5)', 'Difference (RCP8.5)'],
                                            ['Baseline', 'RCP4.5', 'RCP8.5', 'RCP4.5', 'RCP8.5'])

    # Uncertainty
    in_df_q5 = prep_df(df=df, stat='q5')
    in_df_q95 = prep_df(df=df, stat='q95')
    total_q5 = in_df_q5[in_df_q5['variable'].isin(['total_baseline', 'total_rcp45', 'total_rcp85'])][
        ['thd_id', 'Statistic Type', 'variable', 'value']].drop_duplicates()
    total_q95 = in_df_q95[in_df_q95['variable'].isin(['total_baseline', 'total_rcp45', 'total_rcp85'])][
        ['thd_id', 'Statistic Type', 'variable', 'value']].drop_duplicates()
    total_uncertainty = pd.concat([total_q5, total_q95])

    # Visualization
    for thd_id in [i for i in [thd_id_noreg, thd_id_reg] if i is not None]:
        total_proj_rcp45 = in_df[(in_df['thd_id'] == thd_id) & (in_df['x1'] == 'total_rcp45')]['value'].values[0]
        total_proj_rcp85 = in_df[(in_df['thd_id'] == thd_id) & (in_df['x1'] == 'total_rcp85')]['value'].values[0]
        viz_df = in_df[(in_df['x1'] != 'total_baseline') & (in_df['thd_id'] == thd_id) &
                       (in_df['variable'].isin(['Baseline Value', 'Difference (RCP4.5)', 'Difference (RCP8.5)']))]
        if hydro_fp is not None:
            viz_df.reset_index(inplace=True)
            hydro_df = pd.read_excel(hydro_fp, engine='openpyxl')
            hydro_df.columns = ['Percentile', 'Difference (RCP4.5)', 'Difference (RCP8.5)']
            hydro_df['Difference (RCP4.5)'] = hydro_df['Difference (RCP4.5)'] * (-1)
            hydro_df['Difference (RCP8.5)'] = hydro_df['Difference (RCP8.5)'] * (-1)
            hydro_df['Percentile'] = hydro_df['Percentile'].replace([50, 5, 95], ['med', 'q5', 'q95'])
            viz_df = viz_df.append(
                pd.DataFrame([[99, thd_id, stat, 'hydro', 'relative', 'Baseline Value', 0, 'HD', 'Baseline'],
                              [100, thd_id, stat, 'hydro', 'relative', 'Difference (RCP4.5)',
                               hydro_df[hydro_df['Percentile'] == stat]['Difference (RCP4.5)'].values[0], 'HD',
                               'RCP4.5'],
                              [101, thd_id, stat, 'hydro', 'relative', 'Difference (RCP8.5)',
                               hydro_df[hydro_df['Percentile'] == stat]['Difference (RCP8.5)'].values[0], 'HD',
                               'RCP8.5']],
                             columns=viz_df.columns))
            viz_df.sort_values(by=['variable', 'index'], inplace=True)
            viz_df.drop('index', axis=1, inplace=True)
            viz_df.reset_index(drop=True)
            total_proj_rcp85 = total_proj_rcp85 + \
                               hydro_df[hydro_df['Percentile'] == stat]['Difference (RCP8.5)'].values[0]
            total_proj_rcp45 = total_proj_rcp45 + \
                               hydro_df[hydro_df['Percentile'] == stat]['Difference (RCP4.5)'].values[0]
        viz_df = viz_df.append(pd.DataFrame([[thd_id, stat, None, 'total', 'Baseline Value', None, 'Total', 'Baseline'],
                                             [thd_id, stat, None, 'total', 'Difference (RCP4.5)', None, 'Total',
                                              'RCP4.5'],
                                             [thd_id, stat, None, 'total', 'Difference (RCP8.5)', None, 'Total',
                                              'RCP8.5']],
                                            columns=in_df.columns))
        viz_df.sort_values(by=['variable', 'measure'], inplace=True)
        viz_df.reset_index(drop=True, inplace=True)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # RCP4.5
        in_df_rcp45 = viz_df[viz_df['x1'].isin(['RCP4.5', 'Baseline'])]
        base_idx = [i for i in in_df_rcp45[
            (in_df_rcp45['x1'] == 'Baseline') & (in_df_rcp45['measure'] == 'relative')].index.values.astype(int)]
        y = [i for i in in_df_rcp45['value'].values]
        x = [[i for i in in_df_rcp45['x1']], [i for i in in_df_rcp45['x2']]]
        y = [int(i / 1000 + (0.5 if i >= 0 else -0.5)) if not pd.isna(i) else i for i in y]
        measure = in_df_rcp45['measure']
        text = [''] * len(y)
        for idx, val in enumerate(y):
            if not pd.isna(val):
                text[idx] = '{:,}'.format(y[idx])
                if not idx in base_idx and val > 0:
                    text[idx] = '+{}'.format(text[idx])
        fig.add_trace(go.Waterfall(
            x=x, measure=measure, y=y,
            decreasing={"marker": {"color": vc.WRI_COLOR_SCHEME['gray'], "line": {"color": "gray", "width": 0}}},
            increasing={"marker": {"color": vc.WRI_COLOR_SCHEME['gray']}},
            totals={"marker": {"color": vc.WRI_COLOR_SCHEME['blue'],
                               "line": {"color": vc.WRI_COLOR_SCHEME['blue'], "width": 0}}},
            textposition="outside",
            text=text,
            connector={'line': {'width': 1}},
        ))

        # RCP8.5
        in_df_rcp85 = viz_df[viz_df['x1'].isin(['RCP8.5', 'Baseline'])]
        y = [i for i in in_df_rcp85['value'].values]
        x = [[i for i in in_df_rcp85['x1']], [i for i in in_df_rcp85['x2']]]
        y = [int(i / 1000 + (0.5 if i >= 0 else -0.5)) if not pd.isna(i) else i for i in y]
        measure = in_df_rcp85['measure']
        text = [''] * len(y)
        for idx, val in enumerate(y):
            if not pd.isna(val):
                text[idx] = '{:,}'.format(y[idx])
                if not idx in base_idx and val > 0:
                    text[idx] = '+{}'.format(text[idx])
        fig.add_trace(go.Waterfall(
            x=x, measure=measure, y=y,
            decreasing={"marker": {"color": vc.WRI_COLOR_SCHEME['gray'],
                                   "line": {"color": vc.WRI_COLOR_SCHEME['gray'], "width": 0}}},
            increasing={"marker": {"color": vc.WRI_COLOR_SCHEME['gray']}},
            totals={"marker": {"color": vc.WRI_COLOR_SCHEME['blue'],
                               "line": {"color": vc.WRI_COLOR_SCHEME['blue'], "width": 0}}},
            textposition="outside",
            text=text,
            offset=[],
            connector={'line': {'width': 1}},
        ))

        base_round_list = [int(i / 1000 + (0.5 if i >= 0 else -0.5)) if not pd.isna(i) else i
                           for i in [i for i in in_df_rcp45[in_df_rcp45['x1'] == 'Baseline']['value'].values]]
        base_round_sum = int(np.sum([base_round_list[i] for i in base_idx]))
        base = base_round_sum
        total_proj_rcp45 = int(total_proj_rcp45 / 1000 + 0.5)
        total_proj_rcp85 = int(total_proj_rcp85 / 1000 + 0.5)
        # Baseline bars
        fig.add_trace(
            go.Bar(x=[['Baseline', 'RCP4.5', 'RCP8.5'], ['Total', 'Total', 'Total']], y=[base, base, base], width=0.7,
                   marker={"color": vc.WRI_COLOR_SCHEME['blue-dark'],
                           "line": {"color": vc.WRI_COLOR_SCHEME['blue-dark'], "width": 0}}))

        # Extra annotations/labels
        def reform_text(val, plus=True):
            if not pd.isna(val):
                a = '{:,}'.format(val)
                if plus is True:
                    a = '+{}'.format(a) if val > 0 else a
            return a

        if thd_id == 19:
            for i in np.arange(11, 18, 6):
                fig.add_annotation(x=i, y=base / 2, text=reform_text(base, plus=False), showarrow=False,
                                   font={'color': 'white'}, yshift=0)
            for x, y, text in list(zip(np.arange(11, 18, 6),
                                       [(total_proj_rcp45 - base) / 2 + base, (total_proj_rcp85 - base) / 2 + base],
                                       [total_proj_rcp45 - base, total_proj_rcp85 - base])):
                fig.add_annotation(x=x, y=y, text=reform_text(text), showarrow=False, font={'color': 'white'}, yshift=0)
            for x, y, text in list(zip(np.arange(5, 18, 6), [base, total_proj_rcp45, total_proj_rcp85],
                                       [reform_text(base, plus=False), reform_text(total_proj_rcp45, plus=False),
                                        reform_text(total_proj_rcp85, plus=False)])):
                fig.add_annotation(x=x, y=y, text=text, showarrow=False, yshift=20)
        elif thd_id == 21:
            for i in np.arange(13, 22, 7):
                fig.add_annotation(x=i, y=base / 2, text=reform_text(base, plus=False), font={'color': 'white'},
                                   showarrow=False, yshift=0)
            for x, y, text in list(zip(np.arange(13, 22, 7),
                                       [(total_proj_rcp45 - base) / 2 + base, (total_proj_rcp85 - base) / 2 + base],
                                       [total_proj_rcp45 - base, total_proj_rcp85 - base])):
                fig.add_annotation(x=x, y=y, text=reform_text(text), font={'color': 'white'}, showarrow=False, yshift=0)
            for x, y, text in list(zip(np.arange(6, 22, 7), [base, total_proj_rcp45, total_proj_rcp85],
                                       [reform_text(base, plus=False), reform_text(total_proj_rcp45, plus=False),
                                        reform_text(total_proj_rcp85, plus=False)])):
                fig.add_annotation(x=x, y=y, text=text, showarrow=False, yshift=20)

            if not fn_suffix == '_wet-to-air':
                fig.add_trace(
                    go.Bar(x=[['Baseline', 'RCP4.5', 'RCP8.5'], ['WT', 'WT', 'WT']], y=[0.1, 0.1, 0.1],
                           base=[36 + 76 + 40 + 90, 752 + 90 + 85 - 15 + 3796 + 292, 752 + 103 + 133 + 3796 + 360],
                           width=0.7, marker={"color": vc.WRI_COLOR_SCHEME['black'],
                                              "line": {"color": vc.WRI_COLOR_SCHEME['black'], "width": 1}}))

                for x, y, text in list(
                        zip(np.arange(4, 22, 7), [36 + 76 + 40 + 92 / 2, 90 + 85 - 15 + 3796 + 752 + 292 / 2,
                                                  752 + 103 + 133 + 3796 + 360 / 2],
                            ['92', '+292', '+360'])):
                    fig.add_annotation(x=x, y=y, text=text, showarrow=False, yshift=0)
                for x, y, text in list(
                        zip(np.arange(4, 22, 7),
                            [36 + 76 + 40 + 92 + 508 / 2, 90 + 85 - 15 + 3796 + 752 + 292 + 1771 / 2,
                             752 + 103 + 133 + 3796 + 360 + 2170 / 2],
                            ['508', '+1,771', '+2,170'])):
                    fig.add_annotation(x=x, y=y, text=text, showarrow=False, yshift=0)

        # Add uncertainty bars
        def query_uncertainty(q, scenario):
            variable = {
                'baseline': 'total_baseline',
                'rcp45': 'total_rcp45',
                'rcp85': 'total_rcp85',
            }.get(scenario)
            stat = {
                5: 'q5',
                95: 'q95'
            }.get(q)
            hydro_variable = {
                'rcp45': 'Difference (RCP4.5)',
                'rcp85': 'Difference (RCP8.5)'
            }.get(scenario)
            hydro_stat = {
                5: 'q95',
                95: 'q5'
            }.get(q)
            if hydro_fp is not None and scenario != 'baseline':
                return int((total_uncertainty[
                                (total_uncertainty['thd_id'] == thd_id) & (total_uncertainty['Statistic Type'] == stat)
                                & (total_uncertainty['variable'] == variable)]['value'].values[0]
                            + hydro_df[hydro_df['Percentile'] == hydro_stat][hydro_variable].values[0]) / 1000 + 0.5)
            else:
                return int(total_uncertainty[
                               (total_uncertainty['thd_id'] == thd_id) & (total_uncertainty['Statistic Type'] == stat)
                               & (total_uncertainty['variable'] == variable)]['value'].values[0] / 1000 + 0.5)

        fig.add_trace(go.Scatter(
            x=[['Baseline', 'RCP4.5', 'RCP8.5'], ['Total', 'Total', 'Total']],
            y=[base, total_proj_rcp45, total_proj_rcp85],
            mode='markers',
            marker=dict(size=1, color=vc.WRI_COLOR_SCHEME['blue']),
            error_y=dict(
                type='data',
                symmetric=False,
                array=[query_uncertainty(95, 'baseline') - base, query_uncertainty(95, 'rcp45') - total_proj_rcp45,
                       query_uncertainty(95, 'rcp85') - total_proj_rcp85],
                arrayminus=[base - query_uncertainty(5, 'baseline'), total_proj_rcp45 - query_uncertainty(5, 'rcp45'),
                            total_proj_rcp85 - query_uncertainty(5, 'rcp85')],
                color=vc.WRI_COLOR_SCHEME['black'],
                thickness=1,
                width=4,
            )
        ))

        fig.update_yaxes(title='Generation losses (10 GWh)', visible=False)
        range = [-1, 21] if thd_id == 21 else [-1, 18]
        fig.update_xaxes(linewidth=2, range=range, showticklabels=False, nticks=0)
        if not fn_suffix == '_wet-to-air':
            if thd_id == 21:
                fig.update_yaxes(range=[0, 12000])
        fig.update_layout(title="", waterfallgap=0.3, template='simple_white', showlegend=False, font_family='Arial',
                          font_size=8)
        out_fp = os.path.join(work_directory, 'reports', 'assessment', f'waterfall_thd{thd_id}{fn_suffix}.svg')
        # fig.write_image(out_fp)
        if save_fig is True:
            pio.write_image(fig, out_fp, format='svg', scale=300, width=1000, height=400)
            print(f'Find output here: {out_fp}')
        else:
            fig.show()


if __name__ == '__main__':
    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    data_directory = os.path.join(work_directory, 'final assessment', 'processed')
    hydro_fp = os.path.join(work_directory, 'final_assessment', 'hydro_result',
                            'portfolio_level_hydro_2030_gen_losses.xlsx')

    # Default
    in_fp, fn_suffix, thd_id_noreg = os.path.join(data_directory,
                                                  'final-assessment-merge_20210519_ByPlant.xlsx'), '', 19
    main(work_directory=work_directory, in_fp=in_fp, thd_id_noreg=thd_id_noreg, thd_id_reg=21, fn_suffix=fn_suffix,
         hydro_fp=hydro_fp, save_fig=False)

    # # Wet-to-cool
    # in_fp, fn_suffix, thd_id_noreg = os.path.join(data_directory,
    #                                               'final-assessment-merge_20210519_wet-to-air_ByPlant.xlsx'), '_wet-to-air', None
    # main(work_directory=work_directory, in_fp=in_fp, thd_id_noreg=thd_id_noreg, thd_id_reg=21, fn_suffix=fn_suffix,
    #      hydro_fp=hydro_fp, save_fig=False)
