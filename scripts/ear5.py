"""
Pre-process and analyse ERA5 datasets
"""

import ast
import itertools
import math
import os
import pickle
import sys
import warnings
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import psychrolib
from sklearn.model_selection import train_test_split
from sklearn import linear_model


dt = datetime.now()

warnings.simplefilter(action='ignore', category=FutureWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class Ear5:
    WORK_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    DATA_FOLDER = os.path.join(WORK_DIRECTORY, 'ear5')
    OUTPUT_DIRECTORY = os.path.join(WORK_DIRECTORY, 'output')
    ERA5_INDICATOR_LIST = ['v_component_of_wind_10m', 'mean_2m_air_temperature', 'minimum_2m_air_temperature',
                           'maximum_2m_air_temperature', 'dewpoint_2m_temperature', 'total_precipitation',
                           'surface_pressure', 'mean_sea_level_pressure', 'u_component_of_wind_10m']
    DEWPOINT_NAME = 'dewpoint_2m_temperature'
    AIRTEMP_NAME = 'mean_2m_air_temperature'
    HUMIDITY_NAME = 'relative_humidity'
    WBTEMP_NAME = 'wet_bulb_temperature'
    PR_NAME = 'total_precipitation'
    INDICATOR_NAME_LIST = [DEWPOINT_NAME, AIRTEMP_NAME, HUMIDITY_NAME, WBTEMP_NAME]
    PRESSURE = 101325
    GLOBAL_UNIT_SYSTEM = 'K'  # since the GEE climate datasets use Kalvin degrees, we use Kalvin degree as the unit system through this script

    def __init__(self, work_directory=WORK_DIRECTORY, data_folder=DATA_FOLDER, output_directory=OUTPUT_DIRECTORY,
                 dewpoint_name=DEWPOINT_NAME, airtemp_name=AIRTEMP_NAME, humidity_name=HUMIDITY_NAME,
                 wbtemp_name=WBTEMP_NAME, pressure=PRESSURE, era5_indicator_list=ERA5_INDICATOR_LIST, pr_name=PR_NAME):
        """
        :param work_directory: folder path-like string,
        :param data_folder: folder path-like string, folder of plant-level era5 datasets downloaded from GEE.
        :param output_directory: folder path-like string,
        :param dewpoint_name: string, variable name of dewpoint temperature in ERA5 database.
        :param airtemp_name: string, variable name of air temperature in ERA5 database.
        :param humidity_name: string, variable name of humidity in ERA5 database.
        :param wbtemp_name: string, variable name of wet-bulb temperature in ERA5 database.
        :param pressure: int, pressure constant used to calculate wet-bulb temperature.
        :param era5_indicator_list: list, a list of variables/indicators in ERA5 database.
        :param pr_name: string, variable name of precipitation in ERA5 database.
        """

        self.work_directory = work_directory
        self.data_folder = data_folder
        self.output_directory = output_directory

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        self.dewpoint_name = dewpoint_name
        self.airtemp_name = airtemp_name
        self.humidity_name = humidity_name
        self.wbtemp_name = wbtemp_name
        self.pr_name = pr_name

        self.pressure = pressure

        self.era5_indicator_list = era5_indicator_list

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

    @staticmethod
    def kwargs_generator(kwargs, kw, def_val, typ=None):
        if kw not in kwargs or kwargs[kw] is None:
            kwargs[kw] = def_val
        if typ == 'ls':
            if not isinstance(kwargs[kw], list):
                kwargs[kw] == [kwargs[kw]]
        return kwargs[kw]

    def switch_unit_system(self, x):
        return {'K': self.celsius_to_kelvin,
                'C': self.kelvin_to_celsius}.get(x, 'Invalid input!')

    @staticmethod
    def celsius_to_kelvin(x):
        return x + 273.15

    @staticmethod
    def kelvin_to_celsius(x):
        return x - 273.15

    @staticmethod
    def parse_filename(file_name):
        # Here, the file_name refers to conventional basename, which includes the file extension.
        str_list = os.path.splitext(file_name)[0].split('_')
        return {
            'plant_id': str_list[2].replace('TPP', ''),
            'start_year': str_list[4].split('-')[0],
            'end_year': str_list[4].split('-')[-1],
        }

    @staticmethod
    def func_day_month_list(x):
        df = pd.DataFrame(
            np.array(list(zip([i for i in x.keys()], [i for i in x.values()]))).reshape(12, 2),
            columns=['month', 'day'])
        month_list = [i for sub in [[m] * len(x[m]) for m in x.keys()] for i in sub]
        day_list = [i for sub in df.day.to_list() for i in sub]
        return day_list, month_list

    @staticmethod
    def args_to_list(x):
        """
        Convert one int or string value into a list.

        :param x: list, int, or string,
        :return: list,
        """

        if isinstance(x, str) or isinstance(x, int):
            out = [str(x)]
        else:
            out = x
        return out

    def restructure_gee_data(self, **kwargs):
        """
        Restructure (ERA5) datasets downloaded from GEE using associated scripts in /scripts/data/..

        :keyword file_path: file path-like string, file path of era5 datasets downloaded from GEE using associated scripts in /scripts/data/...
        :keyword indicator: list, int or string, ERA5 indicators or a subset.
        :keyword save_output: boolean, whether to save output in local disk.
        :keyword output_name: string, output file name.
        :keyword output_path: boolean, output file path.
        """

        file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        # clean air temp datasets
        df = pd.read_csv(kwargs['file_path'], index_col=0)

        df = df.sort_values(by=['indicatorName', 'year'])
        if 'indicator' in kwargs and kwargs['indicator'] is not None:
            kwargs['indicator'] = self.args_to_list(kwargs['indicator'])
            df = df.loc[df['indicatorName'].isin(kwargs['indicator']), df.columns[1:4]]
        else:
            df = df.loc[:, df.columns[1:4]]

        df['data'] = df['data'].apply(lambda row: ast.literal_eval(row))
        df['n_day'] = df['data'].apply(lambda row: len(row))

        month_list = {365: self.func_day_month_list(self.DDMM_DICT_365)[1], 366: self.func_day_month_list(self.DDMM_DICT_366)[1]}
        day_list = {365: self.func_day_month_list(self.DDMM_DICT_365)[0], 366: self.func_day_month_list(self.DDMM_DICT_366)[0]}
        df.loc[:, 'month'] = df['n_day'].map(month_list)
        df.loc[:, 'day'] = df['n_day'].map(day_list)

        df = df.set_index(['indicatorName', 'year', 'n_day']).apply(pd.Series.explode).reset_index()
        # print(df_airtemp)

        df['plant_id'] = self.parse_filename(os.path.basename(kwargs['file_path']))['plant_id']

        # # Rename indicators
        # old_names = list(set(df['indicatorName'].values))
        # old_max = [i for i in old_names if 'max' in i][0]
        # old_min = [i for i in old_names if 'min' in i][0]
        # indicator_names = {old_max: self.indicator_name_list[0], old_min: self.indicator_name_list[1]}
        # df['indicator'] = df['indicatorName'].map(indicator_names)

        df = df[['plant_id', 'year', 'month', 'day', 'indicatorName', 'data']]
        df.columns = ['plant_id', 'year', 'month', 'day', 'indicator', 'value']

        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name + '_restructure'
            else:
                output_name = 'ear5_restructure_output'

            if 'output_path' in kwargs and kwargs['output_path'] is not None:
                fp_out = kwargs['output_path']
                if os.path.splitext(fp_out)[-1] != '.csv':
                    fp_out = os.path.splitext(fp_out)[0] + '.csv'
            else:
                if not os.path.exists(self.output_directory):
                    os.mkdir(self.output_directory)
                fp_out = os.path.join(self.output_directory, output_name + '.csv')

            df.to_csv(fp_out)
            # print('Output is saved here: %s' % fp_out)

        return df

    def restructure_batch(self, **kwargs):
        """
        Restructure ERA5 datasets from GEE in batch mode.

        :keyword data_folder: folder path-like string, the folder path of ERA5 datasets.
        :keyword output_directory: folder path-like string, output folder.
        :keyword suffix: string, suffix in file name.
        :return: dictionary, df_dict
        """

        if 'data_folder' in kwargs:
            data_folder = kwargs['data_folder']
        else:
            data_folder = self.data_folder

        if 'output_directory' not in kwargs or kwargs['output_directory'] is None:
            output_directory = self.output_directory
        else:
            output_directory = kwargs['output_directory']

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if 'suffix' in kwargs and kwargs['suffix'] is not None:
            suffix = kwargs['suffix']
        else:
            suffix = 'restructure'

        def update_kwargs(kwargs, basename):
            kwargs.update({
                'file_path': os.path.join(data_folder, basename),
                'output_path': os.path.join(
                    output_directory,
                    os.path.splitext(basename)[0] + f'_{suffix}.csv')})

            out = {item: kwargs[item] for item in [i for i in kwargs.keys()
                                                   if not i in ['output_directory', 'suffix', 'data_folder']]}
            # print(out)
            return out

        df_list = list(map(lambda kwargs: self.restructure_gee_data(**kwargs),
                           tqdm([update_kwargs(kwargs, basename) for basename in os.listdir(data_folder) if
                                 not basename.endswith('.zip')])))

        df_dict = dict(zip([os.path.splitext(basename)[0] + f'_{suffix}.csv' for basename in os.listdir(data_folder) if
                            basename.endswith('.csv')], df_list))

        return df_dict

    def humidity_eq(self, td, t, unit_system=None):
        # from DEW point and air temp
        # https: // bmcnoldy.rsmas.miami.edu / Humidity.html
        # RH = 100 * (EXP((17.625 * TD) / (243.04 + TD)) / EXP((17.625 * T) / (243.04 + T)))
        # TD - Dewpoint, T - air temperature, RH - relative humidity
        if unit_system is None:
            unit_system = self.global_unit_system
        if unit_system == 'K':
            td = self.switch_unit_system('C')(td)
            t = self.switch_unit_system('C')(t)
            relative_humidity = 100 * (math.exp((17.625 * td) / (243.04 + td)) / math.exp((17.625 * t) / (243.04 + t)))
        elif unit_system == 'C':
            relative_humidity = 100 * (math.exp((17.625 * td) / (243.04 + td)) / math.exp((17.625 * t) / (243.04 + t)))
        return relative_humidity

    def cal_humidity(self, **kwargs):

        if 'df' in kwargs:
            df = kwargs['df']
        if 'file_path' in kwargs:
            df = pd.read_csv(kwargs['file_path'], index_col=0)
            file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        df = df.pivot(index=['plant_id', 'year', 'month', 'day'], columns='indicator', values='value')

        df[self.humidity_name] = df.apply(lambda row:
                                          self.humidity_eq(row[self.dewpoint_name], row[self.airtemp_name]), axis=1)

        df_out = pd.melt(df.reset_index(), id_vars=['plant_id', 'year', 'month', 'day'],
                         value_vars=[self.dewpoint_name, self.airtemp_name, self.humidity_name])

        if 'only_humidity' in kwargs and kwargs['only_humidity']:
            df_out = df_out[df_out['indicator'] == self.humidity_name]

        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name + '_withHumidity'
            else:
                output_name = 'ear5_calhumidity_output'

            if 'output_path' in kwargs and kwargs['output_path'] is not None:
                fp_out = kwargs['output_path']
                if os.path.splitext(fp_out)[-1] != '.csv':
                    fp_out = os.path.splitext(fp_out)[0] + '.csv'
            else:
                if not os.path.exists(self.output_directory):
                    os.mkdir(self.output_directory)
                fp_out = os.path.join(self.output_directory, output_name + '.csv')

            df_out.to_csv(fp_out)
            # print('Output is saved here: %s' % fp_out)

        return df_out

    def wbtemp_from_dewpoint(self, td, t):
        # Input of psychrolib functions has to be in Celsius degree
        psychrolib.SetUnitSystem(psychrolib.SI)
        if td > t:
            out = np.nan
        else:
            out = psychrolib.GetTWetBulbFromTDewPoint(self.switch_unit_system('C')(t),
                                                      self.switch_unit_system('C')(td),
                                                      self.pressure)
        return self.switch_unit_system('K')(out)

    def wbtemp_from_humidity(self, rh, t):
        # Use K
        psychrolib.SetUnitSystem(psychrolib.IP)
        return psychrolib.GetTWetBulbFromTDewPoint(t, rh, self.pressure)

    def cal_wbtemp(self, **kwargs):
        """
        Calculate wet-bulb temperature from given dry-bulb/air temperature, dew-point temperature, and pressure. To calculate from humidity and air temp using psychrolib (http://www.flycarpet.net/en/psyonline)

        :keyword df: pandas dataframe, restructured ERA5 datasets.
        :keyword file_path: file path-like string, restructured ERA5 datasets saved as csv files in local disk.
        :keyword only_wbtemp: boolean, keep only wet-bulb-related indicators in the final output.
        :keyword save_output: boolean, save output as csv files in local disk.
        :keyword output_name: string, file name.
        :keyword output_path: string, file path-like string,
        :return: pandas dataframe, dataframe with wet-bulb related indicators.
        """

        if 'df' in kwargs:
            df = kwargs['df']
        if 'file_path' in kwargs:
            df = pd.read_csv(kwargs['file_path'], index_col=0)
            file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]

        df = df.pivot(index=['plant_id', 'year', 'month', 'day'], columns='indicator', values='value')

        df[self.wbtemp_name] = df.apply(lambda row:
                                        self.wbtemp_from_dewpoint(row[self.dewpoint_name], row[self.airtemp_name]),
                                        axis=1)

        df_out = pd.melt(df.reset_index(), id_vars=['plant_id', 'year', 'month', 'day'],
                         value_vars=[i for i in self.era5_indicator_list] + [self.wbtemp_name])

        if 'only_wbtemp' in kwargs and kwargs['only_wbtemp']:
            df_out = df_out[df_out['indicator'] == self.wbtemp_name]

        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name
            else:
                output_name = 'ear5_calwebtemp_output'

            if 'output_path' in kwargs and kwargs['output_path'] is not None:
                fp_out = kwargs['output_path']
                if os.path.splitext(fp_out)[-1] != '.csv':
                    fp_out = os.path.splitext(fp_out)[0] + '.csv'
            else:
                if not os.path.exists(self.output_directory):
                    os.mkdir(self.output_directory)
                fp_out = os.path.join(self.output_directory, output_name + '.csv')

            df_out.to_csv(fp_out)
            # print('Output is saved here: %s' % fp_out)

        return df_out

    def cal_wbtemp_batch(self, **kwargs):
        """
        Calculate wet-bulb temperature in batch mode.

        :keyword df_batch: dictionary, keys -> file name, values -> a group of restructured ERA5 datasets in the form of pandas dataframe.
        :keyword data_folder: folder path-like string, the folder of restructured ERA5 datasets.
        :keyword output_directory: folder path-like string, output folder.
        :keyword suffix: string, suffix of output file name, the default value is '_withWetBulbTemp'.
        :return: dictionary, keys -> file name, values -> ERA5 with wet-bulb temperature.
        """

        if 'df_batch' in kwargs and kwargs['df_batch'] is not None:
            check_input = 0
            df_dict = kwargs['df_batch']
            basename_list = [i for i in kwargs['df_batch'].keys()]
        else:
            if 'data_folder' not in kwargs or kwargs['data_folder'] is None:
                data_folder = self.data_folder
            check_input = 1
            basename_list = [i for i in os.listdir(data_folder) if not i.endswith('zip')]

        if 'output_directory' not in kwargs or kwargs['output_directory'] is None:
            output_directory = self.output_directory
        else:
            output_directory = kwargs['output_directory']

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if 'suffix' in kwargs and kwargs['suffix'] is not None:
            suffix = kwargs['suffix']
        else:
            suffix = 'withWetBulbTemp'

        def update_kwargs(basename):
            if check_input == 0:
                input_param = {'df': df_dict[basename]}
            if check_input == 1:
                input_param = {'file_path': os.path.join(data_folder, basename)}

            fn = os.path.splitext(basename)[0]
            start_year_str, end_year_str = fn.split('_')[-3], fn.split('_')[-2]
            fn_prefix = fn.split(f'{start_year_str}_{end_year_str}')[0]
            fn_new = f'{fn_prefix}{start_year_str}-{end_year_str}_restructure_{suffix}.csv'
            kwargs.update({
                [i for i in input_param.keys()][0]: [i for i in input_param.values()][0],
                'output_path': os.path.join(output_directory, fn_new)
            })

            out = {item: kwargs[item] for item in
                   [i for i in kwargs.keys() if not i in ['df_batch', 'output_directory', 'suffix', 'data_folder']]}
            # print(out)
            return out

        df_list = list(map(lambda kwargs: self.cal_wbtemp(**kwargs),
                           tqdm([update_kwargs(basename) for basename in basename_list if
                                 not basename.endswith('.zip')])))

        print(f'Done calculating wet-bulb temperatures. Find output here: {output_directory}')

        df_dict = dict(zip(basename_list, df_list))

        return df_dict

    @staticmethod
    def normalize_feature(X, X_range):
        """
        Return normalized features (input variables) in X for further model fitting.

        :param X: list or np.array or pandas series,
        :param X_range: tuple, ((min1, min2), (max1, max2)).
        :return: np.array, normalized values.
        """

        if X_range is None:
            X_norm = X
        else:
            X_shape = np.array(X).shape
            if np.array(X_range).shape[0] != 2 or np.array(X_range).shape[1] != X_shape[1]:
                sys.exit('Invalid value of X_range in normalize_feature()!')
            X_min, X_max = X_range[0], X_range[1]
            X_min_reshape = np.repeat([X_min], [X_shape[0]], axis=0)
            X_max_reshape = np.repeat([X_max], [X_shape[0]], axis=0)
            X_norm = (X - X_min_reshape) / (X_max_reshape - X_min_reshape)
        return X_norm

    @staticmethod
    def switch_models(x, **kwargs):
        if x == 'ols':
            return linear_model.LinearRegression(**kwargs)
        elif x == 'lasso':
            return linear_model.Lasso(alpha=0.55, **kwargs)

    def multivariate_linear_regressor(self, X, y, test_size=0.2, random_state=0, model='ols', X_range=None):
        """
        Return the coefficients and accuracy metrics of a multivariate linear model. LASSO or ElasticNet Regression algorithm is used for parameter tuning considering the number of training samples.

        reference: https://satishgunjal.com/multivariate_lr_scikit/

        :param X: pandas series or np.array, X values.
        :param y: pandas series or np.array, y values.
        :param test_size: float, value between 0 and 1, percentage of number of values to be used for validation.
        :param random_state: int, randomization reference id.
        :param model: string, name of a regression model, optional ones as listed in switch_model().
        :param X_range: multi-dimension tuple, ((minimum x1, maximum x1), (minimum x2, maximum x2), ...)
        :return: regr, coef, intercept, accuracy
        """

        X = self.normalize_feature(X, X_range) if X_range is not None else X

        # Split samples into testing and training group
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=random_state)

        # Fit model
        regr = self.switch_models(x=model)
        regr.fit(X_train, y_train)
        coef = regr.coef_
        intercept = regr.intercept_

        # Evaluate model performance
        predictions = regr.predict(X_test)
        errors = abs(predictions - y_test)  # Absolute error
        mape = 100 * errors / y_test  # Mean absolute percentage error
        accuracy = round(100 - np.mean(mape), 4)  # Accuracy

        return regr, coef, intercept, accuracy

    def train_mlr(self, **kwargs):
        """
        Return trained models using specific regression algorithms.The models can be stored in pickle format, which enable to reuse the models easily.

        :keyword indicator: list, [y, X1, X2, X3,...], first item is the dependent variable, the rest are the independent variables.
        :keyword file_path: file path-like string, ERA5 datasets.
        :keyword df: pandas dataframe, ERA5 data in the form of pandas dataframe.
        :keyword output_code: string, optional, reference id of one run.
        :keyword X_range: multi-dimension tuple, ((minimum x1, maximum x1), (minimum x2, maximum x2), ...)
        :keyword model: string, name of a regression model, optional ones as listed in switch_model().
        :keyword test_size: float, value between 0 and 1, percentage of number of values to be used for validation.
        :keyword random_state: int, randomization reference id.
        :keyword save_output: boolean, whether to save the output as a pickle file in local disk or not.
        :keyword output_path: file path-like string,
        :keyword output_name: string, output file name.
        :return: regr, coef, intercept, accuracy, output_code
        """

        if 'indicator' not in kwargs or kwargs['indicator'] is None:
            # make sure the first item in the kwargs['indicator'] list is the dependent variable,
            # the rest are independent variables
            kwargs['indicator'] = ['y', 'X']

        kwargs['indicator'] = [kwargs['indicator']] if isinstance(kwargs['indicator'], str) else kwargs['indicator']

        if 'file_path' in kwargs and kwargs['file_path'] is not None:
            basename = os.path.basename(kwargs['file_path'])
            file_name = os.path.splitext(basename)[0]
            plant_id = self.parse_filename(basename)['plant_id']
            df = pd.read_csv(kwargs['file_path'], index_col=0)
        elif 'df' in kwargs and kwargs['df'] is not None:
            df = kwargs['df']
        else:
            sys.exit('No valid inputs!')

        if 'plant_id' in locals():
            output_code = plant_id
        elif 'output_code' in kwargs and kwargs['output_code'] is not None:
            output_code = kwargs['output_code']
        else:
            output_code = int(dt.microsecond)

        df_in = df.loc[df['indicator'].isin(kwargs['indicator'])]
        df_in = df_in.pivot(index=['plant_id', 'year', 'month', 'day'], columns='indicator', values='value')
        df_in = df_in.dropna()

        y = df_in[kwargs['indicator'][0]].values
        X = df_in[kwargs['indicator'][1:]].values

        regr, coef, intercept, accuracy = self.multivariate_linear_regressor(X=X, y=y, **{item: kwargs[item] for item in
                                                                                          kwargs.keys()
                                                                                          if
                                                                                          item in ['X_range', 'model',
                                                                                                   'test_size',
                                                                                                   'random_state']})

        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name + '_mlr'
            else:
                output_name = 'era5_' + '-'.join(kwargs['indicator']) + '_mlr'

            if 'output_path' in kwargs and kwargs['output_path'] is not None:
                fp_out = kwargs['output_path']
                if os.path.splitext(fp_out)[-1] != '.pkl':
                    fp_out = os.path.splitext(fp_out)[0] + '.pkl'
            else:
                if not os.path.exists(self.output_directory):
                    os.mkdir(self.output_directory)
                fp_out = os.path.join(self.output_directory, output_name + '.pkl')

            with open(fp_out, 'wb') as file:
                d = [regr, coef, intercept, accuracy, output_code]
                pickle.dump(d, file)
            # print(f'Output is saved here: {fp_out}')

        return regr, coef, intercept, accuracy, output_code

    def train_mlr_batch(self, **kwargs):
        """
        Train multi-linear regression model in batch mode.

        :keyword df_batch: dictionary, keys -> ERA5 file name, values -> pandas dataframe
        :keyword data_folder: folder path-like string, ERA5 file folder.
        :keyword output_directory: folder path-like string,
        :return: dictionary, keys -> ERA5 file name, values -> [regr, coef, intercept, accuracy, output_code]
        """

        if 'df_batch' in kwargs and kwargs['df_batch'] is not None:
            check_input = 0
            df_dict = kwargs['df_batch']
            basename_list = [i for i in kwargs['df_batch'].keys()]
        else:
            if 'data_folder' not in kwargs or kwargs['data_folder'] is None:
                data_folder = self.data_folder
            check_input = 1
            basename_list = [i for i in os.listdir(data_folder) if not i.endswith('zip')]

        if 'output_directory' not in kwargs or kwargs['output_directory'] is None:
            output_directory = self.output_directory
        else:
            output_directory = kwargs['output_directory']

        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if 'suffix' in kwargs and kwargs['suffix'] is not None:
            suffix = kwargs['suffix']
        else:
            suffix = 'mlr'

        def update_kwargs(basename):
            if check_input == 0:
                input_param = {'df': df_dict[basename]}
            if check_input == 1:
                input_param = {'file_path': os.path.join(data_folder, basename)}

            kwargs.update({
                [i for i in input_param.keys()][0]: [i for i in input_param.values()][0],
                'output_path': os.path.join(
                    output_directory,
                    os.path.splitext(basename)[0] + f'_{suffix}.csv')
            })

            out = {item: kwargs[item] for item in
                   [i for i in kwargs.keys() if i not in ['df_batch', 'output_directory', 'suffix', 'data_folder']]}
            # print(out)
            return out

        df_list = list(map(lambda kwargs: self.train_mlr(**kwargs),
                           tqdm([update_kwargs(basename) for basename in basename_list if
                                 not basename.endswith('.zip')])))
        print(f'Done model training. Find output here: {output_directory}')
        df_dict = dict(zip(basename_list, df_list))

        return df_dict

    def predict_wbtemp(self, **kwargs):
        """
        Model wet-bulb temperatures from air temperature and precipitation from GDDP datasets.

        :keyword model: list, [regr, coef, intercept, accuracy, output_code], output of train_mlr().
        :keyword model_path: file path-like string, pre-saved multi-linear models (pickle files).
        :keyword file_path_train: file path-like string, the file path of a set of training dataset as the input of train_mlr().
        :keyword df_train: pandas dataframe, a set of training dataset as the input of train_mlr().
        :keyword X_range: multi-dimension tuple, ((minimum x1, maximum x1), (minimum x2, maximum x2), ...)
        :keyword test_size: float, value between 0 and 1, percentage of number of values to be used for validation.
        :keyword random_state: int, randomization reference id.
        :keyword df: pandas dataframe, input dataset with all independent variables to predict wet-bulb temperature using pre-trained multi-linear regression models.
        :keyword file_path: file path-like string, bias corrected GDDP datasets with average air temperature and precipitation.
        :keyword indicator: list, [X1, X2, X3,...], a list of label of independent variables in input dataset.
        :keyword corr_coef:
        :keyword save_output: boolean, whether to save the output as a csv file in local disk or not.
        :keyword output_path: file path-like string,
        :keyword output_name: string, output file name.
        :return: pandas dataframe, df_out, input data in the form of pandas dataframe plus an additional column for the predicted wet-bulb temperature.
        """

        if 'indicator' not in kwargs or kwargs['indicator'] is None:
            kwargs['indicator'] = ['X']
        if isinstance(kwargs['indicator'], str):
            kwargs['indicator'] = [kwargs['indicator']]

        if 'model' in kwargs and kwargs['model'] is not None:
            regr, coef, intercept, accuracy, output_code = kwargs['model']
        elif 'model_path' in kwargs and kwargs['model_path'] is not None:
            with open(kwargs['model_path'], 'rb') as file:
                regr, coef, intercept, accuracy, output_code = pickle.load(file)
        else:
            if 'file_path_train' in kwargs and kwargs['file_path_train'] is not None:
                df_train = pd.read_csv(kwargs['file_path_train'], index_col=0)
            elif 'df_train' in kwargs and kwargs['df_train'] is not None:
                df_train = kwargs['df_train']
            else:
                sys.exit('No valid training data or model input!')
            regr, coef, intercept, accuracy, output_code = \
                self.train_mlr(df=df_train, indicator=[self.wbtemp_name] + kwargs['indicator'],
                               **{item: kwargs[item] for item in kwargs.keys() if item in
                                  ['X_range', 'model', 'test_size', 'random_state']})

        if 'file_path' in kwargs and kwargs['file_path'] is not None:
            file_name = os.path.splitext(os.path.basename(kwargs['file_path']))[0]
            df = pd.read_csv(kwargs['file_path'], index_col=0)
        elif 'df' in kwargs and kwargs['df'] is not None:
            df = kwargs['df']
        else:
            sys.exit('No valid data input!')

        # df_na = df[pd.isna(df['value'])]
        # df_na['wet_bulb_temperature_predicted'] = np.nan
        # df = df.dropna()

        ind_list = [i for i in df.columns if i not in ['indicator', 'value']]
        df_in = df.pivot(index=ind_list, columns='indicator', values='value')

        X = df_in[kwargs['indicator']].values
        X = self.normalize_feature(X, kwargs['X_range']) if 'X_range' in kwargs and kwargs['X_range'] is not None else X

        if 'corr_coef' in kwargs:
            X_shape = X.shape
            if np.array(kwargs['corr_coef']).shape[0] != X_shape[1]:
                sys.exit('Invalid value of corr_coef!')
            else:
                harm_coef = np.repeat([kwargs['corr_coef']], [X_shape[0]], axis=0)
                X = X * harm_coef

        y = regr.predict(X)

        df_out = df_in.copy()
        df_out['wet_bulb_temperature_predicted'] = y
        df_out = df_out.reset_index()

        value_columns = [col_name for col_name in df_out.columns
                         if sum([col_name.startswith(i)
                                 for i in ['airTempMax', 'airTempMin', 'airTempAvg', 'pr_nexGddp']]) == 1]

        df_out = pd.melt(df_out, id_vars=ind_list,
                         value_vars=value_columns + ['wet_bulb_temperature_predicted'],
                         var_name='indicator', value_name='value')

        # df_out = pd.concat([df_out, df_na])

        if 'save_output' in kwargs and kwargs['save_output']:
            if 'output_name' in kwargs and kwargs['output_name'] is not None:
                output_name = kwargs['output_name']
            elif 'file_name' in locals():
                output_name = file_name + '_predictedWetBulbTemp'
            else:
                output_name = 'era5_prediction_wetBulbTemp'

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

    def predict_wbtemp_batch(self, model_folder, data_folder, output_directory, plant_id_list=range(1, 26),
                             hist_timespan_str='1980-2005', futu_timespan_str='2030-2070', era5_timespan='1980-2019',
                             X_range=None):
        """
        Predict wet-bulb temperauture in batch mode.

        :param model_folder: folder path-like string, where the multi-linear models saved.
        :param data_folder: folder path-like string, input datasets, including all independent variables used to predict wet-bulb temperatures.
        :param output_directory: folder path-like string,
        :param plant_id_list: list, plant id (int)
        :param hist_timespan_str: string, historical time span in the form of '{start year}-{end year}'
        :param futu_timespan_str: string, future time span in the form of '{start year}-{end year}'
        :param era5_timespan: string, timespan of era5 datasets
        :param X_range: None or multi-dimension tuple, ((minimum x1, maximum x1), (minimum x2, maximum x2), ...)
        :return: dictionary, keys -> file name, values -> pandas dataframe, input datasets with an additional column of predicted wet-bulb temperatures.
        """
        cons = list(itertools.product([str(i) for i in plant_id_list], [hist_timespan_str, futu_timespan_str]))
        self.output_directory = output_directory

        def main(*args):
            plant_id, time_span = args
            model_path = os.path.join(model_folder, f'PL_EBRD_TPP{plant_id}_'
                                                    f'ERA5_{era5_timespan}_restructure_withWetBulbTemp_mlr.pkl')
            file_path = os.path.join(data_folder, f'PL_EBRD_TPP{plant_id}_GDDP_{time_span}'
                                                  f'_withAirTempAvg_biasCorrected.csv')
            return self.predict_wbtemp(model_path=model_path, file_path=file_path, X_range=X_range,
                                       indicator=['pr_nexGddp_biasCorrectedVsEra5', 'airTempAvg_biasCorrectedVsEra5'],
                                       save_output=True)

        df_list = list(map(lambda args: main(*args), tqdm([args for args in cons])))
        df_dict = dict(zip([args for args in cons], df_list))
        print(f'Done wet-bulb temperature modeling. Find output here: {output_directory}')
        return df_dict


if __name__ == '__main__':
    ear5 = Ear5(output_directory=r'C:\Users\yan.cheng\PycharmProjects\EBRD\output_temp')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Test restructure_batch() and cal_wbtemp_batch() (need to test)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ear5.data_folder = r'C:\Users\yan.cheng\PycharmProjects\EBRD\ear5'
    # df_batch = ear5.restructure_batch(save_output=True,
    #                                   output_directory=os.path.join(ear5.work_directory, 'ear5_restructure'))
    # df_batch_wbtemp = ear5.cal_wbtemp_batch(df_batch=df_batch,
    #                                         save_output=True,
    #                                         output_directory=os.path.join(ear5.work_directory, 'ear5_wetbulbtemp'))

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Test train_mlr() (need to test)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # wbtemp_foler = r'C:\Users\yan.cheng\PycharmProjects\EBRD\ear5_wetbulbtemp'
    # plant_id = '3'
    # X_range = ((0, 230), (0.127, 315))
    # ear5.output_directory =  r'C:\Users\yan.cheng\PycharmProjects\EBRD\wbtemp_model_nonorm_ols'
    # regr, coef, intercept, accuracy, output_code = ear5.train_mlr(
    #     file_path=os.path.join(wbtemp_foler, f'PL_EBRD_TPP{plant_id}_ERA5_1980-2019_restructure_withWetBulbTemp.csv'),
    #     indicator=[ear5.wbtemp_name, ear5.pr_name, ear5.airtemp_name],
    #     save_output=False,
    #     model='ols',
    #     X_range=None)
    #
    # print(regr, coef, intercept, accuracy, output_code)
    # # Read pkl files
    # with open(os.path.join(ear5.output_directory, 'PL_EBRD_TPP1_ERA5_1980-2019_restructure_withWetBulbTemp_mlr.pkl'), "rb") as input_file:
    #     d = pickle.load(input_file)
    # print(d)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Test train_mlr_bacth()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # wbtemp_model_folder_name = 'wbtemp_model_nonorm_ols'
    # tpp_wbtemp_era5_folder_name = 'ear5_wetbulbtemp'
    #
    # ear5.data_folder = os.path.join(ear5.work_directory, tpp_wbtemp_era5_folder_name)
    # ear5.output_directory = os.path.join(ear5.work_directory, wbtemp_model_folder_name)
    # ear5.train_mlr_batch(indicator=[ear5.wbtemp_name, ear5.pr_name, ear5.airtemp_name],
    #                      save_output=True, model='ols', X_range=None,
    #                      output_directory=ear5.output_directory)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Test predict_wbtemp() (need to test)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # model_folder = os.path.join(ear5.work_directory, 'wbtemp_model_nonorm_ols')
    # tpp_climate_folder = os.path.join(ear5.work_directory,
    #                                   'tpp_climate_gddp_restructure_all_withAirTempAvg_biasCorrected')
    # ear5.output_directory = os.path.join(ear5.work_directory, 'tpp_climate_gddp_all'
    #                                                           '_withWetBulbTemp_biasCorrected_nonorm_ols')
    #
    # cons = list(itertools.product([str(i) for i in range(1, 26)], ['1980-2005', '2010-2049']))
    #
    #
    # def main(*args):
    #     plant_id, time_span = args
    #     model_path = os.path.join(model_folder, f'PL_EBRD_TPP{plant_id}_'
    #                                             f'ERA5_1980-2019_restructure_withWetBulbTemp_mlr.pkl')
    #     file_path = os.path.join(tpp_climate_folder, f'PL_EBRD_TPP{plant_id}_GDDP_{time_span}'
    #                                                  f'_withAirTempAvg_biasCorrected.csv')
    #     return ear5.predict_wbtemp(model_path=model_path, file_path=file_path, X_range=None,
    #                                indicator=['pr_nexGddp_biasCorrectedVsEra5', 'airTempAvg_biasCorrectedVsEra5'],
    #                                save_output=True)
    #
    #
    # df_list = list(map(lambda args: main(*args), tqdm([args for args in cons])))
    # df_dict = dict(zip([args for args in cons], df_list))
    #
    # print(df_dict)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Test predict_wbtemp_bacth()
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    wbtemp_model_folder_name = 'wbtemp_model_nonorm_ols'
    tpp_airtemp_biasCorrected_folder_name = 'tpp_climate_gddp_restructure_all_withAirTempAvg_biasCorrected'
    tpp_gddp_wbtemp_folder_name = 'tpp_climate_gddp_all_withWetBulbTemp_biasCorrected_nonorm_ols'
    fut_eval_yrs = [2010, 2049]
    plant_id_list = range(1, 26)
    wbtemp_hist_timespan = (1980, 2005)  # time span concerned as baseline condition of wet-bulb temperatures from ERA5.
    era5_timespan = (1980, 2019)

    ear5.predict_wbtemp_batch(model_folder=os.path.join(ear5.work_directory, wbtemp_model_folder_name),
                              data_folder=os.path.join(ear5.work_directory, tpp_airtemp_biasCorrected_folder_name),
                              output_directory=os.path.join(ear5.work_directory, tpp_gddp_wbtemp_folder_name),
                              plant_id_list=plant_id_list,
                              hist_timespan_str='-'.join(map(str, wbtemp_hist_timespan)),
                              futu_timespan_str='-'.join(map(str, fut_eval_yrs)),
                              era5_timespan='-'.join(map(str, era5_timespan)))
