"""
Read historical and projected water temperature by coordinates from netCDF files.
"""

import numpy as np
import netCDF4
import pickle
import os
import datetime
import pandas as pd
from itertools import repeat
import glob

from scripts.data import data_configs as dc

PROJECT_DIRECTORY = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
WEEKLY_NAME_PREFIX = 'waterTemp_weekAvg_output'
CLIMATE_MODEL_LIST = dc.climate_model_list['WTUU']
CLIMATE_SCENARIO_LIST = dc.climate_scenario_list['WTUU']
START_YEAR_LIST = ['2006', '2020', '2030', '2040', '2050', '2060']
END_YEAR_LIST = ['2019', '2029', '2039', '2049', '2059', '2069']


class NcFileReader:
    """
    Read data from netCDF files.
    """
    # Default values
    WORK_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'tpp water temp')
    TIME_SPAN = None
    VAR = 'waterTemp'
    SAVE_PICKLE = False
    SAVE_CSV = False

    def __init__(self, lat, lon, file_path, work_directory=WORK_DIRECTORY, time_span=TIME_SPAN,
                 var=VAR, save_pickle=SAVE_PICKLE, save_csv=SAVE_CSV):
        """
        :param lat: float, latitude.
        :param lon: float longitude.
        :param work_directory: string, work directory path.
        :param file_path: string, netCDF file path.
        :param time_span: tuple or string, if it is tuple -> (start date, end date), string -> single date.
        :param var: string, name of variable.
        :param save_pickle: boolean, save error_idx_list as a pickle file or not.
        :param save_csv: boolean, save var_value_df into csv or not.
        :return:
        """

        self.lat = lat
        self.lon = lon
        self.work_directory = work_directory
        self.file_path = file_path
        self.time_span = time_span
        self.var = var
        self.save_pickle = save_pickle
        self.save_csv = save_csv

    def read_nc(self, file_path=None):
        """
        Read netCDF file using netCDF4.

        :param file_path: file path-like string
        :return: netCDF file
        """

        if file_path is None:
            file_path = self.file_path
        return netCDF4.Dataset(file_path, 'r')

    @staticmethod
    def get_basedate(nc, time_label):
        """
        Get the date of the first layer.

        :param nc: netCDF, netCDF file read using netCDF4.Dataset().
        :param time_label: str, the label of time dimension.
        :return: np.datetime64, the date of the first layer.
        """

        date_unit = nc.variables[time_label].units
        date_str = date_unit.split(' ')[2]
        y = date_str.split('-')[0]
        m = str(date_str.split('-')[1]).zfill(2)
        d = str(date_str.split('-')[2]).zfill(2)
        return np.datetime64('-'.join([y, m, d]))

    @staticmethod
    def get_real_date(base_date, time_value):
        """
        Convert values in the time dimension (i.e., time difference in days from '1901-01-01') to Gregorian date.

        :param base_date: np.datetime64, the date of the first layer in a netCDF file, which is assumed to be the starting date.
        :param time_value: float64, value in time dimension.
        :return: np.datetime64, Gregorian date of a specific layer.
        """

        return base_date + int(time_value)

    @staticmethod
    def get_var_label(nc, string):
        """
        Get the name/label of a variable/dimension.

        :param nc: netCDF, netCDF, netCDF file read using netCDF4.Dataset().
        :param string: str, the string that is included in the label, either one of these three, i.e., 'time', 'lat', 'lon'.
        :return: string, the name/label of specific variable/dimension.
        """

        return [l for l in nc.dimensions if string in l][0]

    @staticmethod
    def get_var_vals(nc, label):
        """
        Get all values of a dimension/variable.

        :param nc: netCDF, netCDF, netCDF file read using netCDF4.Dataset().
        :param label: string, the name/label of a dimension/variable.
        :return: list, all values in a dimension/variable.
        """

        return nc.variables[label][:]

    @staticmethod
    def get_nearest_locidx(lats, lons, lat, lon):
        """
        Get the index of nearest coordinates of a given set of lat and lon.

        :param lats: list, all latitude values.
        :param lons: list, all longitude values.
        :param lat: numeric, single latitude value.
        :param lon: numeric, single longitude value.
        :return: dictionary, the index of the nearest lat and lon in the lat list and lon list of a given pair of lat and lon.
        """

        return {'lat': np.abs(lats - lat).argmin(), 'lon': np.abs(lons - lon).argmin()}

    @staticmethod
    def construct_indexers(nc_dims, indexers=None, **kwargs):
        """
        Construct indexer to value(s) for a given coordinate.

        :param nc_dims: dimension output of a given netCDF file using nc.dimensions.
        :param indexers: list
        :param kwargs: dictinary, keys -> dimension/variable names, values -> int, indexes.
        :return: list, index in each dimension.
        """

        if indexers is None:
            indexers = [np.nan] * len(nc_dims)
        for key, value in kwargs.items():
            indexers[nc_dims.index(key)] = value
        return indexers

    @staticmethod
    def format_timespan(time_span):
        """
        Change the format of each value in time_span to np.datetime64.

        :param time_span: list, a pair of start and end date in the form of 'YYYY-MM-DD'.
        :return: time span in the form of np.datetime64.
        """

        if isinstance(time_span, str):
            timespan_formated = np.datetime64(time_span)
        if isinstance(time_span, tuple):
            timespan_formated = tuple((np.datetime64(time_span[0]), np.datetime64(time_span[1])))
        return timespan_formated

    @staticmethod
    def get_date_index(base_date, date):
        """
        Get the index of a layer.

        :param base_date: np.datetime64, date of the first layer.
        :param date: np.datetime64, date of the layer to be queried.
        :return: int, the index of the layer.
        """

        return (date - base_date).astype(int)

    @staticmethod
    def extract_single(nc, v, indexers):
        """
        Extract single value from a netCDF file.

        :param nc: netCDF
        :param v: string, label of a variable.
        :param indexers: list, a list of indexes of all dimensions.
        :return: one value at a given coordinates.
        """

        try:
            # How to make this dynamic???
            d1, d2, d3 = indexers
            d = nc.variables[v][d1, d2, d3]
        except:
            d = 'error'
        return d

    @staticmethod
    def output_fixchar(x):
        switch = {
            'variable_values': ['var_value_df', 'csv'],
            'invalid_layers': ['error_idx_list', 'pickle']
        }
        return switch[x]

    def name_output(self, prefix_label, input_path, work_directory=None):
        """
        Define the file name of an output.

        :param prefix_label: string,
        :param input_path: file path-like string
        :param work_directory: work directory-like string,
        :return: file path-like string, the file path of output
        """

        if work_directory is None:
            work_directory = self.work_directory
        fixchar = self.output_fixchar(prefix_label)
        nc_file_name = os.path.splitext(os.path.basename(input_path))[0]
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return os.path.join(work_directory, '{}_{}_{}.{}'.format(fixchar[0], nc_file_name, now_str, fixchar[1]))

    def extract_pixel_stack(self, work_directory=None, file_path=None, time_span=None, lat=None, lon=None,
                            var=None, save_pickle=None, pkl_path=None, save_csv=None, csv_path=None):
        """
        Extract values for one pixel from three dimensional netCDF files.

        :param work_directory: directory-like str, work directory.
        :param file_path: file path-like string, netCDF file path.
        :param time_span: tuple or string, a set of start_time and end_time with the date format as 'YYYY-MM-DD'.
        :param lat: numeric, latitude.
        :param lon: numeric, longitude.
        :param var: str, a sting that can be associated with the name/label of a variable, e.g., waterTemp.
        :param save_pickle: boolean, save error_idx_list into pickle file or not.
        :param pkl_path: file path-like str, the file path of a pickle file to save error_idx_list.
        :param save_csv: boolean, save var_value_df into csv file or not.
        :param csv_path: file path-like str, the file path of a csv file to save var_value_df.
        :return: dictionary, including the output of error_idx_list and var_value_df.
        """

        if work_directory is None:
            work_directory = self.work_directory
        if file_path is None:
            file_path = self.file_path
        if time_span is None:
            time_span = self.time_span
        if lat is None:
            lat = self.lat
        if lon is None:
            lon = self.lon
        if var is None:
            var = self.var
        if save_pickle is None:
            save_pickle = self.save_pickle
        if pkl_path is None:
            pkl_path = self.name_output('invalid_layers', file_path, work_directory)
        if save_csv is None:
            save_csv = self.save_csv
        if csv_path is None:
            csv_path = self.name_output('variable_values', file_path, work_directory)

        print('PROCEEDING: {} | lat={}, lon={}, var={}, time span={}'.format(os.path.basename(file_path), lat, lon, var, time_span))

        error_index_list = []
        var_value_list = []

        nc = self.read_nc(file_path)
        latLabel = self.get_var_label(nc, 'lat')
        lonLabel = self.get_var_label(nc, 'lon')
        timeLabel = self.get_var_label(nc, 'time')
        lats = self.get_var_vals(nc, latLabel)
        lons = self.get_var_vals(nc, lonLabel)
        times = self.get_var_vals(nc, timeLabel)
        ncvar = [v for v in nc.variables if var in v][0]

        i, j = self.get_nearest_locidx(lats, lons, lat, lon).values()

        nc_dims = nc.variables[ncvar].dimensions
        kwargs = dict(zip([latLabel, lonLabel], [i, j]))
        indexers = self.construct_indexers(nc_dims, **kwargs)

        if time_span is None:
            for time_index in range(len(times)):
                kwargs = dict(zip([timeLabel], [time_index]))
                indexers = self.construct_indexers(nc_dims, indexers, **kwargs)
                d = self.extract_single(nc, ncvar, indexers)
                if d == 'error':
                    error_index_list.append(time_index)
                    print(time_index)
                    var_value_list.append(np.nan)
                else:
                    var_value_list.append(d)
            var_value_df = pd.DataFrame({'date': [self.get_real_date(self.get_basedate(nc, timeLabel), i) for i in times],
                                         'value': var_value_list})
        else:
            timespan_formated = self.format_timespan(time_span)
            if isinstance(timespan_formated, tuple) is False:
                time_diff = self.get_date_index(self.get_basedate(nc, timeLabel), timespan_formated)
                time_index = np.where(np.array(times) >= time_diff, np.array(times), np.inf).argmin()
                kwargs = dict(zip([timeLabel], [time_index]))
                indexers = self.construct_indexers(nc_dims, indexers, **kwargs)
                d = self.extract_single(nc, ncvar, indexers)
                if d == 'error':
                    error_index_list.append(time_index)
                    print(time_index)
                    var_value_list.append(np.nan)
                else:
                    var_value_list.append(d)
                var_value_df = pd.DataFrame({'date': [self.get_real_date(self.get_basedate(nc, timeLabel), times[time_index])],
                                         'value': var_value_list})

            elif len(timespan_formated) == 2:
                start_diff = self.get_date_index(self.get_basedate(nc, timeLabel), timespan_formated[0])
                end_diff = self.get_date_index(self.get_basedate(nc, timeLabel), timespan_formated[1])
                start_time_index = np.where(times >= start_diff, times, np.inf).argmin()
                end_time_index = np.where(times <= end_diff, times, -np.inf).argmax()

                for time_index in range(start_time_index, end_time_index + 1):
                    kwargs = dict(zip([timeLabel], [time_index]))
                    indexers = self.construct_indexers(nc_dims, indexers, **kwargs)
                    d = self.extract_single(nc, ncvar, indexers)
                    if d == 'error':
                        error_index_list.append(time_index)
                        print(time_index)
                        var_value_list.append(np.nan)
                    else:
                        var_value_list.append(d)
                var_value_df = pd.DataFrame({'date': [self.get_real_date(self.get_basedate(nc, timeLabel), i)
                                                      for i in times[start_time_index:(end_time_index + 1)]],
                                             'value': var_value_list})
            else:
                print('time_span is invalid!')

        if save_pickle is True:
            with open(pkl_path, 'wb') as f:
                pickle.dump(error_index_list, f)
                print('pickle file saved here: {}'.format(pkl_path))
        if save_csv is True:
            var_value_df.to_csv(csv_path)
            print('csv file saved here: {}'.format(csv_path))

        return {'error_index_list': error_index_list, 'var_value_df': var_value_df}


class WeeklyNcFile:
    """
    Read weekly water temperatures (netCDF files).
    """

    def __init__(self, weekly_folder, weekly_name_prefix=WEEKLY_NAME_PREFIX, climate_model_list=CLIMATE_MODEL_LIST,
                 climate_scenario_list=CLIMATE_SCENARIO_LIST, start_year_list=START_YEAR_LIST,
                 end_year_list=END_YEAR_LIST):
        """
        :param weekly_folder: work directory-like string, the directory of the weekly water temperature datasets.
        :param weekly_name_prefix: string, prefix of weekly files.
        :param climate_model_list: list, exhaustive list of climate models.
        :param climate_scenario_list: list, exhaustive list of climate scenarios.
        :param start_year_list: list, list of start years indicated in file names.
        :param end_year_list: list, list of end years indicated in file names.
        """

        self.weekly_folder = weekly_folder
        self.weekly_name_prefix = weekly_name_prefix
        self.climate_model_list = climate_model_list
        self.climate_scenario_list = climate_scenario_list
        self.start_year_list = start_year_list
        self.end_year_list = end_year_list
        self.selector = {
            'climate_model': self.climate_model_list,
            'climate_scenario': self.climate_scenario_list,
                'start_year': self.start_year_list,
                'end_year': self.end_year_list
        }

    @staticmethod
    def get_year(x, idx):
        """
        Extract the information of start and end years from file names.

        :param x: string, file name with the format as 'waterTemp_weekAvg_output_[CLIMATE MODEL]_[CLIMATE SCENARIO]_[START YEAR]-01-07_[END YEAR]-12-30.nc'.
        :param idx: int, the index of start year in the file name.
        :return: list, a list of pairs of start and end year.
        """

        return [x.split('_')[idx].split('-')[0], x.split('_')[idx + 2].split('-')[0]]

    @staticmethod
    def split_file_name(file_name):
        """
        Split file name.

        :param file_name: string, file name.
        :return: dictionary,
        """

        file_split_list = file_name.split('_')
        keys = ['climate_model', 'climate_scenario', 'start_year', 'end_year']
        values = [file_split_list[3], file_split_list[4],
                  file_split_list[5].split('-')[0], file_split_list[7].split('-')[0]]
        return dict(zip(keys, values))

    @staticmethod
    def format_selector(selector):
        """
        Standardize the format of selector.

        :param selector: dictionary, keys -> selector name, values -> list of values.
        :return: dictionary, reformatted selector.
        """

        for k, v in selector.items():
            if type(v) is not list:
                selector[k] = [v]
        return selector

    def select_files(self, weekly_folder=None, selector=None):
        """
        Select/subset input files.

        :param weekly_folder: folder path-like str, folder path of the weekly water temperature database.
        :param selector: dictionary, filter.
        :return: list, a list of file names selected based on user-defined time span.
        """

        if weekly_folder is None:
            weekly_folder = self.weekly_folder
        if selector is None:
            selector = self.selector

        fp_list = []
        for root, dirs, files in os.walk(weekly_folder):
            if len(dirs) != 0:
                for j in dirs:
                    fp_list = [os.path.join(root, j, x) for x in files if x.endswith('.nc') and
                               all([self.split_file_name(x)[k] in v for k, v in
                                    self.format_selector(selector).items()])]
            else:
                fp_list = [os.path.join(root, x) for x in files if x.endswith('.nc') and
                           all([self.split_file_name(x)[k] in v for k, v in
                                self.format_selector(selector).items()])]

            return fp_list


class PlantLocation:
    """
    Read and subset plant geolocations.
    """
    DATA_DIRECTORY = os.path.join(PROJECT_DIRECTORY, 'tpp info')
    FILE_PATH = os.path.join(DATA_DIRECTORY, 'tpp_working.xlsx')
    SHEET_NAME = 'tpp_locs'
    WATER_LAT_NAME = 'Water Source Lat'
    WATER_LON_NAME = 'Water Source Lon'
    FIELD_NAME_DICT = {
        'lat': 'Lat',
        'lon': 'Lon',
        'plant_name': 'Power Plant Name',
        'country': 'Country',
        'unique_id': 'Unique ID',
        'water_powered': 'Source Water',
        'water_lat': 'Water Source Lat',
        'water_lon': 'Water Source Lon'
    }
    SELECTOR = {
        'lat': None,
        'lon': None,
        'plant_name': None,
        'country': None,
        'unique_id': None, # int
        'water_powered': None,
        'water_lat': None,
        'water_lon': None
    }

    def __init__(self, file_path=FILE_PATH, sheet_name=SHEET_NAME, water_lon_name=WATER_LON_NAME,
                 water_lat_name=WATER_LAT_NAME, field_name_dict=FIELD_NAME_DICT, selector=SELECTOR):
        """
        :param file_path: file path-like string, file path of tpp_working.xlsx.
        :param sheet_name: string, sheet name.
        :param water_lon_name: string, field name of water source longitude.
        :param water_lat_name: string, field name of water source latitude.
        :param field_name_dict: dictionary, fields and associated alias.
        :param selector: dictionary, data filter.
        """

        self.file_path = file_path
        self.sheet_name = sheet_name
        self.water_lon_name = water_lon_name
        self.water_lat_name = water_lat_name
        self.field_name_dict = field_name_dict
        self.selector = selector

    def get_field_name(self, x):
        return self.field_names[x]

    def open_dataset(self):
        # Retrieve file extension
        ext = os.path.splitext(self.file_path)[-1]
        # Read file using different methods according to the file extension
        if ext=='.xlsx' and self.sheet_name is not None:
            df = pd.read_excel(self.file_path, self.sheet_name)
        elif ext=='.xlsx' and self.sheet_name is None:
            df = pd.read_excel(self.file_path)
        elif ext=='.csv':
            df = pd.read_csv(self.file_path)

        return df

    def get_loc_df(self):
        """
        Retrieve information of water-related power plants, including the coordinates of water sources.

        :return: pandas.DataFrame, a dataframe including the information of water source locations, query water source lat and lon using corresponding field names
        """

        return self.open_dataset().dropna()

    def select_water_locs(self, **selector):
        """
        Select water source coordinates.

        :param selector: dictionary
        :return: pandas.DataFrame
        """

        df = self.get_loc_df()
        for key, value in selector.items():
            if value is not None:
                if type(value) is list:
                    df = df[df[self.field_name_dict[key]].isin(value)]
                else:
                    df = df[df[self.field_name_dict[key]] == value]
        return df


# extract values for selected locations
def main(file_path, time_span, loc_df, water_lat_name, water_lon_name, tpp_water_temp_folder, save_csv=True,
         save_pickle=False):
    """
    Batch run of time series extraction by coordinates from netCDF files.

    :param file_path: file-path like string, file path of the netCDF file.
    :param time_span: tuple, (start date, end date), e.g., ('1965-01-01', '2010-12-31')
    :param loc_df: pandas DataFrame, read coordinates as pandas dataframe.
    :param work_directory: directory-like path, work directory, where the plant-level water temperature to be saved.
    :return: pandas dataframe, data series retrieved from netCDF at multiple locations.
    """

    def f(lat, lon, coord_id):
        """
        Time series extraction by coordinates from netCDF files.

        :param lat: float, latitude of a plant geolocation.
        :param lon: float, longitude of a plant geolocation.
        :param coord_id: int, plant id.
        :return: same as the output of NcFileReader.extract_pixel_stack().
        """
        nc_reader = NcFileReader(lat=lat, lon=lon, file_path=file_path, work_directory=tpp_water_temp_folder,
                                 time_span=time_span,
                                 save_csv=save_csv, save_pickle=save_pickle)
        now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        if save_pickle is True:
            # pkl_name = 'error_idx_list_{}_{}_{}.pkl'.format(file_name, coord_id, now_str)
            pkl_name = 'error_idx_list_{}_{}.pkl'.format(file_name, coord_id)
            pkl_path = os.path.join(tpp_water_temp_folder, pkl_name)
        else:
            pkl_path = None
        if save_csv is True:
            csv_name = 'var_value_df_{}_{}_{}.csv'.format(file_name, coord_id, now_str)
            # csv_name = 'var_value_df_{}_{}.csv'.format(file_name, coord_id)
            csv_path = os.path.join(tpp_water_temp_folder, csv_name)
        else:
            csv_path = None
        out = nc_reader.extract_pixel_stack(save_pickle=save_pickle, save_csv=save_csv, csv_path=csv_path,
                                            pkl_path=pkl_path)
        return out

    return loc_df.apply(lambda row: f(row[water_lat_name], row[water_lon_name], row['Unique ID']),
                        axis=1)


def combineWeekly_by_time(tpp_water_temp_folder, start_year_list, end_year_list):
    """
    Post-processing of data series retrieved from multiple weekly netCDF files for the same geolocation, i.e., combining output along the time dimension and export as csv files.

    :param tpp_water_temp_folder: folder path-like string, plant-level weekly water temperatures retrieved from netCDF files.
    :param start_year_list: list, a list of starting year which was used to partition weekly water temperatures in the time dimension.
    :param end_year_list: list, a list of end year, together with start year, which was used to partition weekly water temperatures in the time dimension.
    :return: None
    """

    weekly_name = WEEKLY_NAME_PREFIX
    climate_model_list = CLIMATE_MODEL_LIST
    climate_scenario_list = CLIMATE_SCENARIO_LIST

    for ff in os.walk(tpp_water_temp_folder):
        loc_ids = sorted(list(set([i.split('_')[-2] for i in ff[2] if not i.endswith('.zip')])))
        for cm in climate_model_list:
            for cs in climate_scenario_list:
                for idx, item in enumerate(loc_ids):
                    name = '{}_{}_{}'.format(weekly_name, cm, cs)
                    names = [f for f in ff[2] if name in f]
                    fp_weekly = sorted([os.path.join(ff[0], n) for n in names if n.split('_')[-2] == item])
                    df_list = []
                    for f in fp_weekly:
                        df_list.append(pd.read_csv(f, index_col=0))
                    df_com = pd.concat(df_list, axis=0)
                    # print(df_com)
                    df_com.to_csv(os.path.join(ff[0], 'TPP_{}_{}_{}-{}.csv'.format(item, name, start_year_list[0],
                                                                                   end_year_list[-1])))


def change_histDaily_name(tpp_water_temp_folder, hist_timespan):
    """
    Rename daily water temperature datasets retrieved from netCDF files over the historical period.

    :param tpp_water_temp_folder: folder path-like string, plant-level weekly water temperatures retrieved from netCDF files.
    :param hist_timespan: string, time span of the historical period in the form of '{start year}-{end year}'
    :return: None
    """

    return [os.rename(os.path.join(tpp_water_temp_folder, i), os.path.join(tpp_water_temp_folder,
                                                                           'TPP_{}_waterTemperature_mergedV2_{}-{}.csv'.format(
                                                                               i.split('_')[-2],
                                                                               hist_timespan[0].split('-')[0],
                                                                               hist_timespan[1].split('-')[0]))) for
            i in os.listdir(tpp_water_temp_folder) if 'waterTemperature_mergedV2' in i]


def master(project_directory, daily_nc_file, weekly_folder, output_folder_name='tpp water temp', tpp_working_fp=None):
    """
    Master function to retrieve and restructure daily and weekly water temperature from netCDF datasets.

    :param project_directory: directory path-like str, EBRD project folder.
    :param daily_nc_file: file path-like str, daily water temperature datasets.
    :param weekly_folder: folder path-like str, weekly water temperature datasets.
    :param output_folder_name: str, output folder name
    :return: None
    """
    # Pre-defined parameters
    save_pickle = False  # Save error information in the form of pickle files
    save_csv = True  # Save water temperature time series for selected geolocations in the form of CSV files
    hist_timespan = ('1965-01-01', '2010-12-31')
    start_year_list = ['2006', '2020', '2030', '2040', '2050', '2060']
    end_year_list = ['2019', '2029', '2039', '2049', '2059', '2069']
    tpp_working_fp = os.path.join(project_directory, 'tpp info',
                                  'tpp_working.xlsx') if tpp_working_fp is None else tpp_working_fp

    tpp_water_temp_folder = os.path.join(project_directory,
                                         output_folder_name)  # where to save the extracted water temperature for each location
    if not os.path.exists(tpp_water_temp_folder):
        os.mkdir(tpp_water_temp_folder)

    # Select weekly nc files
    weekly_nc = WeeklyNcFile(weekly_folder=weekly_folder,
                             start_year_list=start_year_list,
                             end_year_list=end_year_list)
    weekly_files = weekly_nc.select_files()
    # Select water source locations -> this case selects all locations in the table
    plant_locs = PlantLocation(file_path=tpp_working_fp)
    loc_df = plant_locs.get_loc_df()
    water_lat_name, water_lon_name = plant_locs.water_lat_name, plant_locs.water_lon_name

    # Extract water temperature time series
    # for the projection period
    list(map(main, weekly_files, repeat(None), repeat(loc_df), repeat(water_lat_name), repeat(water_lon_name),
             repeat(tpp_water_temp_folder), repeat(save_csv), repeat(save_pickle)))
    combineWeekly_by_time(tpp_water_temp_folder, start_year_list, end_year_list)
    # for the baseline/historical period
    main(daily_nc_file, hist_timespan, loc_df, water_lat_name, water_lon_name, tpp_water_temp_folder, save_csv=True,
         save_pickle=False)
    change_histDaily_name(tpp_water_temp_folder, hist_timespan)

    # Delete interim output
    temp_fp_list = glob.glob(os.path.join(tpp_water_temp_folder, 'var_value_df_*.csv'))
    list([os.remove(i) for i in temp_fp_list])

    print(print(f'Done retrieving plant-level water temperatures. Find out put here: {tpp_water_temp_folder}'))


if __name__ == '__main__':
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # User defined parameters
    daily_nc_file = r'D:\WRI\temp\waterTemperature_mergedV2.nc'
    weekly_folder = r'D:\WRI\Water Temperature'
    output_folder_name = 'tpp water temp all'
    tpp_working_fp = None
    # None means default file path -> os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx')
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    master(project_directory=PROJECT_DIRECTORY, daily_nc_file=daily_nc_file, weekly_folder=weekly_folder,
           output_folder_name=output_folder_name, tpp_working_fp=tpp_working_fp)
