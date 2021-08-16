'''
Universal functions / settings / information
'''

import ee
import os

import pandas as pd


def authentic_ee():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()
    return 0


def kwargs_generator(kwargs, kw, def_val, typ=None):
    if kw not in kwargs or kwargs[kw] is None:
        kwargs[kw] = def_val
    if typ == 'ls':
        if not isinstance(kwargs[kw], list):
            kwargs[kw] = [kwargs[kw]]
    if typ == 'int':
        if not isinstance(kwargs[kw], int):
            kwargs[kw] = int(kwargs[kw])
    return kwargs[kw]


def create_new_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def date_to_doy(date_series):
    return pd.to_datetime(date_series).dt.dayofyear


if __name__ == '__main__':
    file_path = '/Users/maverickmiaow/Downloads/ee-chart.csv'
    df = pd.read_csv(file_path)
    df['doy'] = date_to_doy(df['date'])
    print(df)
