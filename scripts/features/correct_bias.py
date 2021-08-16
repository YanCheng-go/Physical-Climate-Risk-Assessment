'''
Bias correlation based on continuous density function for climate data
'''

import numpy as np
import pandas as pd
import sys


def refactor_input(in_data):
    return sorted(in_data)


def cdf_transform(obs, bc, cdfn, global_max, global_min):
    if len(obs) == len(bc):
        # cdfn = len(obs)
        obs, bc = refactor_input(obs), refactor_input(bc)
        data = np.transpose([obs, bc])
        columns = ['Obs', 'Bc']
        df = pd.DataFrame(data=data, columns=columns)
        df.sort_values(by=['Obs'], inplace=True)
        obsSo = df['Obs']
        df.sort_values(by=['Bc'], inplace=True)
        bcSo = df['Bc']
        wide = (global_max - global_min) / cdfn
        xbins = np.arange(global_min, global_max + wide, wide)
        pdfobs, bins = np.histogram(obsSo, bins=xbins)
        pdfbc, bins = np.histogram(bcSo, bins=xbins)
        cdfobs = np.cumsum(pdfobs)
        cdfbc = np.cumsum(pdfbc)
    else:
        sys.exit('The length of obj values and bc values are not the same!')
    return cdfobs, cdfbc, xbins


def correct_bias_func(in_data, obs, bc, cdfn, global_max, global_min, reverse_zeros=None):
    in_data, obs, bc = refactor_input(in_data), refactor_input(obs), refactor_input(bc)
    cdfobs, cdfbc, xbins = cdf_transform(obs, bc, cdfn, global_max, global_min)
    cdf1 = np.interp(in_data, xbins[1:], cdfbc)
    out_data_temp = np.interp(cdf1, cdfobs, xbins[1:])
    if reverse_zeros is True:
        dfOut = pd.DataFrame({'Data': in_data, 'Corrected_temp': out_data_temp})
        dfOut['ZeroOrNot'] = dfOut.apply(lambda row: zero_or_not(row), axis=1)
        dfOut['Corrected'] = dfOut['Corrected_temp'] * dfOut['ZeroOrNot']
        out_data = dfOut['Corrected'].values
    else:
        out_data = out_data_temp
    return out_data


def zero_or_not(row):
    if row['Data'] == 0.0:
        z = 0.0
    else:
        z = 1.0
    return z
