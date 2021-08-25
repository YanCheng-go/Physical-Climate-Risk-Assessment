"""
Data configurations / metadata.
"""

# Data pointer
FOLDER_NAME = {
    'SPEI': 'spei',
    'GDDP': 'tpp_climate_gddp_restructure_withAirTempAvg',
    'ERA5': 'era5_wetbulbtemp',
    'WTUU': 'watertemp_output_temp',
}

TIMESPAN = {
    'SPEI': (None, None),  # Retrieve from datasets
    'GDDP': ('1950-2005', '2030-2070'),
    'ERA5': ('1980-2019', None),
    'WTUU': ('1965-2010', '2030-2069'),
}  # timespan that indicates on the file name

FILE_NAME_PREFIX = {
    'SPEI': (None, None),
    'GDDP': ('GDDP', 'GDDP'),
    'ERA5': ('ERA5', 'ERA5'),
    'WTUU': ('waterTemperature-mergedV2', 'waterTemp-WeekAvg'),
}

FILE_NAME_SURFIX = {
    'SPEI': 'cmip5',
    'GDDP': 'withAirTempAvg',
    'ERA5': 'restructure_withWetBulbTemp',
    'WTUU': None,
}

RAWDATA_TIMESPAN = {
    'SPEI': (None, None),  # Retrieve from datasets
    'GDDP': ('1950-2005', '2006-2070'),
    # 'GDDP': ('1980-2019', ''), # -> Indian plants for validation, i.e., using ERA5 datastes
    'ERA5': ('1980-2019', None),
    'WTUU': ('1965-2010', '2030-2069'),
}  # timespan that indicates on the file name

# File name conventions


# Standardized DataFrame structure


# Metadata
climate_model_list = {
    'WTUU': ['noresm', 'miroc', 'gfdl', 'ipsl', 'hadgem'],
}
climate_scenario_list = {
    'WTUU': ['rcp4p5', 'rcp8p5'],
}
