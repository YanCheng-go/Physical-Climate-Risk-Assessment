"""
Download  ERA data from GEE for all power plants to Google Drive.

To be updated:
    - Download data from google drive to local disk.
    - Directly saving output to local disk.
    - Progress bars.
"""

import ee, time
import pandas as pd
import os


def main(tpp_fp, imgcol_url="ECMWF/ERA5/DAILY", time_span=(1980, 2019), calcScale=30, start=0, save_output=True):
    """
    :param tpp_fp: file path-like string, the file path of tpp_working excel table.
    :param imgcol_url: string, gee reference link of a dataset.
    :param calcScale: int, spatial scale defined by GEE.
    :param start: int, index of rows in tpp_working table, indicating download data starting from which plant in the tpp_working table.
    :param save_output: boolean, whether to save output to Google Drive.
    :return: None
    """
    # Authenticates and initializes Earth Engine
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()

    gddp = ee.ImageCollection(imgcol_url)
    dfTpp = pd.read_excel(tpp_fp, sheet_name='tpp_locs')
    lats = dfTpp['Lat'].values
    lons = dfTpp['Lon'].values
    uids = dfTpp['Unique ID'].values

    startYear, endYear = time_span
    indicators = ee.List([
        'mean_2m_air_temperature',
        'minimum_2m_air_temperature',
        'maximum_2m_air_temperature',
        'dewpoint_2m_temperature',
        'total_precipitation',
        'surface_pressure',
        'mean_sea_level_pressure',
        'u_component_of_wind_10m',
        'v_component_of_wind_10m'
    ])
    indicatorLen = indicators.length().getInfo()

    def numerize(id):
        return ee.Number.parse(id)

    def getDataByLocation(f):
        img = ee.Image(f)
        imgReduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bounds,
            crs='EPSG:4326',
            scale=calcScale
        )
        tester = ee.String(indicator).rindex('total_precipitation')
        condition = ee.Algorithms.IsEqual(tester, -1)
        output = ee.Algorithms.If(
            condition,
            ee.Number(imgReduced.get(indicator)),
            ee.Number(imgReduced.get(indicator))
        )
        return output

    for lon, lat, uid in zip(lons[start:], lats[start:], uids[start:]):
        locationName = 'PL_EBRD_TPP%s_ERA5_' % uid
        print(locationName)
        nowPoint = ee.Geometry.Point([lon, lat])

        # try run more models at a time 5--OK!
        fileName = locationName + str(startYear) + '_' + str(endYear)
        featuresList = ee.List([])
        for y in range(endYear - startYear + 1):
            currentYear = startYear + y
            startDate = ee.Date.fromYMD(currentYear, 1, 1)
            endDate = ee.Date.fromYMD(currentYear + 1, 1, 1)
            dailyTemp = gddp.filterDate(startDate, endDate).select(indicators)
            for j in range(indicatorLen):
                indicator = indicators.get(j - 1).getInfo()
                bounds = nowPoint
                # print('bound area: ', bounds.area())
                reduced = dailyTemp.toList(dailyTemp.size()).map(getDataByLocation)
                # print(indicator, reduced)
                featuresList = featuresList.add(ee.Feature(ee.Geometry.Point([1, 1]),
                                                           {'PlantID': 9999, 'year': currentYear,
                                                            'indicatorName': indicator, 'data': reduced}))

        if save_output is True:
            # print (featuresList)
            task = ee.batch.Export.table.toDrive(
                collection=ee.FeatureCollection(featuresList),
                description=fileName,
                fileFormat='CSV'
            )
            task.start()
            # check status
            while task.active():
                time.sleep(30)
                print(task.status())

            # # To be updated
            # def download_from_google_drive():
            #     pass
            # download_from_google_drive()

    # print(f'Done with downloading ERA5 datasets. Find output here: {None}')


if __name__ == '__main__':
    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    main(tpp_fp=os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx'),
         imgcol_url="ECMWF/ERA5/DAILY", time_span=(1980, 2019), calcScale=5000, start=0, save_output=True)

    tpp_era5_folder_name = 'ear5'
    if not os.path.exists(os.path.join(work_directory, tpp_era5_folder_name)):
        os.mkdir(os.path.join(work_directory, tpp_era5_folder_name))
    print(f'Please download era5 datasets from google drive and save it here '
          f'{os.path.join(work_directory, tpp_era5_folder_name)}')
