"""
Download GDDP datasets for a given coordinate
"""

import ee, time, os
import pandas as pd, numpy as np


def main(tpp_fp, data_folder, imgcol_url="NASA/NEX-GDDP", hist_timespan=(1950, 2005), futu_timespan=(2006, 2070),
         scenario_list=None, model_list=None, calcScale=30, start=0, save_output=True):
    """
    :param tpp_fp: file path-like string, the file path of tpp_working excel table.
    :param data_folder: foler path-like string, the folder where to save the dataset downloaded.
    :param imgcol_url: string, gee reference link of a dataset
    :param hist_timespan: tuple or list, the historical time span in the form of (start year, end year).
    :param futu_timespan: tuple or list, the projection/future time span in the form of (start year, end year).
    :param scenario_list: list, a list of climate scenarios, if the value is None, all scenarios will be considered, for GDDP database, the scenarios are ['historical', 'rcp45', 'rcp85']
    :param model_list: list, a list of climate models, if the value is None, all models will be condidered.
    :param calcScale: int, spatial scale defined by GEE.
    :param start: int, index of rows in tpp_working table, indicating download data starting from which plant in the tpp_working table.
    :param save_output: boolean, whether to save output to local disk.
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

    # lon, lat = 23.374721, 42.950212
    startYearFutu, endYearFutu = futu_timespan
    startYearHist, endYearHist = hist_timespan
    indicators = ee.List(['pr', 'tasmin', 'tasmax'])
    scenarios = ['historical', 'rcp45', 'rcp85'] if scenario_list is None else scenario_list
    models = ['inmcm4', 'MPI-ESM-LR', 'MRI-CGCM3', 'IPSL-CM5A-LR', 'MIROC-ESM-CHEM', 'NorESM1-M', 'GFDL-ESM2M', 'CCSM4',
              'CNRM-CM5'] if model_list is None else model_list
    indicatorLen = indicators.length().getInfo()

    def numerize(id):
        return ee.Number.parse(id)

    # get pr tasmin tasmax
    def getDataByLocation(f):
        img = ee.Image(f)
        imgReduced = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=bounds,
            crs='EPSG:4326',
            scale=calcScale
        )
        tester = ee.String(indicator).rindex('pr')
        condition = ee.Algorithms.IsEqual(tester, -1)
        output = ee.Algorithms.If(
            condition,
            ee.Number(imgReduced.get(indicator)),
            ee.Number(imgReduced.get(indicator))
        )
        return output

    for lon, lat, uid in zip(lons[start:], lats[start:], uids[start:]):
        locationName = 'V2_PL_EBRD_TPP%s_' % uid
        print(locationName)
        nowPoint = ee.Geometry.Point([lon, lat])
        for scenario in scenarios:
            if scenario == 'historical':
                startYear, endYear = startYearHist, endYearHist
            else:
                startYear, endYear = startYearFutu, endYearFutu
            print(scenario, startYear, endYear)
            for model in models:
                years = []
                data = []
                metrics = []
                fileName = locationName + scenario + '_' + model + '_' + str(startYear) + '_' + str(endYear)
                featuresList = ee.List([])
                for y in range(endYear - startYear + 1):
                    currentYear = startYear + y
                    startDate = ee.Date.fromYMD(currentYear, 1, 1)
                    endDate = ee.Date.fromYMD(currentYear + 1, 1, 1)
                    dailyTemp = gddp.filterDate(startDate, endDate).filter(ee.Filter.eq('scenario', scenario)).filter(
                        ee.Filter.eq('model', model)).select(indicators)
                    for j in range(indicatorLen):
                        indicator = indicators.get(j - 1).getInfo()
                        bounds = nowPoint
                        reduced = dailyTemp.toList(dailyTemp.size()).map(getDataByLocation)
                        d = reduced.getInfo()
                        print(currentYear, indicator, scenario, model)
                        print(d)
                        years.append(currentYear)
                        data.append(d)
                        metrics.append(indicator)
                data = np.transpose([metrics, years, data])
                cols = ['indicator', 'year', 'data']
                df = pd.DataFrame(data, columns=cols)

                if save_output is True:
                    if not os.path.exists(data_folder):
                        os.mkdir(data_folder)
                    out_fp = os.path.join(data_folder, '%s.xlsx' % fileName)
                    df.to_excel(out_fp, index=None)

    print(f'Done with downloading GDDP datasets. Find output here: {data_folder}')


if __name__ == '__main__':
    work_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
    main(tpp_fp=os.path.join(work_directory, 'tpp info', 'tpp_working.xlsx'),
         data_folder=os.path.join(work_directory, 'tpp_climate_gddp_all'),
         imgcol_url="NASA/NEX-GDDP", hist_timespan=(1950, 2005), futu_timespan=(2006, 2070),
         scenario_list=None, model_list=None, calcScale=5000, start=0, save_output=True)
