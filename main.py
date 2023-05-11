import pandas as pd
import matplotlib.pyplot as plt
import rasterio.mask
import geopandas as gpd
import pyogrio
import matplotlib.cm as cm
from osgeo import gdal
import numpy as np
from geographiclib.geodesic import Geodesic
from IPython.display import display
from geopy.geocoders import Nominatim
import time
import shutil
import fiona
import glob
import rasterio
from rasterio.merge import merge
import rasterio.mask
from rasterio.errors import RasterioIOError
import os
import requests


# error handling from class lectures and https://www.programiz.com/python-programming/user-defined-exception
class UnrecognizedInputDir(Exception):
    """
    When the input directory is not within the excepted parameters

    Attributes:
        input_dir -- The input directory string that caused the error
        message -- explanation of the error
    """

    def __init__(self, input_dir, message="input directory is not within the expected options 'annual', 'monthly', or "
                                          "'treecover'"):
        self.input_dir = input_dir
        self.message = message
        super().__init__(self.message)


class UnplannedConditionError(Exception):
    """
    When the input directory is not within the excepted parameters

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message="You are seeing this message because you have attempted to run the program "
                               "in a condition it was not designed for. You may or may not be able to get "
                               "it working with modifications"):
        self.message = message
        super().__init__(self.message)


def open_file(filename):
    """
    The function opens the give name given, while right now this may not contain any major processing, the function has
     not been removed in anticipation of future common preprocessing requirements
    :param filename: path to the file that is required to be opened
    :return: returns the file object
    """
    df = pd.read_csv(f'{filename}')
    return df


def viirs_year_extractor(input_dir, filename):
    """
    This function extracts the year/ year-month from a given VIIRS filepath and returns it.
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'
    :param filename: the filepath that needs to be processed
    :return: either the year or the year-month string

    # Extract year from an annual filepath
    >>> viirs_year_extractor('annual', 'output\annual\VNL_v21_npp_2022_example.tif')
    '2022'

    # Extract year-month from a monthly filepath
    >>> viirs_year_extractor('monthly', 'output\to\monthly\SVDNB_npp_202206_example.tif')
    '202206'

    # Return 'treecover' for input_dir 'treecover'
    >>> viirs_year_extractor('treecover', 'output\treecover\file.tif')
    'treecover'

    # Raise an exception for unrecognized input_dir
    >>> try:
    ...     viirs_year_extractor('unknown', r'path\to\file.tif')
    ... except UnrecognizedInputDir as e:
    ...     print(str(e))
    input directory is not within the expected options 'annual', 'monthly', or 'treecover'
    """
    input_dir = input_dir.lower()
    if input_dir == 'monthly':
        return filename.split('\\')[-1].replace('SVDNB_npp_', '')[:6]
    elif input_dir == 'annual':
        return filename.split('\\')[-1].replace('VNL_v21_npp_', '')[:4]
    elif input_dir == 'treecover':
        return 'treecover'
    else:
        raise UnrecognizedInputDir(input_dir)


def file_explorer(input_dir, country, locator, file_type='tif'):
    """
    This function is the file explorer for the program, it is built to handle the file structure of the project.
    It works by taking in the various parameters of a file search and returning the list of files that satisfies the
    search parameters. This allows other functions to be less verbose and more dynamic. The function is branched put
    on the basis of the file type and the corresponding input file type
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'.
    :param country: The country name to be processed.
    :param locator: The 1st level file directory, usually 'data', input data', and 'output'.
    :param file_type: the type of file that's being searched, 'csv' or 'tif'.
    :return: returns the list of files that have been found.
    >>> file_explorer('annual', 'India', locator='output', file_type='jpeg')
    Traceback (most recent call last):
    ...
    FileNotFoundError
    >>> file_explorer('treecover', 'thailand', 'input data')
    Traceback (most recent call last):
    ...
    FileNotFoundError
    """
    input_dir = input_dir.lower()
    country = country.lower()
    locator = locator.lower()
    file_type = file_type.lower()
    VIIRS_files = []
    if locator == 'input data' and not (input_dir == 'treecover'):
        country = ''

    if file_type == 'tif':
        if input_dir == 'monthly':
            years = glob.glob(f'./{locator}/monthly/*')
            years_new = [year[-4:] for year in years]
            for year in years_new:
                if locator == 'input data':
                    VIIRS_files = glob.glob(f'./{locator}/{input_dir}/{year}/*.{file_type}')
                else:
                    VIIRS_files = VIIRS_files + glob.glob(f'./{locator}/{input_dir}/{year}/*'
                                                          f'{country}_masked.{file_type}')
        elif input_dir == 'treecover':
            if locator == 'input data':
                VIIRS_files = glob.glob(f'./{locator}/{input_dir}/{country}/*{file_type}')
            elif locator == 'output':
                VIIRS_files = glob.glob(f'./{locator}/{input_dir}/{country}_merged.{file_type}')
        elif input_dir == 'annual':
            if locator == 'input data':
                VIIRS_files = glob.glob(f'./{locator}/{input_dir}/*')
            else:
                VIIRS_files = glob.glob(f'./{locator}/{input_dir}/*{country}_masked.{file_type}')
        else:
            raise UnrecognizedInputDir(input_dir)
    elif file_type == 'csv':
        if input_dir == 'monthly':
            years = glob.glob(f'./{locator}/monthly/*')
            years_new = [year[-4:] for year in years]
            for year in years_new:
                VIIRS_files = VIIRS_files + glob.glob(f'./{locator}/{input_dir}/{year}/csv/*{country}_masked.'
                                                      f'{file_type}')
        elif input_dir == 'treecover':
            VIIRS_files = VIIRS_files + glob.glob(f'./{locator}/{input_dir}/{country}/*{file_type}')
        elif input_dir == 'annual':
            VIIRS_files = glob.glob(f'./{locator}/{input_dir}/csv/*{country}_masked.{file_type}')
        else:
            raise UnrecognizedInputDir(input_dir)
    if not VIIRS_files:
        raise FileNotFoundError
    return VIIRS_files


# the following way of cropping a large geotiff image to a national boundary has been obtained from
# https://stackoverflow.com/questions/69938501/clipping-raster-through-a-shapefile-using-python
def map_clipper(input_dir, country, locator='input data'):
    """
    This function clips the given map to the bounds of the country and saves it as a new file
    :param locator: The level 1 file directory
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'.
    :param country: The country name to be processed.
    """
    input_dir = input_dir.lower()
    country = country.lower()
    locator = locator.lower()

    VIIRS_files = file_explorer(input_dir, country, locator)

    shp_file_path = glob.glob(f"./input data/shape files/{country}/*.shp")[0]

    # open the shape file
    with fiona.open(shp_file_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    # process the input files against the shapefiles
    for VIIRS_file in VIIRS_files:
        if input_dir == 'monthly':
            output_raster_path = './output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:] + "/" + \
                                 VIIRS_file.split('\\')[-1][:-4] + f"_{country}_masked.tif "
        else:
            output_raster_path = './output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-1][:-4] + \
                                 f"_{country}_masked.tif "

        try:
            with rasterio.open(VIIRS_file) as src:
                out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
                out_meta = src.meta
        except RasterioIOError:
            continue

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "compress": "lzw",
                         "transform": out_transform})
        if (not os.path.exists('./output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:])) and input_dir == \
                'monthly':
            os.makedirs('./output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:])
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)


# the following method of sampling the data points from a raster layer has been obtained from:
# https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations
def sampling(input_dir, country):
    """
    The function samples the required files against the sampling locations data which have a standard set of
    latitude and longitude. Each of the sampling point will be assigned its corresponding value from the raster.
    This output will be saved in a csv
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'
    :param country: The country name to be processed.

    """
    input_dir = input_dir.lower()
    country = country.lower()

    VIIRS_files = file_explorer(input_dir, country, 'output')
    print('start')
    print(VIIRS_files)
    print(glob.glob(f"./input data/sampling points/{country}/*.shp")[0])
    pts = pyogrio.read_dataframe(glob.glob(f"./input data/sampling points/{country}/*.shp")[0])
    print('conversion')
    pts = gpd.GeoDataFrame(pts, geometry=gpd.points_from_xy(pts.Longitude, pts.Latitude))
    print('start shapefile')
    pts = pts[['Latitude', 'Longitude', 'geometry']]
    pts.index = range(len(pts))
    print('shapefile step 1')
    coords = [(x, y) for x, y in zip(pts.Longitude, pts.Latitude)]  # use zip(pts.geometry.x, pts.geometry.y)
    # if you don't have specific coordinates
    print('end shapefile')
    for VIIRS_file in VIIRS_files:

        # Read points from shapefile
        print('start CSV printing round')

        # Open the raster and store metadata
        src = rasterio.open(VIIRS_file)
        pts['Raster Value'] = [x[0] for x in src.sample(coords)]
        if (not os.path.exists('./output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:])) and input_dir == \
                'monthly':
            os.makedirs('./output/' + f'{input_dir}/' + '/csv/' + VIIRS_file.split('\\')[-2][-4:])

        if input_dir == 'monthly':
            pts[['Latitude', 'Longitude', 'Raster Value']].to_csv(
                './output/' + f'{input_dir}/' + '/csv/' + VIIRS_file.split('\\')[-2][-4:] + "/" +
                VIIRS_file.split('\\')[-1][:-4] + f"_{country}_masked.csv")
        else:
            pts[['Latitude', 'Longitude', 'Raster Value']].to_csv(
                './output/' + f'{input_dir}/' + '/csv/' + VIIRS_file.split('\\')[-1][:-4]
                + f"_{country}_masked.csv")
        print('end CSV printing round')


# The following style of data extraction has been inferred from
# https://www.geeksforgeeks.org/visualizing-tiff-file-using-matplotlib-and-gdal-using-python/#
def data_analysis(input_dir, country):
    """
    Extracts the data from a given tif raster files and converts it to an array
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'
    :param country: The country name to be processed.
    :return: A dictionary with arrays with all the values on a raster
    >>> data_analysis('annual', 'india').keys()
    dict_keys(['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'])
    >>> data_analysis('treecover','india').keys()
    dict_keys(['treecover'])
    """
    input_dir = input_dir.lower()
    country = country.lower()

    VIIRS_files = file_explorer(input_dir, country, 'output')
    extracted_data = {}
    for VIIRS_file in VIIRS_files:
        dataset = gdal.Open(rf'{VIIRS_file}')
        band1 = dataset.GetRasterBand(1)
        b1 = band1.ReadAsArray()
        # extracted_data[viirs_year_extractor(input_dir, VIIRS_file)] = np.array(b1)
        extracted_data[viirs_year_extractor(input_dir, VIIRS_file)] = b1
    return extracted_data


def summation_analysis(input_dir, country, direction=None, cutoff: float = 0.98, get_plot=True, get_data_return=False,
                       extrapolate_economic_data=False):
    """
    This function generated analyzes the data from the required files, it can be manipulated to reuse the same set of
    code for a multitude of functionalities .
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'
    :param country: The country name to be processed.
    :param direction: The direction in which the function's analysis needs to proceed. This is in the context of
     quantile analysis. Either 'top' or 'bottom' is expected.
    :param cutoff: The cutoff point for the quantile wise analysis.
    :param get_plot: The selection option on weather or not a plot needs to be printed by the function
    :param get_data_return: The selection option on weather or not the program should return the data it has generated
    :param extrapolate_economic_data: The selection option to weather or not the function should also plot the
     economic data
    :return: return the data that has been worked on.
    >>> summation_analysis('annual', 'india', get_plot=False, get_data_return=True).keys()
    dict_keys(['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021'])
    >>> try:
    ...     summation_analysis('monthly', 'india', extrapolate_economic_data= True)
    ... except UnplannedConditionError as e:
    ...     print(str(e))
    You are seeing this message because you have attempted to run the program in a condition it was not designed for.\
 You may or may not be able to get it working with modifications
    >>> summation_analysis('monthly', 'india', get_plot=False, get_data_return=True).keys()
    dict_keys(['202201', '202202', '202203', '202204', '202205', '202206', '202207', '202209', '202210', '202211'])
    >>> try:
    ...     summation_analysis('unknown', 'country')
    ... except UnrecognizedInputDir as e:
    ...     print(str(e))
    input directory is not within the expected options 'annual', 'monthly', or 'treecover'
    """
    if input_dir == 'monthly' and extrapolate_economic_data:
        raise UnplannedConditionError
    elif input_dir == 'treecover':
        raise UnplannedConditionError
    elif input_dir != 'annual' and input_dir != 'monthly':
        raise UnrecognizedInputDir(input_dir)
    VIIRS_files = data_analysis(input_dir, country)

    # sum the data
    for year in VIIRS_files.keys():
        if direction is None:
            VIIRS_files[year] = VIIRS_files[year].sum()
        elif direction == 'top':
            quantile = np.quantile(VIIRS_files[year], cutoff)
            VIIRS_files[year] = VIIRS_files[year][VIIRS_files[year] > quantile].sum()
        elif direction == 'bottom':
            quantile = np.quantile(VIIRS_files[year], cutoff)
            VIIRS_files[year] = VIIRS_files[year][VIIRS_files[year] < quantile].sum()

    if extrapolate_economic_data:
        df_econ = pd.read_csv('./data/economic data/API_NY.GDP.PCAP.CD_DS2_en_csv_v2_5358417.csv')
        country_df = df_econ[df_econ['Country Name'] == country.capitalize()].loc[:, '2012':'2021']
    if get_plot and (not extrapolate_economic_data):
        plt.xlabel('Year')
        plt.ylabel('total brightness')
        plt.title(f'{input_dir.capitalize()} {country.capitalize()} Night Light, Quantile: {direction}')
        plt.plot(VIIRS_files.keys(), VIIRS_files.values())
        plt.xticks(rotation=90)
        plt.ylim(bottom=0)
        plt.show()

    if get_plot and extrapolate_economic_data:
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('Year')
        ax1.set_ylabel('total brightness')
        ax2.set_ylabel('GDP per capita (current US$)')
        ax1.set_title(f'{input_dir.capitalize()} {country.capitalize()} Night Light, Quantile: {direction}'
                      )
        ax1.plot(VIIRS_files.keys(), VIIRS_files.values(), label='Night Light')
        ax2.plot(country_df.columns, country_df.iloc[0], label='economy', linestyle='dotted')
        ax2.set_ylim(ymin=0)
        ax1.set_ylim(ymin=0)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels, loc='center left')
        plt.show()

    if get_data_return and (not extrapolate_economic_data):
        return VIIRS_files

    if get_data_return and extrapolate_economic_data:
        return VIIRS_files, country_df


def city_check(city_latitude, city_longitude, radius, df_all, title, skip_plotter: str):
    """
    The function analyzes the data around a city and prints the plots if needed.
    :param city_latitude: The city's latitude.
    :param city_longitude: The city's longitude.
    :param radius: The radius upto which the function needs to analyze data
    :param df_all: The complete dataframe
    :param title: The title for the plot
    :param skip_plotter: The selection option to skip the printing of the plot. Expects a 'N' or 'Y'
    :return: return the total raster value of a city
    """
    result = Geodesic.WGS84.Direct(city_latitude, city_longitude, 0, radius)
    north_lat = result['lat2']
    result = Geodesic.WGS84.Direct(city_latitude, city_longitude, 90, radius)
    east_lon = result['lon2']
    result = Geodesic.WGS84.Direct(city_latitude, city_longitude, 270, radius)
    west_lon = result['lon2']
    result = Geodesic.WGS84.Direct(city_latitude, city_longitude, 180, radius)
    south_lat = result['lat2']
    # print(f'{north_lat}|{east_lon}|{west_lon}|{south_lat}')
    df_city = df_all[((df_all['Latitude'] <= north_lat) & (df_all['Latitude'] >= south_lat)) &
                     ((df_all['Longitude'] >= west_lon) & (df_all['Longitude'] <= east_lon))]
    if not (skip_plotter.upper() == "N"):
        plt.figure(figsize=(12, 8), facecolor='white')
        plt.scatter(x=df_city['Longitude'], y=df_city['Latitude'], s=df_city['Raster Value'] * 0.5)
        plt.title(f'{title}')
        plt.show()
    return df_city['Raster Value'].sum()


def city_isolation(input_dir, country: str, print_national_stat=False):
    """
    The function analyzes the cities in a country, by a population cutoff.
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'
    :param country: The country name to be processed.
    :param print_national_stat: The selection option to toggle the printing of the national stats to the plot
    """
    input_dir = input_dir.lower()
    country = country.lower()

    if input_dir == 'monthly' or input_dir == 'treecover':
        raise UnplannedConditionError
    elif input_dir != 'annual':
        raise UnrecognizedInputDir(input_dir)
    df_cities = pd.read_csv('./data/worldcities.csv')
    df_cities = df_cities[df_cities['country'] == country.capitalize()]
    VIIRS_files = file_explorer(input_dir, country, 'output', file_type='csv')

    population_cutoff = input(f'Input the population cutoff of the city you want to time visualize: ')
    skip_visualization = input(f'Type "N" to skip visualization by individual city, Y for otherwise: ')
    legend_labels = {}

    try:
        population_cutoff = int(population_cutoff)
        df_cities = df_cities[df_cities['population'] >= population_cutoff]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.set_xlabel('Year')
        ax1.set_ylabel('total brightness')
        ax1.set_title(f'{country.capitalize()} City-wise Total Night Light')
        if print_national_stat:
            country_dict = summation_analysis(input_dir, country, get_plot=False, get_data_return=True)
            ax2.plot(country_dict.keys(), country_dict.values(), label=country, linestyle='dotted')
        for index, row in df_cities.iterrows():
            radius = input(f'Enter the radius for the city {row["city_ascii"]} in meters: ')
            print(f'{row["city_ascii"]}')
            for file_name_city in VIIRS_files:
                year = viirs_year_extractor(input_dir, file_name_city)
                df = open_file(file_name_city)
                legend_labels[year] = city_check(row['lat'], row['lng'], int(radius), df, f"{row['city_ascii']} - "
                                                                                    f"{year}", skip_visualization)
            ax1.plot(legend_labels.keys(), legend_labels.values(), label=row["city_ascii"])

        ax2.set_ylim(ymin=0)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels, loc='center left')
        plt.show()
    except TypeError:
        pass


def distribution_curve(input_dir, country):
    """
    Prints a distribution curve histogram for a given country.
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , 'monthly', or 'treecover'.
    :param country: The country name to be processed.
    """
    if input_dir == 'treecover':
        raise UnplannedConditionError
    elif input_dir != 'annual' and input_dir != 'monthly':
        raise UnrecognizedInputDir(input_dir)
    display('Analysis begin', display_id='message')
    VIIRS_files = data_analysis(input_dir, country)
    colors = cm.rainbow(np.linspace(0, 1, len(VIIRS_files)))
    bins = np.linspace(0, 4, num=400)
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.title(f'{input_dir.capitalize()} {country.capitalize()} Night Light distribution')
    legend_labels = []

    for year, color in zip(VIIRS_files.keys(), colors):
        display(f'processing file {year}', display_id='message', update=True)
        plt.hist(VIIRS_files[year][VIIRS_files[year] > 0].flatten(), label=year, bins=bins, color=color, alpha=0.5)

    display(f'', display_id='message', update=True)
    plt.legend(loc='upper right')
    plt.show()


def Location_analysis(input_dir, country):
    """
    Finds the addresses of the brightest locations across the country.
    :param input_dir: The input directory, It is the 2nd level file directory, can be expected to be 'annual'
    , or 'monthly'
    :param country:
    """
    display('Analysis start', display_id='message1')
    VIIRS_files = file_explorer(input_dir, country, "output", file_type="csv")
    for file_name in VIIRS_files:
        print(f'processing file {viirs_year_extractor(input_dir, file_name)}')
        display(f'processing file {viirs_year_extractor(input_dir, file_name)}', display_id='message1', update=True)
        df = open_file(file_name)
        df_lighted = df[df['Raster Value'] > 0.0]
        print(f'Minimum brightness and Location')
        print(f'{df_lighted[df_lighted["Raster Value"] == min(df_lighted["Raster Value"])]}')
        print(f'Maximum brightness and Location')
        print(f'{df_lighted[df_lighted["Raster Value"] == max(df_lighted["Raster Value"])]}')
        print('\n')
        df_new = df_lighted.sort_values(by='Raster Value', ascending=False)
        top_3 = df_new.head(3)
        geolocator = Nominatim(user_agent='my_project')
        for index, row in top_3.iterrows():
            lat = row['Latitude']
            long = row['Longitude']
            location = geolocator.reverse(f"{lat}, {long}", timeout=None)
            print(f"Address: {location.address}, \n Brightness: {row['Raster Value']}")
            print('\n')
            print('__________________________________________________________________________________________________')
            print('\n')
            time.sleep(3)


def treecover_downloader(country, url: str):
    """
    This function downloads the tree cover geotiff file from the server
    :param country: The country name to be processed.
    :param url: the url from which the data is to be downloaded
    """
    r = requests.get(url, allow_redirects=True)
    if not os.path.exists(f'./input data/treecover/{country}'):
        os.makedirs(f'./input data/treecover/{country}')
    open(f'./input data/treecover/{country}/{url.split("/")[-1]}', 'wb').write(r.content)


# Open the GeoTIFF file
def treecover_tif_stitch(country):
    """
    This function gets the tree cover geotiff files for a country merges the individual files to create one big raster
    :param country: The country name to be processed.
    """
    shape_file_path = glob.glob(f"./input data/shape files/{country}/*.shp")
    map_range = [[], []]
    with fiona.open(shape_file_path[0], "r") as src:
        # Get the boundary coordinates in the CRS units
        left, bottom, right, top = src.bounds
        # Get the CRS information
        # Print the results

    left = (left // 10) * 10
    bottom = (bottom // 10) * 10
    right = ((right + 10) // 10) * 10
    top = ((top + 10) // 10) * 10

    for lat in range(int(bottom), int(top), 10):
        if lat < 0:
            map_range[0].append(f'{lat}S')
        else:
            map_range[0].append(f'{lat}N')
    for lon in range(int(left), int(right), 10):
        if len(str(lon)) < 3:
            lon = '0' + str(lon)
        if int(lon) < 0:
            map_range[1].append(f'{lon}W')
        else:
            map_range[1].append(f'{lon}E')

    for lat in map_range[0]:
        for lon in map_range[1]:
            treecover_downloader(country,
                                 url=f'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2021-v1.9/'
                                     f'Hansen_GFC-2021-v1.9_lossyear_{lat}_{lon}.tif')

    # The technique to merging the individual tif raster files has been obtained from
    # https://rasterio.readthedocs.io/en/latest/api/rasterio.merge.html
    snippet_file_paths = glob.glob(f'./input data/treecover/{country}/*.tif')

    snippet_file_raster_set = []
    run_count = False
    for snippet_file_path in snippet_file_paths:
        try:
            snippet_file_raster = rasterio.open(snippet_file_path)
            snippet_file_raster_set.append(snippet_file_raster)
        except RasterioIOError:
            pass
    out_image, out_transform = merge(snippet_file_raster_set)
    out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "count": out_image.shape[0],
                     "compress": "lzw",
                     "transform": out_transform,
                     "dtype": "uint8"})

    with rasterio.open(f'./output/treecover/{country}_merged.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    for snippet_file_raster in snippet_file_raster_set:
        snippet_file_raster.close()
    shutil.rmtree(f'./input data/treecover/{country}')


def treecover_tracker(country):
    '''
    This function plots the forest cover loss rate for a given county
    :param country: The name of the country that is being processed.
    '''
    h1 = display('Analysis begin', display_id='message')
    file_data = data_analysis('treecover', country)
    plt.xlabel('year')
    plt.ylabel('tree cover loss')
    plt.title(f'The tree cover loss tracker')
    count = {}
    for year in range(12, 21):
        display(f'working on the {year + 2000}', display_id=f'message', update=True)
        count[year + 2000] = np.count_nonzero(file_data['treecover'] == year)

    plt.plot(count.keys(), count.values())
    plt.show()


if __name__ == '__main__':
    """ This is a placeholder main function please refer to the main.ipynb file for executing the program"""
    pass
