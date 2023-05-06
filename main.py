import pandas as pd
import matplotlib.pyplot as plt
import glob
import fiona
import rasterio
import rasterio.mask
import os
import geopandas as gpd
import pyogrio
import matplotlib.cm as cm
from osgeo import gdal
import numpy as np
from geographiclib.geodesic import Geodesic
from IPython.display import display
from geopy.geocoders import Nominatim
import time


def open_file(filename):
    df = pd.read_csv(f'{filename}')
    return df


def viirs_year_extractor(input_dir, filename):
    if input_dir == 'monthly':
        return filename.split('\\')[-1].replace('SVDNB_npp_', '')[:6]
    else:
        return filename.split('\\')[-1].replace('VNL_v21_npp_', '')[:4]


def file_explorer(input_dir, country, locator, file_type='tif'):
    VIIRS_files = []
    if locator == 'input data':
        country = ''
    if file_type == 'tif':
        if input_dir == 'monthly':
            years = glob.glob(f'./{locator}/monthly/*')
            years_new = [year[-4:] for year in years]
            for year in years_new:
                VIIRS_files = VIIRS_files + glob.glob(f'./{locator}/{input_dir}/{year}/*{country}_masked.{file_type}')
        else:
            VIIRS_files = glob.glob(f'./{locator}/{input_dir}/*{country}_masked.{file_type}')
    elif file_type == 'csv':
        if input_dir == 'monthly':
            years = glob.glob(f'./{locator}/monthly/*')
            years_new = [year[-4:] for year in years]
            for year in years_new:
                VIIRS_files = VIIRS_files + glob.glob(f'./{locator}/{input_dir}/{year}/csv/*{country}_masked.{file_type}')
        else:
            VIIRS_files = glob.glob(f'./{locator}/{input_dir}/csv/*{country}_masked.{file_type}')
    return VIIRS_files


# the following way of cropping a large geotiff image to a national boundary has been obtained from
# https://stackoverflow.com/questions/69938501/clipping-raster-through-a-shapefile-using-python
def map_clipper(input_dir, country):
    VIIRS_files = file_explorer(input_dir, country, 'input data')
    if input_dir == 'monthly':
        years = glob.glob(f'./input data/monthly/*')
        years_new = [year[-4:] for year in years]
        for year in years_new:
            VIIRS_files = VIIRS_files + glob.glob(f'./input data/{input_dir}/{year}/*.tif')
    else:
        VIIRS_files = glob.glob(f'./input data/{input_dir}/*.tif')

    shp_file_path = glob.glob(f"./input data/shape files/{country}/*.shp")[0]

    with fiona.open(shp_file_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
    for VIIRS_file in VIIRS_files:
        if input_dir == 'monthly':
            output_raster_path = './output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:] + "/" + \
                                 VIIRS_file.split('\\')[-1][:-4] + f"_{country}_masked.tif "
        else:
            output_raster_path = './output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-1][:-4] + f"_{country}" \
                                                                                                   f"_masked.tif "
        with rasterio.open(VIIRS_file) as src:
            out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
            out_meta = src.meta

        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        if (not os.path.exists('./output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:])) and input_dir == \
                'monthly':
            os.makedirs('./output/' + f'{input_dir}/' + VIIRS_file.split('\\')[-2][-4:])
        with rasterio.open(output_raster_path, "w", **out_meta) as dest:
            dest.write(out_image)


# the following method of sampling the data points from a raster layer has been obtained from:
# https://gis.stackexchange.com/questions/317391/python-extract-raster-values-at-point-locations
def sampling(input_dir, country):
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
    VIIRS_files = file_explorer(input_dir, country, 'output')
    extracted_data = {}
    for VIIRS_file in VIIRS_files:
        dataset = gdal.Open(rf'{VIIRS_file}')
        band1 = dataset.GetRasterBand(1)
        b1 = band1.ReadAsArray()
        extracted_data[viirs_year_extractor(input_dir, VIIRS_file)] = np.array(b1)
    return extracted_data


def summation_analysis(input_dir, country, direction=None, cutoff=0.98, get_plot=True, get_data_return=False,
                       extrapolate_economic_data=False):
    VIIRS_files = data_analysis(input_dir, country)
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
    if not (skip_plotter.upper()=="N"):
        plt.figure(figsize=(12, 8), facecolor='white')
        plt.scatter(x=df_city['Longitude'], y=df_city['Latitude'], s=df_city['Raster Value']*0.5)
        plt.title(f'{title}')
        plt.show()
    return df_city['Raster Value'].sum()


def city_isolation(input_dir, country: str, print_national_stat=False):
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
            print(f'{row["city_asciicity_ascii"]}')
            for file_name_city in VIIRS_files:
                year = viirs_year_extractor(input_dir, file_name_city)
                df = open_file(file_name_city)
                legend_labels[year] = city_check(row['lat'], row['lng'], 55000, df, f"{row['city_asciicity_ascii']} - "
                                                                                    f"{year}", skip_visualization)
            ax1.plot(legend_labels.keys(), legend_labels.values(), label=row["city_asciicity_ascii"])

        ax2.set_ylim(ymin=0)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2
        plt.legend(handles, labels, loc='center left')
        plt.show()
    except TypeError:
        print('test')


def distribution_curve(input_dir, country):
    h1 = display('Analysis begin', display_id='message')
    VIIRS_files = data_analysis(input_dir, country)
    colors = cm.rainbow(np.linspace(0, 1, len(VIIRS_files)))
    bins = np.linspace(0, 4, num=400)
    plt.xlabel('Brightness')
    plt.ylabel('Frequency')
    plt.title(f'{input_dir.capitalize()} {country.capitalize()} Night Light distribution')
    legend_labels = []

    for year, color in zip(VIIRS_files.keys(), colors):
        display(f'processing file {year}', display_id='message', update=True)
        plt.hist(VIIRS_files[year][VIIRS_files[year]>0].flatten(), label=year, bins=bins, color=color, alpha=0.5)
    display(f'', display_id='message', update=True)
    plt.legend(loc='upper right')
    plt.show()


def Location_analysis(input_dir, country):
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


if __name__ == '__main__':
    VIIRS_files_main = glob.glob('./data/*.csv')
    i = 1
    for file_name in VIIRS_files_main:
        print(f'processing file {i}')
        i = i + 1
        df_main = open_file(file_name)
        df_lighted = df_main[df_main['Raster Value'] != 0.0]
        fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
        plt.scatter(x=df_lighted['Longitude'], y=df_lighted['Latitude'], s=df_lighted['Raster Value'] * 0.00000003)
        plt.title(file_name[-8:-3])
        plt.show()
