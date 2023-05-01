import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import fiona
import rasterio
import rasterio.mask
import os
import csv
import geopandas as gpd
import pyogrio
from osgeo import gdal
import numpy as np


def open_file(filename):
    df = pd.read_csv(f'{filename}')
    return df


def file_explorer(input_dir, country, locator):
    VIIRS_files = []
    if locator == 'input data':
        country = ''
    if input_dir == 'monthly':
        years = glob.glob(f'./{locator}/monthly/*')
        years_new = [year[-4:] for year in years]
        for year in years_new:
            print(f'./{locator}/{input_dir}/{year}/*.tif')
            VIIRS_files = VIIRS_files + glob.glob(f'./input data/{input_dir}/{year}/*{country}_masked.tif')
    else:
        VIIRS_files = glob.glob(f'./{locator}/{input_dir}/*{country}_masked.tif')
    return VIIRS_files


def map_clipper(input_dir, country ):
    VIIRS_files = file_explorer(input_dir, country, 'input data')
    if input_dir == 'monthly':
        years = glob.glob(f'./input data/monthly/*')
        years_new = [year[-4:] for year in years]
        for year in years_new:
            print(f'./input data/{input_dir}/{year}/*.tif')
            VIIRS_files = VIIRS_files + glob.glob(f'./input data/{input_dir}/{year}/*.tif')
    else:
        VIIRS_files = glob.glob(f'./input data/{input_dir}/*.tif')

    print(VIIRS_files)
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


def summation_analysis(input_dir, country):
    VIIRS_files = file_explorer(input_dir, country, 'input')
    summation_values = {}
    for VIIRS_file in VIIRS_files:
        dataset = gdal.Open(r'land_shallow_topo_2048.tif')
        print(dataset.RasterCount)
        band1 = dataset.GetRasterBand(1)
        b1 = band1.ReadAsArray()
        b1 = np.array(b1)
        summation_values[VIIRS_file.split('\\')][-1].replace('VNL_v21_npp_2013_global_vcmcfg_c', '')[:4] = b1
    return summation_values


if __name__ == '__main__':
    VIIRS_files = glob.glob('./data/*.csv')
    i = 1
    for file_name in VIIRS_files:
        print(f'processing file {i}')
        i = i + 1
        df = open_file(file_name)
        df_lighted = df[df['NTL'] != 0.0]
        fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
        plt.scatter(x=df_lighted['Longitude'], y=df_lighted['Latitude'], s=df_lighted['NTL'] * 0.00000003)
        plt.title(file_name[-8:-3])
        plt.show()
