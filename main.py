import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob


def open_file(filename):
    df = pd.read_csv(f'{filename}')
    return df


if __name__ == '__main__':
    VIIRS_files = csv_files = glob.glob('./data/*.csv')
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
