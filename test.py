
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
import glob
matplotlib.use('nbAgg')

def open_file(filename):
    df_def = pd.read_csv(f'{filename}', chunksize=100000)
    i = 100
    chunks = []
    for chunk_def in df_def:
        chunks.append(chunk_def)
        i = i-1
        if i == 0:
            break
    return chunks


def file_scatter_plot(file_name):
    df = open_file(file_name)
    fig = plt.figure(figsize=(12, 8), dpi=300, facecolor='white')
    plt.title(file_name[-8:-4])
    ax = fig.add_subplot(111)
    for chunk in df:
        df_lighted = chunk[chunk['NTL'] > 0.0]
        ax.scatter(x=df_lighted['Longitude'], y=df_lighted['Latitude'], s=df_lighted['NTL']*0.0000000005)
    return fig
