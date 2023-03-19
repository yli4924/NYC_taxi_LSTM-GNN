import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from pandas import concat


# function for generating the lagged matrix
def split_sequence(sequence, window_size):
    X = []
    y = []
    # for all indexes
    for i in range(len(sequence)):
        end_idx = i + window_size
        # exit condition
        if end_idx > len(sequence) - 1:
            break
        # get X and Y values
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# series to supervised function
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg



def read_data(file_name, prints = False):

    url = "datasets/"
    path = Path.cwd()
    df = pd.read_csv(path.joinpath(url+file_name))

    print(df.columns)
    print(df.shape)
    #print(df.describe())
    print(df["datetime"].dtype)
    meta_df = df.describe()
    meta_df.to_csv(path.joinpath("datasets/meta_data.csv"))
    if prints:
        print("medallion num of unique",df["medallion"].nunique())
        print("hack_license num of unique",df["hack_license"].nunique())
        print("vendor_id num of unique",df["vendor_id"].nunique())
        print("rate_code num of unique",df["rate_code"].nunique())

        print("passenger_count min:", df["passenger_count"].min(),"and max:", df["passenger_count"].max() )
        print("datetime durations from:",df['datetime'].min(), "to:", df['datetime'].max())
    return df

def plot_graph(df, fig_name):
    plt.figure(figsize=(15,10))
    graph_1 = sns.scatterplot(x="longitude", y="latitude", data=df, s=1)
    graph_1.set_xlim([-74.04, -73.75])
    graph_1.set_ylim([40.62, 40.89])
    graph_1.set_xlabel('Longtitude')
    graph_1.set_ylabel('Latitude')
    plt.title("Map")
    plt.savefig("result/"+fig_name)

def plot_timeSeries(df,fig_name):
    plt.figure(figsize=(15,10))
    graph_1 = sns.scatterplot(x="datetime", y="trip_distance", data=df, s=1)
    graph_1.set_xlim(['2013-01-01', '2013-01-01'])
    graph_1.set_ylim([0, 10])
    plt.title("trip time for a driver")
    plt.savefig("result/"+fig_name)

def check_map(df):
    df_filt = df.query("(longitude > -74.03) | (longitude < -73.75)")
    df_filt = df_filt.query("(latitude < 40.9) | (latitude > 40.63)")
    print(df_filt.shape)
    print(df_filt.shape[0]/df.shape[0])