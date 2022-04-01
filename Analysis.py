import iexDataProcessing
import rawData
from iexDataProcessing import open_excel
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import concurrent.futures
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
import plotly.express as px
import config
from scipy import stats
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, Normalizer


def pca1(df, dim=3):
    df.pop("time")
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
    final = df.pop("Close2")

    pca = PCA(n_components=dim)
    pca2 = pca.fit_transform(df)

    print(df.columns)
    print(pca.get_covariance())
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_ratio_.cumsum())

    if dim == 2:
        plt.scatter(pca2[:, 0], pca2[:, 1], c=final)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.show()

    if dim == 3:
        fig = plt.figure(figsize=(21, 9))
        ax = fig.add_subplot(projection="3d")
        ax.scatter(pca2[:, 0], pca2[:, 1], pca2[:, 2], c=final)
        ax.set_zlim([-0.45, 0.5])
        ax.set_ylim([-2.5, 5])
        # ax.set_xlim([-2, 2])
        plt.show()


def price_flow_all(file):
    df = open_excel(file, False)
    df = iexDataProcessing.consolidation(df).sort_values(by="time", ignore_index=True)
    days = df["time"].drop_duplicates().reset_index(drop=True)

    x = df["v"].min
    df["v"] = df["v"] - x
    df["v"] = df["v"] / 100000

    arr1 = []
    arr2 = []
    arr3 = []
    arr4 = []

    x_axis = "time"
    y_axis = "Delta"
    y2_axis = "v"
    y3_axis = "Delta"

    for i in tqdm(days):
        arr1.append(df[df["time"] == i][y_axis].mean())
        arr2.append(df[df["time"] == i][x_axis].mean())
        arr3.append(df[df["time"] == i][y2_axis].mean())
        arr4.append(df[df["time"] == i][y3_axis].mean())
    df = pd.DataFrame({y_axis: arr1, x_axis: arr2, y2_axis: arr3, y3_axis: arr4})

    # plt.scatter(1625119200, 0, s=10)
    plt.plot(df[x_axis], df[y_axis], color='blue', marker='o')
    plt.plot(df[x_axis], df[y2_axis], color='red', marker='o')
    # plt.plot(df[x_axis], df[y3_axis], color='black', marker='o')
    plt.grid(True)

    plt.show()


def price_flow_single(ticker="UVXY", start=1514790000000, stop=1628438432000):
    df = rawData.td(ticker, start, stop)
    days = df["datetime"].drop_duplicates().reset_index(drop=True)
    df["delta"] = (df["close"] - df["open"]) / df["open"]
    arr = []

    for i in tqdm(days):
        arr.append(df[df["datetime"] == i]["delta"].mean())

    plt.scatter(x=range(0, len(arr)), y=arr, c="blue")
    plt.show()

# =IF(OR([@Delta]>0.01, [@Delta]<-0.01), 1, 0)


def set_builder(file, final_len=300):
    df = open_excel(file, False)
    df = df.tail(int(len(df) / 2))
    print(df["time"].min())
    print("Start", "\n")
    # def avg_delta(dfx, ticker=None):
    #     days = df["time"].drop_duplicates()
    #     initial = 0
    #     delta = 0
    #
    #     for day in days:
    #         if initial == 0:
    #             initial = dfx[dfx["time"] == day]["Close2"].mean()
    #         else:
    #             final = dfx[dfx["time"] == day]["Close2"].mean()
    #             delta += (final-initial)/initial
    #             initial = final
    #
    #     if ticker is None:
    #         return delta / (len(days) - 1)
    #     return [ticker, delta/(len(days)-1)]

    def avg_delta(dfx, ticker=None):
        days = df["time"].drop_duplicates()
        initial = 0
        delta = 0

        for day in days:
            if initial == 0:
                initial = dfx[dfx["time"] == day]["Close2"].mean()
            else:
                final = dfx[dfx["time"] == day]["Close2"].mean()
                delta += abs((final-initial)/initial)
                initial = final

        if ticker is None:
            return delta / (len(days) - 1)
        return [ticker, delta/(len(days)-1)]

    length = len(df["Symbol"].drop_duplicates())
    print(f"{length} - {avg_delta(df)}")

    for _ in range(len(df["Symbol"].drop_duplicates())-final_len):
        matrix = pd.DataFrame(columns=["Symbol", "Delta"])
        # for ticker in df["Symbol"].drop_duplicates():
        #     matrix.loc[len(matrix)] = [ticker, avg_delta(df[df["Symbol"] != ticker])]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(avg_delta, df[df["Symbol"] != ticker], ticker))
                       for ticker in df["Symbol"].drop_duplicates()]

            for f in concurrent.futures.as_completed(results):
                matrix.loc[len(matrix)] = f.result()

        id = matrix["Delta"].idxmin()
        df = df[df["Symbol"] != matrix["Symbol"][id]]
        length = len(df["Symbol"].drop_duplicates())
        print(f"{length} - {avg_delta(df)}")

    for i in df["Symbol"].drop_duplicates():
        print(i)


def simple_graph(file):
    df = open_excel(file, False)
    df = df[df["Symbol"] == "SNDL"]
    plt.scatter(df["time"], df["high"])
    plt.show()
