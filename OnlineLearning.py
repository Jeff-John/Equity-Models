from OnlineHelperFuncts import data_prep, mean_slice, path_short, path_long, algo_basic_mean
from matplotlib import pyplot as plt
import pandas as pd
import skmultiflow
import config
import river


def multiflow_reg(file, dat="Min-Feb", target="Min-Mean", path="long", const=0.02, top_x=5):
    dat = pd.read_pickle(config.path(dat))
    df = data_prep(file)
    df = mean_slice(df, target)
    df = df.reset_index(drop=True).sort_values(by="time", ignore_index=True)
    bank = {}

    drops = ["volume", "high", "low", "close", "Min-STD", "Symbol", "trade_count", "vwap"]
    days = df["time"].drop_duplicates().reset_index(drop=True).sort_values(ignore_index=True)
    df = df.fillna(0)
    df2 = df.copy()
    print(len(days))
    graph_pl1 = []

    y = df.pop(target)
    df = df.drop(columns=drops)
    t = config.Counter()
    pred = []
    arr = []

    ht = skmultiflow.trees.HoeffdingAdaptiveTreeRegressor(grace_period=2,
                                                          split_confidence=0.000000000000000000000000000000000000000001,
                                                          tie_threshold=0.000000000000000000000000000000000000000000005)
    stream1 = skmultiflow.data.DataStream(df, y=pd.DataFrame(y))
    stream2 = skmultiflow.data.DataStream(df, y=pd.DataFrame(y))
    counter = 0

    for day in days[:]:
        pos = 0
        metric = river.metrics.Precision()
        length = range(len(df[df["time"] == day]))
        pl = pd.DataFrame(columns=["Y", "p", "n", "s"])

        for _ in length:
            X, Y = stream1.next_sample()
            p = ht.predict(X)[0]
            pl.loc[len(pl)] = [Y[0], p, counter, df2["Symbol"].loc[counter]]
            counter += 1

        pl = pl.sort_values(by="p", ignore_index=True).tail(top_x).reset_index(drop=True)
        for i in range(len(pl)):
            if path == "short":
                Y, p = path_short(pl, i, const)
            if path == "long":
                Y, p = path_long(pl, i, const)
            else:
                print("PATH NOT FOUND")

            metric.update(Y, p)
            pred.append(pl["p"].loc[i])
            pos = pl["p"].mean()

        bank = algo_basic_mean(df2, pl, bank, top_x, dat, day)
        print(bank["BM"])
        graph_pl1.append(bank["BM"])

        for _ in length:
            X, Y = stream2.next_sample()
            ht.partial_fit(X, Y)

        if pos > 0 or metric.get() > 0:
            arr.append(metric.get())
        print(f"{t.count()}) {metric.get()}  {pos} - {day}")
        print()

    plt.plot(range(0, len(graph_pl1)), graph_pl1, color='blue', marker='o')
    plt.show()
    mean = sum(arr) / len(arr)
    print("Mean: ", mean)
