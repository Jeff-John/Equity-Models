import pandas as pd
import rawData
import config
import random


def data_prep(file):
    df = pd.read_pickle(config.path(file)).sort_values(by="time", ignore_index=True)
    df = df.tail(int(len(df) / 3)).reset_index(drop=True)
    df["Min-Mean"] = (df["Min-Mean"] - df["open"]) / df["open"]
    return df


def mean_slice(df, target):
    time_split = (((df["time"].max() + df["time"].min()) / 2) + df["time"].max()) / 2

    group = df[df["time"] > time_split]
    group = group.groupby("Symbol").mean().sort_values(by="Min-STD")
    group = group.head(200).index
    df = df[df['Symbol'].isin(group)].reset_index(drop=True)

    group = df[df["time"] > time_split]

    group = group.groupby("Symbol").mean().sort_values(by=target)
    group = group.tail(40).index
    df = df[df['Symbol'].isin(group)].reset_index(drop=True)

    return df


def path_short(pl, i, const):
    Y = p = 0

    if (pl["Y"].loc[i]) < const:
        Y = 1
    if (pl["p"].loc[i]) < const:
        p = 1

    return Y, p


def path_long(pl, i, const):
    Y = p = 0

    if (pl["Y"].loc[i]) > const:
        Y = 1
    if (pl["p"].loc[i]) > const:
        p = 1

    return Y, p


def algo_low_high(df, pl, bank):
    if "LH" not in bank:
        bank["LH"] = 1

    for i in range(len(pl)):
        low = df["low"].loc[pl["n"].loc[i]]
        high = df["high"].loc[pl["n"].loc[i]]
        close = df["close"].loc[pl["n"].loc[i]]

        if low < -0.04:
            bank["LH"] -= -0.04 * bank["LH"]
        else:
            bank["LH"] -= close * bank["LH"]

    return bank


def algo_low_high_random(df, pl, bank, const, top_x):
    if "LH" not in bank:
        bank["LH"] = 1

    i = random.randrange(top_x)
    low = df["low"].loc[pl["n"].loc[i]]
    high = df["high"].loc[pl["n"].loc[i]]
    close = df["close"].loc[pl["n"].loc[i]]
    print()

    if low < const:
        bank["LH"] -= const * bank["LH"]
    else:
        bank["LH"] -= close * bank["LH"]

    # bank["LH"] -= close * bank["LH"]

    return bank


def algo_limit_buy(df, pl, bank, top_x, dat):
    if "LH" not in bank:
        bank["LH"] = 1

    high = -1
    for i in range(top_x):
        stop = False
        symbol = df["Symbol"].loc[pl["n"].loc[i]]
        t = df["time"].loc[pl["n"].loc[i]]
        dat2 = dat[dat["Symbol"] == symbol]
        dat2 = dat2[(dat2["time"] > t) & (dat2["time"] < t + 86400)].reset_index(drop=True)
        if len(dat2) < 10:
            t = config.get_time(t, only_date=True)
            dat2 = rawData.iex(symbol, t, t, time="min")
            if len(dat2) < 10:
                print("Fail")
                continue

        o = dat2["open"].loc[0]
        final = None
        b = None
        c = 1

        for i in dat2["close"][1:]:
            c += 1
            if (i - o) / o > 0.02:
                stop = True
                print("--BUY--", (i - o) / o)
                b = i
                break

        if b:
            for i in dat2["close"][c:]:
                if (i - b) / b < -0.02:
                    final = (i - b) / b * bank["LH"]
                    bank["LH"] -= final
                    print("--SELL--")
                    break

                if (i - b) / b > 0.04:
                    final = (i - b) / b * bank["LH"]
                    bank["LH"] -= final
                    print("--LIMIT--")
                    break

            if final is None:
                o = dat2["open"].loc[len(dat2) - 1]
                final = (o - b) / b * bank["LH"]
                bank["LH"] -= final
                print("--CLOSE--", (o - b) / b)

        if stop:
            break

    return bank


def algo_basic_mean(df, pl, bank, top_x, dat, t):
    if "BM" not in bank:
        bank["BM"] = 1

    symbols = [pl["s"].iloc[i] for i in range(top_x)]
    t2 = config.get_time(t, only_date=True)

    dat = dat[(dat["Symbol"].isin(symbols)) &
              (dat["time"] > t) &
              (dat["time"] < t + 86400)] \
        .reset_index(drop=True)

    for ticker in symbols:
        if ticker not in dat["Symbol"]:
            dat2 = rawData.iex(ticker, t2, t2, time="min")
            dat2["Symbol"] = ticker
            if len(dat2) < 10:
                print("Fail")
            else:
                dat = dat.append(dat2, ignore_index=True)

    dat = dat.sort_values(by=['time', 'Symbol'], ignore_index=True)

    current = 1
    fop = None
    time = 0
    while current < len(dat):
        cl = dat["close"].iloc[current]
        op = df[(df["Symbol"] == dat["Symbol"].iloc[current]) &
                        (df["time"] == t)]
        op = op["open"].iloc[0]

        m = pl["p"][pl["s"] == dat["Symbol"].iloc[current]].iloc[0]
        if (cl-op)/op < m*0.70:
            dat = dat[(dat["Symbol"] == dat["Symbol"].iloc[current]) &
                      (dat["time"] >= dat["time"].iloc[current])]
            current = 0
            fop = op
            print("~~BUY~~")
            time = dat["time"].iloc[current]
            break
        current += 1

    m = pl["p"][pl["s"] == dat["Symbol"].iloc[current]].iloc[0]

    while current < len(dat):
        cl = dat["close"].iloc[current]
        op = dat["close"].iloc[0]

        if current == len(dat)-1:
            bank["BM"] += ((cl - op) / op) * bank["BM"]
            print("~~LIMIT~~")
            print(time)
            time -= dat["time"].iloc[current]
            print(-time/60)
            break

        if (cl - fop) / fop > m*1.10:
            bank["BM"] += ((cl - op) / op) * bank["BM"]
            print("~~SELL~~")
            print(time)
            time -= dat["time"].iloc[current]
            print(-time/60)
            break

        current += 1

    return bank
