from dateutil.relativedelta import relativedelta
from datetime import datetime
from termcolor import colored
from config import get_time
import concurrent.futures
from config import path
from tqdm import tqdm
import pandas as pd
import numpy as np
import rawData
import config
import time


def iex(ticker_file, output, start, stop):
    """
    Time format: 2020-02-14
    """
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx", index_col="Symbol")
    xl = xl[~xl.index.duplicated()]
    # length = len(rawData.iex("SPY").df) * 0.2
    length = 10
    pt = config.Printer(output)
    print()
    kl = 1

    mean = xl["Shares"].mean()
    std = xl["Shares"].std()
    xl["Mean1-Sh"] = xl["Shares"] / mean
    xl["STD1-Sh"] = xl["Shares"] / std

    mean = xl["Cap"].mean()
    std = xl["Cap"].std()
    xl["Mean1-Cap"] = xl["Cap"] / mean
    xl["STD1-Cap"] = xl["Cap"] / std

    for tickers in xl.index[:]:
        data = rawData.iex(tickers, start, stop, time="day")
        if len(data) >= length:
            x = iex2(data, xl, tickers)
            pt.print(x)
            print(colored(f"{kl}) {len(x)} {tickers} - Done!", color="green"))
            kl += 1

        else:
            print(colored(f"{kl}) {tickers} - FAIL", color="red"))

    print(f"Start: {len(xl)}")
    print(f"Total: {kl - 1}")
    pt.save()


def td(ticker_file="ScanPyWDec", output="W-Feb"):
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx", index_col="Symbol")
    xl = xl[~xl.index.duplicated()]
    # length = len(rawData.td("SPY")) * 0.5
    length = 10
    pt = config.Printer(output)
    print()
    kl = 1

    mean = xl["Shares"].mean()
    std = xl["Shares"].std()
    xl["Mean1-Sh"] = xl["Shares"] / mean
    xl["STD1-Sh"] = xl["Shares"] / std

    mean = xl["Cap"].mean()
    std = xl["Cap"].std()
    xl["Mean1-Cap"] = xl["Cap"] / mean
    xl["STD1-Cap"] = xl["Cap"] / std
    for tickers in xl.index[:]:
        data = rawData.td(ticker=tickers)

        if len(data) >= length:
            data["datetime"] = data["datetime"] / 1000
            data.rename(columns={"datetime": "time"}, inplace=True)
            x = iex2(data, xl, tickers)
            pt.print(x)
            print(colored(f"{kl}) {len(x)} {tickers} - Done!", color="green"))
            kl += 1

        else:
            print(colored(f"{kl}) {tickers} - Fail!", color="red"))

    print(f"Start: {len(xl)}")
    print(f"Total: {kl}")
    pt.save()


def iex2(df, xl, i):
    final = pd.DataFrame()
    array = []
    for extra in xl:
        df[extra] = xl.loc[i, extra]

    vmean = df["volume"].mean()
    vstd = df["volume"].std()
    omean = df["open"].mean()
    ostd = df["open"].std()

    df["Volume-M"] = vmean
    df["Volume-STD"] = vstd
    df["Price-M"] = omean
    df["Price-STD"] = ostd

    df["Price1-M"] = df["open"] / omean
    df["Price1-STD"] = df["open"] / ostd

    # Rate
    for col in ["high", "low", "close"]:
        df[col] = (df[col] - df["open"]) / df["open"]

    df["open2"] = df["open"].pct_change()
    df = df.iloc[1:].reset_index(drop=True)

    # # Normal
    # for col in ["open", "high", "low", "close"]:
    #     df[col] = (df[col] - df[col].mean()) / df[col].std()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [(executor.submit(iex3, df, index, i)) for index in (df.index[5:])]
        for f in concurrent.futures.as_completed(results):
            df = f.result()
            if df is not None:
                array.append(df)
    final = final.append(array, ignore_index=True)

    return final


def iex3(xls, index, ticker):
    new = None
    # pre_index_high = xls.iloc[index - 1]["high"]
    # pre_index_low = xls.iloc[index - 1]["low"]
    # pre_index_close = xls.iloc[index - 1]["close"]
    # index_open = xls.iloc[index]["open"]

    # gap = (index_open - pre_index_close) / pre_index_close
    # pre_range = (pre_index_high - pre_index_low) / pre_index_low

    new = xls.loc[index]
    # pre_index_open = xls.iloc[index - 1]["open"]
    new["Symbol"] = ticker

    for i in range(1, 5 + 1):
        new = new.append(xls.iloc[index - i][["open", "high", "low", "close", "volume", "trade_count", "vwap"]].
                         rename({"open": f"PO{i}", "high": f"PH{i}",
                                 "low": f"PL{i}", "close": f"PC{i}",
                                 "volume": f"PV{i}", "trade_count": f"TC{i}",
                                 "vwap": f"vwap{i}"}))

    # new["PreReturn%"] = (pre_index_close - pre_index_open) / pre_index_open
    # new["PreRange"] = pre_range
    # new["Gap%"] = gap

    return new


def splicing(original="WSep-RAW", dataset="DS-H-IEX-SEP", output="Short-S-SEP-IEX"):
    dataset = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{dataset}.xlsx")
    xp = open_excel(original, identifiers=False)
    tickers = xp["Symbol"].drop_duplicates().reset_index(drop=True)
    pt = config.Printer(output)
    c = config.Counter()

    xp["o"] = np.nan
    xp["h"] = np.nan
    xp["l"] = np.nan
    xp["c"] = np.nan
    xp["v"] = np.nan
    xp["t"] = np.nan

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [(executor.submit(splicing2, xp.loc[ticker == xp["Symbol"]],
                                    pd.read_excel(dataset, ticker), ticker, c))
                   for ticker in tickers if ticker in dataset.sheet_names]

        for f in concurrent.futures.as_completed(results):
            pt.print(f.result())

    pt.save()


def splicing2(df, data, ticker, c):
    for i in df.index[:]:
        # ct = df["time"].loc[i] * 1000
        ct = df["time"].loc[i]
        dataset = data[data["t"] == (ct + 28800 + 1800) * 1000]

        if not dataset.empty:
            df["o"].loc[i] = dataset["o"].iloc[0]
            df["h"].loc[i] = dataset["h"].max()
            df["l"].loc[i] = dataset["l"].min()
            df["c"].loc[i] = dataset["c"].iloc[-1]
            df["v"].loc[i] = dataset["v"].sum()
            df["t"].loc[i] = dataset["t"].iloc[0]

        else:
            df.drop(index=i, inplace=True)

    df.reset_index(drop=True, inplace=True)
    print(f"{c.count()}) {ticker}")

    return df


def open_excel(file, identifiers=True):
    if type(file) is pd.DataFrame:
        return file

    df = pd.DataFrame()
    xl = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{file}.xlsx")
    xl = df.append([pd.read_excel(xl, sheets) for sheets in xl.sheet_names],
                   ignore_index=True).sort_values(by="time", ignore_index=True)

    if "Close2" not in xl and identifiers:
        xl["V-Result"] = 0
        xl["Close2"] = 0
        xl["Delta"] = ((xl["close"] - xl["c"]) / xl["c"])
        xl["V-Result"][xl["volume"] > 250000] = 1
        xl["Close2"][xl["Delta"] > 1.01] = 1

    return xl


def splitter(file1, file2, months=1):
    train = open_excel(file1)
    xp2 = open_excel(file2)

    start = xp2["time"].iloc[0]
    end = xp2["time"].iloc[-1]
    train = train.tail(900000)
    counter = 1

    start = datetime.strptime(get_time(start * 1000, 2), "%Y-%m-%d")
    date = start.replace(day=1)
    start = start.timestamp()

    while date.timestamp() < end:
        date += relativedelta(months=months)
        print(f"{counter}) {date}")

        test = xp2[(xp2["time"] <= date.timestamp()) &
                   (xp2["time"] > start)]
        start = date.timestamp()

        print(f"Train: {len(train)}")
        config.Printer(f"Train-31-{counter}", train)
        print(f"Test: {len(test)}")
        config.Printer(f"Test-31-{counter}", test)

        counter += 1
        train = train.append(test, ignore_index=True) \
            .tail(900000)


def excel_agg(file_names, max_files, output):
    pt = config.Printer(output)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = [(executor.submit(open_excel, file_names.format(i)))
                   for i in range(1, max_files + 1)]

        for f in concurrent.futures.as_completed(results):
            # df = f.result()
            # df = df[(df["M1"] == 1)]
            pt.print(f.result())

    pt.save()


def make_series(file):
    xl = open_excel(file, False)
    pr = config.Printer("Series-Test")

    tickers = xl["Symbol"] \
        .drop_duplicates() \
        .reset_index(drop=True)
    length = len(tickers)

    for ticker in tickers[:]:
        print(f"{length}) {ticker}")
        length = length - 1

        df = rawData.td(ticker)
        #
        #     if not df.empty:
        #         test = True
        #         while test:
        #             for i in xl[xl["Symbol"] == ticker].iterrows():
        #                 make_series2(df, i)
        #                 test = False
        #                 break

        if not df.empty:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = [(executor.submit(make_series2, df, i))
                           for i in xl[xl["Symbol"] == ticker].iterrows()]

                for f in concurrent.futures.as_completed(results):
                    arr = f.result()
                    if not arr.empty:
                        pr.print(arr)

    pr.save()


def make_series2(minute, day):
    day = day[1]
    minute = minute[(minute["datetime"] > day["time"]) &
                    (minute["datetime"] < day["time"] + 86400000)] \
        .reset_index(drop=True)

    if len(minute) != 78:
        return pd.DataFrame()

    minute["datetime"] = (minute["datetime"] - minute["datetime"][0]) / 300000
    minute["datetime"] = minute["datetime"].astype(int)

    df = pd.DataFrame()
    for row in range(77, -1, -1):
        day2 = day

        for i in minute.iterrows():
            if i[0] == row:
                series = i[1].rename({"open": f"open-F", "high": f"high-F",
                                      "low": f"low-F", "close": f"close-F",
                                      "volume": f"volume-F", "datetime": f"datetime-F"})
                day2 = day2.append(series)
                df = df.append(day2, ignore_index=True)
                break
            else:
                series = i[1].rename({"open": f"open{i[0]}", "high": f"high{i[0]}",
                                      "low": f"low{i[0]}", "close": f"close{i[0]}",
                                      "volume": f"volume{i[0]}", "datetime": f"datetime{i[0]}"})
                day2 = day2.append(series)

    df = df.fillna(0)
    return df


def add_spy(file, ticker="SPY", history=5):
    df = open_excel(file, False)
    spy = rawData.td(ticker, df["time"].min() * 1000, df["time"].max() * 1000)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    pr = config.Printer("SPY-Test")
    print(len(spy))
    arr = []

    for day in tqdm(days):
        df3 = df[df["time"] == day]

        for j in range(1, history + 1):
            today = spy.index[spy["datetime"] == day * 1000][0]
            df3[f"{ticker}-{j}-open"] = spy.iloc[today]["open"]
            df3[f"{ticker}-{j}-close"] = spy.iloc[today - j]["close"]
            df3[f"{ticker}-{j}-high"] = spy.iloc[today - j]["high"]
            df3[f"{ticker}-{j}-low"] = spy.iloc[today - j]["low"]
            df3[f"{ticker}-{j}-volume"] = spy.iloc[today - j]["volume"]

        arr.append(df3)

    df = pd.DataFrame().append(arr, ignore_index=True)

    pr.print(df)
    pr.save()
    # return df


def basic_data(file="ScsnPyCap", output="Cap-RAW"):
    df = pd.read_excel(rf"{config.TEST_FILE_PATH}\{file}.xlsx", index_col="Symbol")
    pt = config.Printer(output)
    length = 100
    kl = 1

    for tickers in df.index[:]:
        data = rawData.td(ticker=tickers)

        if len(data) >= length:
            data["datetime"] = data["datetime"] / 1000
            data.rename(columns={"datetime": "time"}, inplace=True)
            data["Symbol"] = tickers
            pt.print(data)
            print(colored(f"{kl}) {len(data)} {tickers} - Done!", color="green"))
            kl += 1
        else:
            print(colored(f"{kl}) {tickers} - Fail!", color="red"))

    print(f"Total: {kl}")
    pt.save()


def consolidation(file):
    df = open_excel(file, False)
    df.drop(columns=["Symbol"], inplace=True)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    df3 = pd.DataFrame(columns=df.columns)

    def mean(df2, bar2):
        arr = []
        for col in df2.columns:
            arr.append(df2[col].mean())
        bar2.update(1)
        return arr

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(mean, df.loc[df["time"] == day], bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                df3.loc[len(df3)] = f.result()

    # config.Printer("TDW-Mean", df3)
    return df3


def consolidation_ticker(file):
    df = open_excel(file, False)
    # df.drop(columns=["Symbol"], inplace=True)
    # df = df[(df["time"] <= 1625119200) & (df["time"] >= 1625119200 - 5259492*2)]
    tickers = df["Symbol"].drop_duplicates().reset_index(drop=True)
    df3 = pd.DataFrame(columns=df.columns)

    def mean(df2, ticker2, bar2):
        arr = []
        for col in df2.columns:
            if col == "Symbol":
                arr.append(ticker2)
            else:
                arr.append(df2[col].mean())
        bar2.update(1)
        return arr

    with tqdm(total=len(tickers)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(mean, df.loc[df["Symbol"] == ticker], ticker, bar)) for ticker in tickers]

            for f in concurrent.futures.as_completed(results):
                df3.loc[len(df3)] = f.result()

    # config.Printer("TDW-Ticker-Mean", df3)
    return df3


def mean_splice(original, means):
    df = open_excel(original, False)
    means = open_excel(means, False)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    pr = config.Printer("Mean_Splice")

    def combine(df2, df3, bar2):
        df2["Close3"] = df3["Close2"].mean()
        bar2.update()
        return df2

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(combine, df.loc[df["time"] == day],
                                        means.loc[means["time"] == day],
                                        bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                pr.print(f.result())

    pr.save()


def all_splice(original, means):
    df = open_excel(original, False)
    means = open_excel(means, False)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    pr = config.Printer("All-Splice2")

    def combine(df2, df3, bar2):
        for col in df3.columns:
            df2[f"{col}_Splice"] = df3[col].mean()
        bar2.update()
        return df2

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(combine, df.loc[df["time"] == day],
                                        means.loc[means["time"] == day],
                                        bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                pr.print(f.result())

    pr.save()


def all_splice_pivot(original, means):
    df = open_excel(original, False)
    df = df.drop(columns=["Close2", "volume", "high", "low", "close",
                          "Delta", "h", "l", "c", "v", "Mean1-Sh",
                          "STD1-Sh", "Mean1-Cap", "STD1-Cap", "Volume-M",
                          "Volume-STD", "Price-M", "Price-STD"])

    means = open_excel(means, False)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    arr = []

    def combine(df2, df3, bar2):
        df3 = df3.copy()
        for row in df2.iterrows():
            for key in row[1].keys():
                if key != "Symbol":
                    r = row[1]["Symbol"]
                    df3[f"{key}_{r}"] = row[1][key]
        bar2.update()
        return df3

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(combine,
                                        df.loc[df["time"] == day],
                                        means.loc[means["time"] == day],
                                        bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                arr.append(f.result())

    df = pd.DataFrame().append(arr, ignore_index=True)
    df2 = df.fillna(0)
    config.Printer("Features", df2)
    df2 = df.dropna()
    config.Printer("Features-NoNA", df2)


def add_prev(original, output="Prev-Test", num_prev=5):
    df = open_excel(original, False)
    target_col = ["h", "l", "c", "v"]
    tickers = df["Symbol"].drop_duplicates().reset_index(drop=True)
    arr = []

    def combine(xls, index, bar2):
        new = xls.loc[index]
        for i in range(1, num_prev + 1):
            new2 = xls.iloc[index - i][target_col]
            for col in target_col:
                new2 = new2.rename({col: f"{col}{i}"})
            new = new.append(new2)

        bar2.update()
        return new

    results = []
    with tqdm(total=len(df)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:

            for ticker in tickers:
                df2 = df[df["Symbol"] == ticker].reset_index(drop=True).sort_values(by="time", ignore_index=True)
                for index in df2.index[num_prev:]:
                    results.append((executor.submit(combine, df2, index, bar)))

            for f in concurrent.futures.as_completed(results):
                arr.append(f.result())

    df = pd.DataFrame().append(arr, ignore_index=True).sort_values(by="time", ignore_index=True)
    config.Printer(output, df)
    # with tqdm(total=len(df)) as bar:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         results = [(executor.submit(combine,
    #                                     df[df["Symbol"] == ticker].reset_index(drop=True),
    #                                     index,
    #                                     bar))
    #                    for ticker in tickers
    #                    for index in df[df["Symbol"] == ticker].reset_index(drop=True).index[num_prev:]]

    # results = []
    # with tqdm(total=len(df)) as bar:
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #
    #         for ticker in tickers:
    #             df2 = df[df["Symbol"] == ticker].reset_index(drop=True).sort_values(by="time", ignore_index=True)
    #             for index in df2.index[num_prev:]:
    #                 results.append((executor.submit(combine, df2, index, bar)))
    #
    #         for f in concurrent.futures.as_completed(results):
    #             arr.append(f.result())


def x_splice(original, means, output):
    df = open_excel(original, False)
    means = open_excel(means, False)
    days = df["time"].drop_duplicates().reset_index(drop=True)
    targets = ["CL2-1", "CL2-2", "CL2-3", "CL2-4", "CL2-5", "CL2-6"]
    pr = config.Printer(output)

    def combine(df2, df3, bar2):
        for col in df3.columns:
            if col in targets:
                df2[f"{col}_Splice"] = df3[col].mean()
        bar2.update()
        return df2

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(combine, df.loc[df["time"] == day],
                                        means.loc[means["time"] == day],
                                        bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                pr.print(f.result())

    pr.save()


def ticker_splice(original, splice, output):
    """
    Use for Scraper data
    """
    df = open_excel(original, False)
    splice = pd.read_excel(rf"{config.TEST_FILE_PATH}\{splice}.xlsx")
    days = df["Symbol"].drop_duplicates().reset_index(drop=True)
    pr = config.Printer(output)

    def combine(df2, df3, bar2):
        for col in df3.columns:
            if col != "Ticker":
                df2[col] = df3[col].mean()
        bar2.update()
        return df2

    with tqdm(total=len(days)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(combine, df.loc[df["Symbol"] == day],
                                        splice.loc[splice["Ticker"] == day],
                                        bar)) for day in days]

            for f in concurrent.futures.as_completed(results):
                pr.print(f.result())

    pr.save()


def fct(ticker_file, time_file, out, history=5):
    tickers = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx")
    times = pd.read_excel(rf"{config.TEST_FILE_PATH}\{time_file}.xlsx")
    spy = rawData.td("SPY", times["time"].min() * 1000, times["time"].max() * 1000)
    arr = []

    for ticker in tqdm(tickers["Symbol"][:]):
        data = rawData.td(ticker, times["time"].min() * 1000, times["time"].max() * 1000)

        if len(spy) == len(data):
            arr2 = []
            for day in data["datetime"][history:]:
                df2 = pd.DataFrame({"PL": [0]})

                for j in range(1, history + 1):
                    today = data.index[data["datetime"] == day][0]
                    df2[f"{ticker}-{j}-open"] = data.iloc[today]["open"]
                    df2[f"{ticker}-{j}-close"] = data.iloc[today - j]["close"]
                    df2[f"{ticker}-{j}-high"] = data.iloc[today - j]["high"]
                    df2[f"{ticker}-{j}-low"] = data.iloc[today - j]["low"]
                    df2[f"{ticker}-{j}-volume"] = data.iloc[today - j]["volume"]

                df2.drop(columns=["PL"], inplace=True)
                arr2.append(df2)

            arr.append(pd.DataFrame().append(arr2, ignore_index=True))

    df = pd.concat(arr, axis=1)
    df["time"] = spy["datetime"][history:].reset_index(drop=True)
    df.to_pickle(f"{config.TEST_FILE_PATH}/{out}")
    # config.Printer(out, df)


def fct_combine(og, fct):
    ff = pd.read_pickle(path(fct)).sort_values(by="time", ignore_index=True).tail(800)
    df = pd.read_pickle(path(og)).sort_values(by="time", ignore_index=True)
    arr = []

    ff["time2"] = ff.pop("time") / 1000

    with tqdm(total=len(ff)) as bar:
        for day in ff["time2"]:
            # df2 = df[df["time"] == day]
            # df = df.tail(len(df) - len(df2))
            #
            # df3 = pd.DataFrame(
            #     [ff.iloc[0].to_numpy()],
            #     index=df2.index,
            #     columns=ff.keys())

            df2 = pd.concat([
                df[df["time"] == day],
                pd.DataFrame(
                    [ff.iloc[0].to_numpy()],
                    index=df[df["time"] == day].index,
                    columns=ff.keys())],
                axis=1)

            ff = ff.iloc[1:, :]
            arr.append(df2)
            bar.update()
    ff = 0
    df = 0
    df = pd.concat(arr, ignore_index=True)

    return df


def jar(file):
    pd.read_excel(path(file + ".xlsx")).to_pickle(path(file))


def deltas(df):
    cols = [["high", "low", "open"]]
    cols += [[f"PC{i}", f"PH{i}", f"PL{i}", f"PO{i}"] for i in range(1, 6)]

    for i in range(len(cols)):
        if len(cols[i]) == 3:
            df[cols[i][0]] = (df[cols[i][0]] - df[cols[i][2]]) / df[cols[i][2]]
            df[cols[i][1]] = (df[cols[i][1]] - df[cols[i][2]]) / df[cols[i][2]]
            df[cols[i][2]] = (df[cols[i][2]] - df[cols[i + 1][0]]) / df[cols[i + 1][0]]
        else:
            for j in range(3):
                df[cols[i][j]] = (df[cols[i][j]] - df[cols[i][3]]) / df[cols[i][3]]
            if i != (len(cols) - 1):
                df[cols[i][3]] = (df[cols[i][3]] - df[cols[i + 1][0]]) / df[cols[i + 1][0]]

    return df


def round_partial(value, resolution=86400):
    return (np.floor(value / resolution) * resolution).astype(int)


def collate(out):
    df = pd.DataFrame()
    all = []
    arr = []

    for i in range(1, 13):
        arr.append(f"Min-2021-{i}")

    for i in range(1, 3):
        arr.append(f"Min-2022-{i}")

    for file in arr:
        print("\n", file)
        xl = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{file}.xlsx")

        for ticker in tqdm(xl.sheet_names):
            xls = pd.read_excel(xl, ticker)
            xls["Symbol"] = ticker
            all.append(xls)

    print("\n", "Combining")
    df = df.append(all, ignore_index=True)
    print("Dropping Duplicates")
    df = df.drop_duplicates(ignore_index=True)
    print("Printing")
    df.to_pickle(f"{config.TEST_FILE_PATH}/{out}")


def get_mean(out, data="Min-Feb", file="Test2"):
    df = pd.read_pickle(path(file)).sort_values(by="time")
    dat = pd.read_pickle(path(data))
    tkr = df["Symbol"].unique()
    df3 = pd.DataFrame()
    arr = []

    def comp(df2, dat2, bar2):
        df2 = df2.reset_index(drop=True)
        means = []
        stds = []
        for i in range(len(df2)):
            t = df2["time"].loc[i]
            try:
                dat3 = dat2[(dat2["time"] > t) & (dat2["time"] < t+86400)]
                dat3 = dat3[["open", "close"]].unstack().reset_index(drop=True)
                m = dat3.mean()
                s = dat3.std()
            except:
                m = s = None

            means.append(m)
            stds.append(s)
            bar2.update(1)

        df2["Min-Mean"] = means
        df2["Min-STD"] = stds
        return df2

    with tqdm(total=len(df)) as bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [(executor.submit(comp, df[df["Symbol"] == t],
                                        dat[dat["Symbol"] == t], bar))
                       for t in tkr]

            for f in concurrent.futures.as_completed(results):
                arr.append(f.result())

    print()
    print("Combining")
    df3 = df3.append(arr, ignore_index=True)
    print("Dropping NaNs")
    df3 = df3.dropna().reset_index(drop=True)
    print("Printing")
    df3.to_pickle(f"{config.TEST_FILE_PATH}/{out}")
