import pickle

import requests
import config
import pandas as pd
from termcolor import colored
import iexDataProcessing
import rawData
from config import Printer
import time


def make_hourly(ticker_file="ScanPyWSep", output="DS-H-IEX-SEP", start='2018-01-01 00:00:00', stop='2021-09-12 00:00:00'):
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx")
    pr = config.Printer(output)
    c = 1
    session = requests.session()
    print()
    fails = 0

    for ticker in xl["Symbol"][:]:
        try:
            df = rawData.iex_v2(ticker, time="1Hour", start=start, stop=stop, session=session)
        except:
            df = pd.DataFrame()

        if len(df) < 100:
            print(colored(f"{c}) {ticker} - FAIL", color="red"))
            fails += 1
        else:
            print(colored(f"{c}) {ticker} - {len(df)}", color="green"))
            pr.print(df, ticker)
        c += 1

    print(f"\nTotal Fails: {fails}")
    pr.save()


def make_hourly_td(ticker_file="ScanPyWSep", output="DS-H60-TD-SEP", start=1570487908000, stop=1633559881000):
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx")
    pr = config.Printer(output)
    session = requests.session()
    c = config.Counter()
    fails = 0
    print()

    for ticker in xl["Symbol"][:]:
        try:
            df = rawData.td(ticker, start=start, end=stop, session=session)
            df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                               "volume": "v", "datetime": "t"}, inplace=True)
        except:
            df = pd.DataFrame()

        if len(df) < 100:
            print(colored(f"{c.count()}) {ticker} - FAIL", color="red"))
            fails += 1
        else:
            print(colored(f"{c.count()}) {ticker} - {len(df)}", color="green"))
            pr.print(df, ticker)

    print(f"\nTotal Fails: {fails}")
    pr.save()


def make_hourly_new(ticker_file, output, start, stop):
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx")
    pr = config.Printer(output)
    fails = 0
    c = 1
    print()

    for ticker in xl["Symbol"][:]:
        if c%40 == 0:
            time.sleep(40)
        try:
            df = rawData.iex(ticker, start, stop)
        except:
            try:
                time.sleep(20)
                df = rawData.iex(ticker, start, stop)
            except:
                df = pd.DataFrame()

        if len(df) < 10:
            print(colored(f"{c}) {ticker} - FAIL", color="red"))
            fails += 1
        else:
            print(colored(f"{c}) {ticker} - {len(df)}", color="green"))
            pr.print(df, ticker)
        c += 1

    print(f"\nTotal Fails: {fails}")
    pr.save()


def update_hourly(original="6YearHourlyDataset-edit"):
    xl = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{original}.xlsx")
    pr = Printer("6YearHourlyDataset-4-26-Polygon")
    today = 1619479640000
    day = 86400000
    fails = 0
    c = 1

    for ticker in xl.sheet_names:
        xls = pd.read_excel(xl, ticker)
        date = xls["t"].iloc[-1]
        pr.print(xls, ticker)

        try:
            df = rawData.polygon(ticker, (date + day / 2).__floor__(), today)
            # df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
            #                "volume": "v", "datetime": "t"}, inplace=True)
            df = df[['o', 'h', 'l', 'c', 'v', "t"]]
        except:
            df = pd.DataFrame()

        if len(df) < 10:
            try:
                df = rawData.polygon(ticker, (date + day / 2).__floor__(), today)
                # df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                #                    "volume": "v", "datetime": "t"}, inplace=True)
                df = df[['o', 'h', 'l', 'c', 'v', "t"]]

            except:
                df = pd.DataFrame()

        if len(df) < 10:
            print(f"{c}) {ticker} - FAIL")
            fails += 1
        else:
            print(f"{c}) {ticker} - {len(df)}")
            pr.print(df, ticker)
        c += 1

    print(f"Total Fails: {fails}")
    pr.save()


def update_hourly2(original="6YearHourlyDataset-edit"):
    xl = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{original}.xlsx")
    pr = Printer("6YearHourlyDataset-4-31")
    session = requests.session()
    today = 1620084971000
    fails = 0
    c = 1

    for ticker in xl.sheet_names:
        xls = pd.read_excel(xl, ticker)
        # pr.print(xls, ticker)

        try:
            time.sleep(1.25)
            df = rawData.td(session, ticker, 0, today)
            df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                               "volume": "v", "datetime": "t"}, inplace=True)
        except:
            df = pd.DataFrame()

        if len(df) < 10:
            try:
                time.sleep(5)
                df = rawData.td(session, ticker, 0, today)
                df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                                   "volume": "v", "datetime": "t"}, inplace=True)

            except:
                df = pd.DataFrame()

        if len(df) < 10:
            print(f"{c}) {ticker} - FAIL")
            pr.print(df, ticker)
            fails += 1
        else:
            print(f"{c}) {ticker} - {len(df)}")
            df2 = xls[xls["t"] < df["t"].iloc[0]]
            pr.print(df2, ticker)
            pr.print(df, ticker)

        c += 1

    print(f"Total Fails: {fails}")
    pr.save()


def update_hourly3(original="6YearHourlyDataset-edit"):
    xl = pd.ExcelFile(rf"{config.TEST_FILE_PATH}\{original}.xlsx")
    session = requests.session()
    pr = Printer("6YearHourlyDataset-4-31")
    today = 1620084971000
    day = 86400000
    fails = 0
    c = 1

    for ticker in xl.sheet_names:
        xls = pd.read_excel(xl, ticker)
        date = xls["t"].iloc[-1]
        pr.print(xls, ticker)

        try:
            df = rawData.td(session, ticker, (date + day / 2).__floor__(), today)
            df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                               "volume": "v", "datetime": "t"}, inplace=True)
        except:
            df = pd.DataFrame()

        if len(df) < 10:
            try:
                time.sleep(5)
                df = rawData.td(session, ticker, (date + day / 2).__floor__(), today)
                df.rename(columns={"open": "o", "high": "h", "low": "l", "close": "c",
                                   "volume": "v", "datetime": "t"}, inplace=True)
            except:
                df = pd.DataFrame()

        if len(df) < 10:
            print(f"{c}) {ticker} - FAIL")
            fails += 1
        else:
            print(f"{c}) {ticker} - {len(df)}")
            pr.print(df, ticker)
        c += 1

    print(f"Total Fails: {fails}")
    pr.save()





