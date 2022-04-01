import random
import sys

import numpy as np
import pandas as pd
import river
import skmultiflow
from matplotlib import pyplot as plt
from tqdm import tqdm
import config
from itertools import chain, combinations

import rawData

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


class Dropper:

    def __init__(self, train, test, reduce=25):
        train["low"] = -train["low"]
        test["low"] = -test["low"]
        self.df = train.copy()
        self.test = test.copy()
        self.df_og = train.copy()
        self.reduce = reduce
        self.drop = ""

        self.run_model()

    def run_model(self):
        self.pick_direction()
        print(self.test["result"].mean())
        self.drop_low_tickers()
        og_length = len(self.df)
        print(len(self.df), len(self.test))
        print(self.test["result"].mean())
        print()

        while len(self.df) > 0.05 * og_length and len(self.test) > 1:
            best = self.find_best_drop()
            self.df = self.df[(self.df[best[1]] < best[2]) | (self.df[best[1]] > best[3])]
            self.test = self.test[(self.test[best[1]] < best[2]) | (self.test[best[1]] > best[3])]
            print(best[0], best[1], len(self.test), self.test["result"].mean())

        print(self.test[["result", self.drop]])

    def pick_direction(self):
        group = self.df.groupby("Symbol")
        result = group.mean()["high"].sort_values()
        high = result.tail(self.reduce).mean()
        group = self.df.groupby("Symbol")
        result = group.mean()["low"].sort_values()
        low = result.head(self.reduce).mean()
        print(self.df["high"].mean() + self.df["low"].mean(), end=" ")

        if high > abs(low):
            result = "high"
            self.drop = "low"
            print("Long")
        else:
            result = "low"
            self.drop = "high"
            print("Short")

        self.df = self.df.rename(columns={result: "result"})
        self.test = self.test.rename(columns={result: "result"})
        self.df_og = self.df_og.rename(columns={result: "result"})

    def drop_low_tickers_mean(self):
        group = self.df.groupby("Symbol")
        result = group.mean()["result"].sort_values()
        mean = result.mean()
        final = result[result > mean].index

        self.df = self.df[self.df['Symbol'].isin(final)].reset_index(drop=True)
        self.test = self.test[self.test['Symbol'].isin(final)].reset_index(drop=True)

    def drop_low_tickers(self):
        group = self.df.groupby("Symbol")
        result = group.mean()["result"].sort_values()
        final = result.tail(50).index

        self.df = self.df[self.df['Symbol'].isin(final)].reset_index(drop=True)
        self.test = self.test[self.test['Symbol'].isin(final)].reset_index(drop=True)
        print(self.df["result"].mean())

        group = self.df.groupby("Symbol")
        result = group.mean()[self.drop].sort_values()
        final = result.tail(25).index

        self.df = self.df[self.df['Symbol'].isin(final)].reset_index(drop=True)
        self.test = self.test[self.test['Symbol'].isin(final)].reset_index(drop=True)
        print(self.df["result"].mean())

    def find_best_drop(self):
        best = [0, 0, 0, 0]

        for col in self.df.columns:
            if col not in ["time", "result", self.drop]:
                sorted_df = self.df[["time", "result", col]].sort_values(by=col, ignore_index=True)
                sorted_df = sorted_df.drop_duplicates(subset=[col])

                current = self.find_continuous_col(sorted_df, col)
                if current[0] > best[0]:
                    best = current

        return best

    @staticmethod
    def find_continuous_col(sorted_df, col):
        sorted_df = sorted_df.reset_index(drop=True)

        times = sorted_df["time"].unique()[::-1]
        groped = sorted_df.groupby(["time"])
        mn = groped.mean()["result"]
        means = dict(zip(times, np.array([mn.loc[t] for t in times])))
        longest = 0
        current = 0
        start = 0
        end = 0

        for i in range(len(sorted_df)):
            cv = sorted_df["result"].iloc[i]
            ct = sorted_df["time"].iloc[i]

            if cv < means[ct]:
                current += 1
            else:
                if longest < current:
                    longest = current
                    end = sorted_df[col].iloc[i - 1]
                    start = sorted_df[col].iloc[i - current]
                current = 0

        return [longest, col, start, end]





