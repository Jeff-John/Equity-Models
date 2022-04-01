import numpy as np
import matplotlib.pyplot as plt
from iexDataProcessing import open_excel
import pandas as pd


class Windowed:

    @staticmethod
    def calculate_weighted_moving_average(arr, window, alpha=0.333):
        # EMA[1] = SMA[1]
        ma = [np.sum(arr[:window]) / window]

        for idx in range(1, arr.shape[0]):
            if idx + window - 1 > arr.shape[0] - 1:
                ma.append(None)
                continue

            # (1-a) * MA[t-1] + a * S[t + window - 1]
            ma.append((1 - alpha) * ma[-1] + alpha * arr[idx + window - 1])
        return np.array(ma, dtype=np.float64)

    @staticmethod
    def transform_to_sliding_windows(arr, moving_average, window):
        result = None
        for idx in range(moving_average.shape[0]):
            if len(arr[idx: idx + window]) < window:
                break

            window_values = arr[idx: idx + window].copy()
            window_values /= moving_average[idx]
            result = window_values if result is None else np.vstack([result, window_values])

        return result

    @staticmethod
    def discard_outliers(arr, threshold=1.5):
        q1, q3 = np.quantile(arr, [0.25, 0.75])
        l_lim = q1 - threshold * (q3 - q1)
        g_lim = q3 + threshold * (q3 - q1)
        greater_than = arr > g_lim
        less_than = arr < l_lim

        valid = np.invert(
            np.any(
                np.logical_or(greater_than, less_than), axis=1
            )
        )

        return arr[valid], {
            'valid': valid,
            'has_less': np.any(less_than),
            'has_greater': np.any(greater_than),
            'lower_limit': l_lim,
            'upper_limit': g_lim
        }

    @staticmethod
    def normalize(arr, minimum=None, maximum=None):
        _min = minimum if minimum else np.min(arr)
        _max = maximum if maximum else np.max(arr)
        return (arr - _min) / (_max - _min), _min, _max

    @staticmethod
    def denormalize(arr, minimum, maximum):
        return arr * (maximum - minimum) + minimum

    @staticmethod
    def detransform(arr, ma):
        _arr = arr[:, -1]
        _first = arr[0, :-1]

        means = ma[:len(_arr)]
        _arr_m = _arr * means
        _first_m = _first * ma[0]
        return np.concatenate([_first_m, _arr_m])

    @staticmethod
    def discard_limits(s, k, w):
        to_discard = k - (w - 1)

        if to_discard <= 0:
            return s, None

        print(f"\nTo discard: {to_discard}")
        return s[to_discard:], s[:to_discard]

    @staticmethod
    def calc_level_adjustment(s, ma, w, i):

        adjustments = []

        for j in range(i, i + w):
            adjustments.append(np.square(s[j - 1] - ma[i - 1]))

        return sum(adjustments) / w

    @staticmethod
    def calc_level_adjustments_ma(s, ma, w):

        phi = len(s) - w + 1
        adjustments = []

        for i in range(1, phi + 1):
            adjustments.append(Windowed.calc_level_adjustment(s, ma, w, i))

        return sum(adjustments) / phi

    @staticmethod
    def plot_filtered(arr, org, valid, ax, c):
        _row = 0
        for index in range(0, len(org)):
            if valid[index]:
                ax.plot(arr[_row], color=c[index])
                _row += 1
        return ax

    @staticmethod
    def review_window_lengths(arr, dsw_window, window_range=(2, 8)):
        print(f'DSW window length: {dsw_window}.\nProcessing moving average windows:')
        for length in range(*window_range):
            filtered = Windowed.calculate_weighted_moving_average(arr, length)
            adjustment = Windowed.calc_level_adjustments_ma(arr, filtered, dsw_window)
            print(f"Window length: {length} | Adjustment: {adjustment}")

    @staticmethod
    def process(arr, ma_window, dsw_window):

        moving_average = Windowed.calculate_weighted_moving_average(arr, ma_window)
        filtered_data, discarded = Windowed.discard_limits(arr, ma_window, dsw_window)
        windowed_data = Windowed.transform_to_sliding_windows(filtered_data, moving_average, dsw_window)
        filtered_windowed_data, outlier_info = Windowed.discard_outliers(windowed_data, threshold=100)

        normalized_data, norm_minimum, norm_maximum = Windowed.normalize(
            filtered_windowed_data,
            minimum=outlier_info['lower_limit'] if outlier_info['has_less'] else None,
            maximum=outlier_info['upper_limit'] if outlier_info['has_greater'] else None
        )

        denormalized_data = Windowed.denormalize(normalized_data, norm_minimum, norm_maximum)
        reconstructed_data = Windowed.detransform(denormalized_data, moving_average)

        if discarded:
            reconstructed_data = np.concatenate([discarded, reconstructed_data])

        return normalized_data, reconstructed_data, filtered_windowed_data

    @staticmethod
    def plot(original, windowed, normalized, reconstructed):
        fig, axs = plt.subplots(2, 2)
        axs[0][0].plot(original)
        axs[0][0].set_title('Original')

        axs[0][1].plot(windowed)
        axs[0][1].set_title('Transformed Windows')

        axs[1][0].plot(normalized)
        axs[1][0].set_title('Filtered Normalized Windows')

        axs[1][1].plot(reconstructed)
        axs[1][1].set_title('Reconstructed')
        plt.show()


def daily_norm(df):
    df = open_excel(df, False)

    drops = ["open", "high", "low", "close", "volume", "time", "Mean1-Sh", "STD1-Sh", "Mean1-Cap",
             "STD1-Cap", "Volume-M", "Volume-STD", "Price-M", "Price-STD", "Price1-M", "Price1-STD",
             "PO1", "PH1", "PL1", "PC1", "PV1", "PO2", "PH2", "PL2", "PC2", "PV2", "PO3", "PH3", "PL3", "PC3", "PV3",
             "PO4", "PH4", "PL4", "PC4", "PV4", "PO5", "PH5", "PL5", "PC5", "PV5", "PreReturn%", "PreRange", "Gap%",
             "h", "l", "c", "v"]

    drops = ["open", "high", "low", "close", "volume", "time", "Mean1-Sh", "STD1-Sh", "Mean1-Cap",
             "STD1-Cap", "Volume-M", "Volume-STD", "Price-M", "Price-STD", "Price1-M", "Price1-STD",
             "PO1", "PH1", "PL1", "PC1", "PV1", "PO2", "PH2", "PL2", "PC2", "PV2", "PO3", "PH3", "PL3", "PC3", "PV3",
             "PO4", "PH4", "PL4", "PC4", "PV4", "PO5", "PH5", "PL5", "PC5", "PV5", "PreReturn%", "PreRange", "Gap%",
             "h", "l", "c", "v"]

    df2 = df[drops].copy()
    df = df.drop(drops, axis=1)
    up_time = df2["time"].copy()

    print("Normalizing")
    avg, dev = df2.mean(), df2.std()
    df2 = (df2 - avg) / dev

    print("Removing Daily Volatility and Seasonality \n")
    volatility = df2.groupby(df2["time"]).std()
    seasonality = df2.groupby(df2["time"]).mean()

    df_vol = []
    df_ses = []

    for day in volatility.index:

        df_vol_pl = pd.DataFrame(
            [volatility.loc[day].to_numpy()],
            index=df2[df2["time"] == day].index,
            columns=volatility.keys())

        df_ses_pl = pd.DataFrame(
            [seasonality.loc[day].to_numpy()],
            index=df2[df2["time"] == day].index,
            columns=seasonality.keys())

        df_vol.append(df_vol_pl)
        df_ses.append(df_ses_pl)

    df_vol = pd.concat(df_vol, ignore_index=True)
    df_ses = pd.concat(df_ses, ignore_index=True)
    df2.pop("time")

    df2 = df2 - df_ses
    df2 = df2 / df_vol

    columns = df2.columns.tolist() + df.columns.tolist() + ["time"]
    df = pd.concat([df2, df, up_time], ignore_index=True, axis=1)
    df = df.rename(columns=dict(zip(range(len(columns)), columns)))

    return df


def daily_norm2(df, cols):
    df2 = open_excel(df, False)
    up_time = df2["time"].copy()
    df2 = df2[cols+["time"]]

    print("Normalizing")
    avg, dev = df2.mean(), df2.std()
    df2 = (df2 - avg) / dev

    print("Removing Daily Volatility and Seasonality \n")
    volatility = df2.groupby(df2["time"]).std()
    seasonality = df2.groupby(df2["time"]).mean()

    df_vol = []
    df_ses = []

    for day in volatility.index:

        df_vol_pl = pd.DataFrame(
            [volatility.loc[day].to_numpy()],
            index=df2[df2["time"] == day].index,
            columns=volatility.keys())

        df_ses_pl = pd.DataFrame(
            [seasonality.loc[day].to_numpy()],
            index=df2[df2["time"] == day].index,
            columns=seasonality.keys())

        df_vol.append(df_vol_pl)
        df_ses.append(df_ses_pl)

    df_vol = pd.concat(df_vol, ignore_index=True)
    df_ses = pd.concat(df_ses, ignore_index=True)
    df2.pop("time")

    df2 = df2 - df_ses
    df2 = df2 / df_vol

    columns = df2.columns.tolist() + ["time"]
    df = pd.concat([df2, up_time], ignore_index=True, axis=1)
    df = df.rename(columns=dict(zip(range(len(columns)), columns)))

    return df
