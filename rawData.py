import time
import time as t
import requests
import pandas as pd
import numpy as np
import config
import dateutil.parser as dp
from alpaca_trade_api.rest import TimeFrame


def td(ticker, start=1577862000000, end=1638857212000, session=requests.session()):
    global df
    """
    "periodType": "day",
    "frequencyType": "minute",

    "periodType": "year",
    "frequencyType": "daily",
    """
    try:
        time.sleep(0.5)
        response = session.get(f"{config.TD_BASE_URL}/marketdata/{ticker}/pricehistory",
                               params={"apikey": config.TD_TOKEN,
                                       "periodType": "year",
                                       "frequencyType": "daily",
                                       # "frequency": "30",
                                       # "period": "3",
                                       "startDate": start,
                                       "endDate": end,
                                       "needExtendedHoursData": "false"}
                               ).json()
        df = pd.DataFrame(response["candles"])

    except:
        df = pd.DataFrame()
        time.sleep(30)

    return df


def polygon(ticker, start, stop):
    response = requests.get(f"{config.POLYGON_BASE_URL}aggs/ticker/{ticker}/range/1/hour/{start}/{stop}"
                            f"?unadjusted=false&sort=asc&limit=50000&apiKey={config.POLYGON_API_KEY}").json()
    t.sleep(12)
    return pd.DataFrame(response['results'])


def iex(ticker, start, stop, time="min"):
    """
    Time format: 2020-02-14
    """

    if time=="day":
        time = TimeFrame.Day
    elif time=="min":
        time = TimeFrame.Minute
    elif time=="hour":
        time = TimeFrame.Hour

    api = config.iex_api()
    barset = api.get_bars(ticker, time,  start, stop, adjustment='raw').df
    barset.reset_index(inplace=True)
    barset.rename(columns={"timestamp": "time"}, inplace=True)

    barset['time'] = pd.to_datetime(barset['time'],
                                    format='%Y-%m-%d %H:%M:%S+%z',
                                    errors='coerce').view(np.int64)
    barset['time'] = (barset['time'] / 1000000000).astype(np.int64)

    return barset


def iex_aggs(tickers, time="hour", start='2018-01-02', stop='2021-01-03'):
    """
    Start: inclusive
    End: exclusive
    """
    api = config.iex_api()
    barset = api.get_aggs(symbol=tickers, timespan=time, multiplier=1,
                          _from=start, to=stop)
    return barset


def iex_v2(ticker, time="day", start='2021-02-28 00:00:00', stop='2022-02-28 00:00:00',
           session=requests.session(), page=None):
    t.sleep(0.5)
    chicago = 'America/Chicago'
    start = pd.Timestamp(start, tz=chicago).isoformat()
    end = pd.Timestamp(stop, tz=chicago).isoformat()

    response = session.get(f"{config.APCA_API_DATA_URL_V2}/stocks/{ticker}/bars",
                           params={"start": start,
                                   "end": end,
                                   "timeframe": time,
                                   "limit": "10000",
                                   "page_token": page},
                           headers={"APCA-API-KEY-ID": config.APCA_API_KEY_ID,
                                    "APCA-API-SECRET-KEY": config.APCA_API_SECRET_KEY}
                           ).json()

    token = response["next_page_token"]
    df = pd.DataFrame(response["bars"])

    if "t" in df:
        for i in range(0, len(df["t"])):
            df.loc[i, "t"] = dp.parse(df["t"].loc[i]).timestamp()

    if token is not None:
        df = df.append(iex_v2(ticker, time, start, stop, session, page=token),
                       ignore_index=True)

    return df
