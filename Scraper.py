from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import requests
import config


def finviz_scraper(ticker_file, output):
    xl = pd.read_excel(rf"{config.TEST_FILE_PATH}\{ticker_file}.xlsx", index_col="Symbol").index
    percent_cols = ['Dividend', 'Payout Ratio', 'EPS this Y', 'EPS next Y', 'EPS past 5Y', 'EPS next 5Y',
                    'Sales past 5Y', 'EPS Q/Q', 'Sales Q/Q', 'Insider Own', 'Insider Trans',
                    'Inst Own', 'Inst Trans', 'Float Short', 'ROA', 'ROE', 'ROI', 'Gross M',
                    'Oper M', 'Profit M', 'Perf Week', 'Perf Month', 'Perf Quart',
                    'Perf Half', 'Perf Year', 'Perf YTD', 'Volatility W', 'Volatility M',
                    'SMA20', 'SMA50', 'SMA200', '50D High', '50D Low', '52W High',
                    '52W Low', 'from Open', 'Gap', 'Change']

    enc_cols = ["Sector", "Industry", "Country", "Earn1"]
    large_cols = ["Market Cap", "Outstanding", "Float", "Avg Volume"]
    drops = ["Market Cap", "Company", "from Open", "Gap", "Price", "Volume", "Earnings"]
    df = pd.DataFrame()

    def get_finviz(tickers):
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like'
                                 ' Gecko) Chrome/50.0.2661.102 Safari/537.36'}

        screen = requests.get(f'https://finviz.com/screener.ashx?v=152&t={tickers}&o=ticker&c=1,2,3,4,5,6,7,8,9,10,11,'
                              f'12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,'
                              f'40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,'
                              f'68,69,70,', headers=headers).text

        tables = pd.read_html(screen)
        tables = tables[-2]
        tables.columns = tables.iloc[0]
        tables = tables[1:]
        tables = tables.replace("-", "0")

        return tables

    for split in range(0, len(xl), 20):
        ticker_string = f"{xl[split]}"
        for ticker in xl[split+1:split+20]:
            ticker_string += f",{ticker}"

        df = df.append(get_finviz(ticker_string), ignore_index=True)
        df = df.reset_index(drop=True)

    for col in percent_cols:
        frame = df[col]
        for i in frame.index:
            if frame[i][-1] == "%":
                frame[i] = frame[i][:-1]

    for col in large_cols:
        frame = df[col]
        for i in frame.index:
            if frame[i][-1] == "K":
                frame[i] = float(frame[i][:-1]) / 1000**2
            elif frame[i][-1] == "M":
                frame[i] = float(frame[i][:-1]) / 1000
            elif frame[i][-1] == "B":
                frame[i] = float(frame[i][:-1])

    earn1 = []
    earn2 = []
    earn3 = []
    frame = df["Earnings"]
    for i in frame.index:
        if frame[i] != "0":
            earn1.append(frame[i][0:3])
            earn2.append(int(frame[i][4:6]))
            if frame[i][-1] == "a":
                earn3.append(1)
            else:
                earn3.append(0)
        else:
            earn1.append("0")
            earn2.append(0)
            earn3.append(2)

    df["Earn1"] = earn1
    df["Earn2"] = earn2
    df["Earn3"] = earn3

    frame = df["IPO Date"]
    for i in frame.index:
        frame[i] = config.get_time(frame[i], slash=True)

    for col in enc_cols:
        df[col] = OrdinalEncoder().fit_transform(df.loc[:, [col]])

    df = df.drop(columns=drops)
    for col in df.columns:
        if col != "Ticker":
            df[col] = df[col].astype(float)

    config.Printer(output, df)

