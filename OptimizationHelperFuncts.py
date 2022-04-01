from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from math import exp, sqrt
import sklearn.metrics
import pandas as pd
import skmultiflow
import river


def sigmoid_function(x, a=1, b=1):
    return 1 / (1 + exp((a - x) / (sqrt(a * b))))


def drop_split(df, var, drops, split_date):
    split_size = len(df[df["time"] > split_date])
    X_train, X_test, y_train, y_test = train_test_split(df, df[var], test_size=split_size, shuffle=False)
    X_train = X_train.drop(columns=drops)

    return X_train, X_test, y_train, y_test


def delta_sim(X_test, y_test, y_pred, space=True):
    bank = 1
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    X_test = X_test.loc[X_test["Pred1"] == 1]
    precision = precision_score(y_test, y_pred)

    for i in X_test["Delta"]:
        bank += -i * bank

    if space:
        print()
    print("Len: ", len(X_test))
    print("Bank: ", bank)
    print("Precision: ", precision)
    print("Delta: ", X_test["Delta"].mean())

    return 1.0 - precision


def sig_precision(X_test, y_test, y_pred, a=52, b=3, space=True):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    X_test = X_test.loc[X_test["Pred1"] == 1]
    sig = 1 - sigmoid_function(len(X_test), a, b)
    precision = precision_score(y_test, y_pred)

    # r1 = X_test["DeltaH"].mean() / X_test["DeltaL"].mean()
    # X_test["R2"] = X_test["DeltaH"] / X_test["DeltaL"]

    if space:
        print()
    print("Len: ", len(X_test))
    print("Precision: ", precision)
    print("Delta: ", X_test["Delta"].mean())
    # print("Range1: ", r1)
    # print("Range2: ", X_test["Delta2"].mean())

    return (1.0 - precision) + sig


def precision(X_test, y_test, y_pred, space=True):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    X_test = X_test.loc[X_test["Pred1"] == 1]
    precision = precision_score(y_test, y_pred)
    r1 = X_test["DeltaH"].mean() / X_test["DeltaL"].mean()
    X_test["R2"] = X_test["DeltaH"] / X_test["DeltaL"]

    if space:
        print()
    print("Len: ", len(X_test))
    print("Precision: ", precision)
    print("Delta: ", X_test["Delta"].mean())
    print("Range1: ", r1)
    print("Range2: ", X_test["Delta2"].mean())

    return 1.0 - precision


def acc(X_test, y_test, y_pred, space=True):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    days = X_test["time"].drop_duplicates().reset_index(drop=True)
    y_test = []
    y_pred = []

    for day in days:
        df = X_test[X_test["time"] == day]
        y_test.append(df["Close2"].mean())
        y_pred.append(df["Pred1"].mean())
        print(df["Close2"].mean())
        print(df["Pred1"].mean())
        print()

    precision = accuracy_score(y_test, y_pred)
    if space:
        print()
    print("accuracy: ", precision)
    return 1.0 - precision


def delta_precision(X_test, _, y_pred, a=52, b=3):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    X_test = X_test.loc[X_test["Pred1"] == 1]
    sig = 1 - sigmoid_function(len(X_test), a, b)
    if len(X_test) > 0:
        delta = X_test["Delta"].mean()
    else:
        delta = -10

    print("\nLen: ", len(X_test))
    print("Delta: ", delta)

    return (-1*delta) + sig


def mean_reg(X_test, _, y_pred):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    length = len(X_test)
    correct = len(X_test[X_test["Delta"] < -0.01]) / length
    estimate = len(X_test[X_test["Pred1"] < -0.01]) / length
    r2 = sklearn.metrics.r2_score(X_test["Delta"], X_test["Pred1"])
    dev = correct - estimate

    print("\nMean Dev: ", dev)
    print("Mean: ", correct)
    print("Pred: ", estimate)
    print("R2: ", r2)

    return (dev**2)*10000


def dev(X_test, y_test, y_pred):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    acc = accuracy_score(y_test, y_pred)
    correct = X_test["Close2"].mean()
    estimate = X_test["Pred1"].mean()
    dev = correct - estimate

    print("\nMean Dev: ", dev)
    print("Mean: ", correct)
    print("Pred: ", estimate)
    print("Accuracy: ", acc)

    return (dev**2)*10000


def time_series(X_test, y_test, y_pred):
    X_test = X_test.copy()
    X_test["Pred1"] = y_pred
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)

    print("\nAccuracy: ", acc)
    print("Precision: ", pre)

    df = X_test[X_test["Pred1"] == 1]
    if len(df) < 10:
        return 100
    df = df.drop(columns=["Pred1"])

    drops = ["volume", "high", "low", "close", "Delta", "h", "l", "c", "v", "Symbol"]
    days = df["time"].drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=drops)
    y = df.pop("Close3")
    arr = []

    stream1 = skmultiflow.data.DataStream(df, y=pd.DataFrame(y))
    stream2 = skmultiflow.data.DataStream(df, y=pd.DataFrame(y))
    ht = skmultiflow.trees.HoeffdingAdaptiveTreeClassifier(grace_period=20)

    for day in days:
        metric = river.metrics.Precision()
        length = range(len(df[df["time"] == day]))
        for _ in length:
            X, Y = stream1.next_sample()
            if day > days.median():
                metric.update(Y[0], ht.predict(X)[0])

        for _ in length:
            X, Y = stream2.next_sample()
            ht.partial_fit(X, Y, classes=stream2.target_values)

        arr.append(metric.get())

    mean = sum(arr)/len(arr)
    print("Mean: ", mean)

    if mean == 0:
        return 100
    return 1/mean