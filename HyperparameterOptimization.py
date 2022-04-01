from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from OptimizationHelperFuncts import sig_precision, time_series, drop_split
from sklearn.tree import DecisionTreeClassifier
from iexDataProcessing import open_excel
from skopt.utils import use_named_args
from skopt.space import Integer, Real
from skopt import gp_minimize
import pandas as pd


def bayesian_hp_opt(file, weeks=1):
    file = open_excel(file)
    file = file.fillna(0)
    file = file[file["time"] < 1e15]
    end = file["time"].max()
    search_space = list()

    search_space.append(Integer(2, 8, name='estimators'))
    search_space.append(Integer(1, 111, name='max_depth'))
    search_space.append(Integer(2, 10000, name='min_samples_split'))
    search_space.append(Real(0.0000001, 0.9999999, name='learning_rate'))
    # search_space.append(Real(0.00001, 0.9999, name='max_samples'))
    # search_space.append(Real(0.00001, 0.9999, name='max_features'))

    skip = 604800 * weeks
    split = 1625202000 - 604800
    print("BaggingClassifier")
    print(f"Weeks: {weeks}")
    print(search_space)

    @use_named_args(search_space)
    def split_agg_opt(**params):
        start = split
        X_test = pd.DataFrame()
        y_test = pd.DataFrame()
        y_pred = pd.DataFrame()

        while start < end:
            df = file[file["time"] < start + skip]

            X_t, y_t, y_p = evaluate_model(df, params, start)
            X_test = X_test.append(X_t, ignore_index=True)
            y_test = y_test.append(pd.DataFrame(y_t), ignore_index=True)
            y_pred = y_pred.append(pd.DataFrame(y_p), ignore_index=True)
            start += skip

        sig_precision(X_test, y_test, y_pred)
        sig = time_series(X_test, y_test, y_pred)
        print(params)
        return sig

    def evaluate_model(df, params, start):
        drops = ["Close2", "volume", "Symbol", "high", "low", "close", "Delta", "h", "l", "c", "v", "Close3"]
        X_train, X_test, y_train, y_test = drop_split(df, var="volume", split_date=start, drops=drops)

        model = AdaBoostClassifier(DecisionTreeClassifier(
                min_samples_split=params['min_samples_split'],
                max_depth=params['max_depth']),
            learning_rate=params['learning_rate'],
            n_estimators=params['estimators'])
        # model = RandomForestClassifier(max_depth=params['max_depth'],
        #                                n_estimators=10 * params['estimators'])

        # model = BaggingClassifier(DecisionTreeClassifier(max_depth=params['max_depth']),
        #                           max_samples=params['max_samples'],
        #                           max_features=params['max_features'],
        #                           n_estimators=params['estimators'])

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.drop(columns=drops))
        return X_test, y_test, y_pred

    result = gp_minimize(split_agg_opt, search_space, verbose=True, n_jobs=1, n_calls=50)
    print('\nBest Value: %.3f' % result.fun)
    print('Best Parameters: %s' % result.x)