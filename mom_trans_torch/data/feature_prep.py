"""Module to prepare features for ML"""
import importlib
from typing import Union, List
import pandas as pd
import numpy as np

from datetime import datetime

from empyrical import sharpe_ratio


def prepare(
    data: Union[pd.DataFrame, pd.Series], feature_dict: dict
) -> Union[pd.DataFrame, pd.Series]:
    """Abstract method to prepare features for ML

    :param pd.DataFrame data: the data to prepare
    :param dict feature_dict: the features to prepare
    :return: the prepared data
    :rtype: pd.DataFrame
    """
    module = importlib.import_module(__name__)
    if feature_dict:
        for args in feature_dict:
            feat = args[0]
            func = args[1]
            prepped = getattr(module, func)(
                data["ticker"], data[feat], *args[2:]
            )  # .values
            cols = prepped.columns.tolist()
            data[cols] = prepped[cols].values
            drop_columns = np.setdiff1d(feat, cols).tolist()
            data = data.drop(columns=drop_columns)
    return data


def ffill(tickers: pd.Series, series: pd.Series) -> pd.Series:
    """Forward fill a series by ticker

    :param pd.Series tickers: the tickers
    :param pd.Series series: the series to forward fill
    :return: the forward filled series
    :rtype: pd.Series
    """
    return series.groupby(tickers).ffill()


def rolling_sharpe(
    tickers: pd.DataFrame, series: pd.Series, lookback_windows: List[int]
) -> pd.DataFrame:
    features = pd.concat(
        [series]
        + [
            series.groupby(tickers).rolling(lbw).apply(sharpe_ratio)
            for lbw in lookback_windows
        ],
        axis=1,
    )
    return features


