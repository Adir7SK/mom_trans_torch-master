"""Traditional models for benchmarking DMNs"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

DEFAULT_SIGN_FEATURES = ["r1y"]
DEFAULT_MACD_FEATURES = ["macd_8_24", "macd_16_48", "macd_32_96"]


def position_sizing(x: float) -> float:  # pylint: disable=invalid-name
    """Trading signal for MACD trend

    :param x: trend
    :type x: float
    :return: trading signal
    :rtype: float
    """
    return x * np.exp(-(x**2) / 4) / 0.89


class AbstractModel(ABC):  # pylint: disable=too-few-public-methods
    """Abstract class for models"""

    def _prepare(self, data: pd.DataFrame, tickers: List[str], start_date: str):
        """Prepare data for prediction

        :param data: data
        :type data: pd.DataFrame
        :param tickers: tickers
        :type tickers: List[str]
        :param start_date: start date
        :type start_date: str
        :return: prepared data
        :rtype: pd.DataFrame

        """
        return data[data.ticker.isin(tickers)].loc[start_date:]

    @abstractmethod
    def predict(
        self, data: pd.DataFrame, tickers: List[str], start_date: str
    ) -> pd.DataFrame:
        """Predict positions
        :param data: data
        :type datga: pd.DataFrame
        :param tickers: tickers
        :type tickers: List[str]
        :param start_date: start date
        :type start_date: str
        :return: positions
        :rtype: pd.DataFrame
        """

        # tickers included here because of Neural Process Edge case which
        # conditions on all assets (not just traded assets)
        raise NotImplementedError


class Long(AbstractModel):  # pylint: disable=too-few-public-methods
    """Long only model"""

    def __init__(self, features=None, **kwargs):
        super().__init__()

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str):
        data = self._prepare(data, tickers, start_date)
        return (
            data[[]]
            .assign(position=1.0)
            .set_index([data.index, data.ticker])["position"]
            .unstack()
        )


class Sign(AbstractModel):  # pylint: disable=too-few-public-methods
    """Position based on of returns over different timescales"""

    def __init__(self, features=None, **kwargs):
        if features is None:
            features = DEFAULT_SIGN_FEATURES
        self.features = features
        super().__init__()

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str):
        data = self._prepare(data, tickers, start_date)
        return (
            data[self.features]
            .apply(np.sign)
            .set_index([data.index, data.ticker])
            .mean(axis=1)
            .unstack()
        )


class EqualWeight(AbstractModel):  # pylint: disable=too-few-public-methods
    """Position based on of returns over different timescales"""

    def __init__(self, features=None, **kwargs):
        if features is None:
            features = DEFAULT_SIGN_FEATURES
        self.features = features
        super().__init__()

    def predict(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        output_signal_weights=False,
    ):
        data = self._prepare(data, tickers, start_date)
        if output_signal_weights:
            weights = data[["ticker"] + self.features].copy()
            weights.loc[:,self.features] = 1/len(self.features)
            return weights

        return (
            data[self.features]
            # .apply(np.sign)
            .set_index([data.index, data.ticker])
            .mean(axis=1)
            .unstack()
        )


class MACD(AbstractModel):  # pylint: disable=too-few-public-methods
    """MACD model"""

    def __init__(self, features=None, **kwargs):
        if features is None:
            features = DEFAULT_MACD_FEATURES
        self.features = features
        super().__init__()

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str):
        data = self._prepare(data, tickers, start_date)
        return (
            data.groupby("ticker", group_keys=True)[self.features]
            .apply(position_sizing)
            .mean(axis=1)
            .unstack()
            .T
        )
