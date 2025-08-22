import os

import numpy as np
import pandas as pd

from mom_trans_torch.data.classical_strategies import (
    MACDStrategy,
    calc_returns,
    calc_daily_vol,
    calc_vol_scaled_returns,
)

VOL_THRESHOLD = 5  # multiple to winsorise by
HALFLIFE_WINSORISE = 252
TARGET_MAX_ABS = 20.0


def read_changepoint_results_and_fill_na(
    file_path: str, lookback_window_length: int
) -> pd.DataFrame:
    """Read output data from changepoint detection module into a dataframe.
    For rows where the module failed, information for changepoint location and severity is
    filled using the previous row.


    Args:
        file_path (str): the file path of the csv containing the results
        lookback_window_length (int): lookback window length - necessary for filling in the blanks for norm location

    Returns:
        pd.DataFrame: changepoint severity and location information
    """

    return (
        pd.read_csv(file_path, index_col=0, parse_dates=True)
        .fillna(method="ffill")
        .dropna()  # if first values are na
        .assign(
            cp_location_norm=lambda row: (row["t"] - row["cp_location"])
            / lookback_window_length
        )  # fill by assigning the previous cp and score, then recalculate norm location
    )


def prepare_cpd_features(folder_path: str, lookback_window_length: int) -> pd.DataFrame:
    """Read output data from changepoint detection module for all assets into a dataframe.


    Args:
        file_path (str): the folder path containing csvs with the CPD the results
        lookback_window_length (int): lookback window length

    Returns:
        pd.DataFrame: changepoint severity and location information for all assets
    """

    return pd.concat(
        [
            read_changepoint_results_and_fill_na(
                os.path.join(folder_path, f), lookback_window_length
            ).assign(ticker=os.path.splitext(f)[0])
            for f in os.listdir(folder_path)
        ]
    )


def iterative_winsorize(
    series, threshold=VOL_THRESHOLD, halflife=HALFLIFE_WINSORISE, max_iter=50
):
    """
    Iteratively winsorizes a time series by capping extreme movements beyond
    the given threshold of standard deviations from the mean.

    Parameters:
    - series (pd.Series): Input time series
    - halflife (int): Halflife of the exponential moving average used to calculate mean and standard deviation
    - threshold (float): Number of standard deviations beyond which values are capped
    - max_iter (int): Maximum number of iterations to ensure convergence

    Returns:
    - pd.Series: Winsorized time series
    """
    series = series.copy()
    prev_series = series.copy()

    for i in range(max_iter):
        mean = series.ewm(halflife=halflife).mean()
        std = series.ewm(halflife=halflife).std()

        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std

        series = series.clip(lower=lower_bound, upper=upper_bound)

        # If no changes, stop iterating
        if series.equals(prev_series):
            break

        if i == max_iter - 1:
            print(
                "Warning: Iterative winsorization did not converge after",
                max_iter,
                "iterations",
            )

        prev_series = series.copy()

    return series


def deep_momentum_strategy_features(df_asset: pd.DataFrame) -> pd.DataFrame:
    """prepare input features for deep learning model

    Args:
        df_asset (pd.DataFrame): time-series for asset with column close

    Returns:
        pd.DataFrame: input features
    """

    df_asset = df_asset[
        ~df_asset["close"].isna()
        | ~df_asset["close"].isnull()
        | (df_asset["close"] > 1e-8)  # price is zero
    ].copy()

    # winsorize using rolling 10X standard deviations to remove outliers
    df_asset["srs"] = df_asset["close"]

    # TODO - disaled for now
    # df_asset["srs"] = iterative_winsorize(
    #     df_asset["srs"], threshold=VOL_THRESHOLD, halflife=HALFLIFE_WINSORISE
    # )

    df_asset["daily_returns"] = calc_returns(df_asset["srs"])
    df_asset["daily_vol"] = calc_daily_vol(df_asset["daily_returns"])
    # vol scaling and shift to be next day returns

    def calc_normalised_returns(day_offset):
        return (
            calc_returns(df_asset["srs"], day_offset)
            / df_asset["daily_vol"]
            / np.sqrt(day_offset)
        )

    df_asset["r1d"] = calc_normalised_returns(1)
    df_asset["r1w"] = calc_normalised_returns(5)
    df_asset["r1m"] = calc_normalised_returns(21)
    df_asset["r3m"] = calc_normalised_returns(63)
    df_asset["r6m"] = calc_normalised_returns(126)
    df_asset["r1y"] = calc_normalised_returns(252)

    trend_combinations = [(4, 12), (8, 24), (16, 48), (32, 96)]
    for short_window, long_window in trend_combinations:
        df_asset[f"macd_{short_window}_{long_window}"] = MACDStrategy.calc_signal(
            df_asset["srs"], short_window, long_window
        )

    df_asset["vs_factor"] = 1 / df_asset["daily_vol"]
    df_asset["target"] = (
        calc_returns(df_asset["srs"]).shift(-1) * df_asset["vs_factor"] 
    )  # want daily vol to be approx gaussian

    df_asset["target"] = df_asset["target"].clip(-TARGET_MAX_ABS, TARGET_MAX_ABS)

    # # date features
    # if len(df_asset):
    #     df_asset["day_of_week"] = df_asset.index.dayofweek
    #     df_asset["day_of_month"] = df_asset.index.day
    #     df_asset["week_of_year"] = df_asset.index.weekofyear
    #     df_asset["month_of_year"] = df_asset.index.month
    #     df_asset["year"] = df_asset.index.year
    #     df_asset["date"] = df_asset.index  # duplication but sometimes makes life easier
    # else:
    #     df_asset["day_of_week"] = []
    #     df_asset["day_of_month"] = []
    #     df_asset["week_of_year"] = []
    #     df_asset["month_of_year"] = []
    #     df_asset["year"] = []
    #     df_asset["date"] = []

    return df_asset.dropna()


def prepare_features(data):
    """
    Prepare features for deep learning model
    
    Args:
    - data (pd.DataFrame): Input data with columns date, ticker, close
    
    Returns:
    - pd.DataFrame: Features for deep learning model
    """
    
    data = (
        data.stack()
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "ticker", 0: "close"})
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    mask = (data.groupby("ticker")["close"].pct_change() != 0).values
    data = (
        data[mask]
        # .set_index("date")
        .groupby("ticker")
        .apply(deep_momentum_strategy_features)
        .reset_index(drop=True)
        # .set_index("date")
    )
    data["date"] = pd.to_datetime(data["date"])
    data.index = data["date"]
    return data

def include_changepoint_features(
    features: pd.DataFrame, cpd_folder_name: pd.DataFrame, lookback_window_length: int
) -> pd.DataFrame:
    """combine CP features and DMN featuress

    Args:
        features (pd.DataFrame): features
        cpd_folder_name (pd.DataFrame): folder containing CPD results
        lookback_window_length (int): LBW used for the CPD

    Returns:
        pd.DataFrame: features including CPD score and location
    """
    features = features.merge(
        prepare_cpd_features(cpd_folder_name, lookback_window_length)[
            ["ticker", "cp_location_norm", "cp_score"]
        ]
        .rename(
            columns={
                "cp_location_norm": f"cp_rl_{lookback_window_length}",
                "cp_score": f"cp_score_{lookback_window_length}",
            }
        )
        .reset_index(),  # for date column
        on=["date", "ticker"],
    )

    features.index = features["date"]

    return features
