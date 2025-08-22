import os
import torch
import torch.utils.data
import pandas as pd
import itertools
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Union

from colorama import Fore, Style

import multiprocessing

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from mom_trans_torch.data.feature_prep import prepare


class ContextSet(Enum):
    NONE = 0
    RANDOM = 1
    SEGMENTED = 2


# ENERGY 2
# BOND 2
# US EQUITY 0.5
# INTERNATIONAL EQUITY 1.0
# CURRENCY 0.5
# EM fx 2
# COMMODITY 3.0
# OTHER METALS SILVER COPPER 2.0
# GOLD 1.0
# Paper
# The Impact of Volatility Targeting
# In the tables, we will report the Sharpe ratio both gross and net of transaction costs. We use the
# following transaction cost estimates, expressed as fraction of the notional value traded: 1.0bp (or
# 0.01%) for equities, 0.5bp for bonds, 0.5bp for credit, 1.0 for gold, 2.0bp for oil, and 3.5bp for
# copper.13 In figures we will just show returns gross of transaction costs, but results are very similar on
# a net basis.
def get_transaction_costs(file_path: str):
    full_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  "mom_trans_torch", "configs", "tcost", file_path + ".csv")
    costs = pd.read_csv(
        full_file_path,
        index_col=[0],
    )
    return costs.reset_index()[["ticker", "transaction_cost"]]


def calc_vs_factor_scaler(data_ex_test: pd.DataFrame) -> float:
    vs_factor_scaler = (
        1.0 / data_ex_test["vs_factor"].dropna().values.astype(np.float32).std()
    )
    return vs_factor_scaler


def calc_trans_cost_scaler(ticker_ref: pd.DataFrame) -> float:
    trans_cost_scaler = 1.0 / ticker_ref["transaction_cost"].max()
    return trans_cost_scaler


def datetime_embedding_global(start_year: int, end_year: int):
    """Create a global datetime embedding for the given years."""

    date_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year+1}-01-01")
    date_range = date_range[date_range.weekday < 5]  # pylint: disable=no-member
    return pd.Series(list(range(len(date_range))), index=date_range)


class MomentumDataset(torch.utils.data.Dataset):
    """Some Information about PinnacleData"""

    def __init__(
        self,
        settings,
        data,
        first_year=2015,
        end_year=2020,
        keep_first_perc=None,
        drop_first_perc=None,
        tickers_dict=None,
        test_set=False,
        live_mode=False,
        date_time_embedding=None,
        target_override=None,
        # scale_up_to_dailystd1=False,
    ):
        super(MomentumDataset, self).__init__()
        try:
            if keep_first_perc:
                assert not drop_first_perc
                assert not test_set
        except ValueError:
            print(Fore.RED + "Options for train/valid/test are mutually exclusive")
            print(Style.RESET_ALL)
            raise

        self.settings = settings
        self.seq_len = settings["seq_len"]

        self.num_features = len(settings["features"])
        self.num_targets = 1
        self.pre_loss_steps = (
            settings["pre_loss_steps"] + settings["extra_data_pre_steps"]
        )

        pred_steps = self.seq_len - self.pre_loss_steps
        self.pred_steps = pred_steps

        data = data.copy()

        data.index = pd.to_datetime(data.index)
        data = prepare(data, settings["feature_prepare"])

        if target_override is not None:
            data["target"] = data[target_override]

        if date_time_embedding is not None:
            data = data.loc[data.index.isin(date_time_embedding.index)]

        if live_mode:
            if not test_set:
                test_set = True
                print(Fore.RED + "WARNING: defaulting to test_set=True in live mode")
                print(Style.RESET_ALL)
            data["target"] = 0.0

        data["date"] = data.index

        if "fillna_features" in settings:
            if "ffill" in settings["fillna_features"]:
                fill_cols = settings["fillna_features"]["ffill"]
                data[fill_cols] = data[fill_cols].ffill()

            if "value" in settings["fillna_features"]:
                data = data.fillna(settings["fillna_features"]["value"])
            # else:
            #     raise NotImplementedError("Only value fillna implemented")

        if "dataset_filter_year" in settings:
            data = data[data.index.year >= settings["dataset_filter_year"]]

        ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
        # TODO  maybe move to include in full univ

        data = data[data.ticker.isin(ticker_ref["ticker"])]

        extra_cols = ["vs_factor", "vs_factor_prev", "transaction_cost", "ticker_id"]
        data["vs_factor_prev"] = data.groupby("ticker")["vs_factor"].shift(1).copy()

        data["date"] = pd.to_datetime(data["date"])
        data = data.reset_index(drop=True).sort_values(["ticker", "date"])

        buffer = self.seq_len if test_set else self.pre_loss_steps
        if first_year:
            data = data[
                data.groupby("ticker")["date"].shift(-buffer + 1).ffill().dt.year
                >= first_year
            ]

        if end_year:
            data = data[data["date"].dt.year < end_year]

        if first_year or end_year:
            data = data.reset_index(drop=True)

        data["date_index"] = data.groupby("date").ngroup()

        self.num_tickers_full_univ = data["ticker"].nunique()

        self.signal_weight_inputs = (
            []
            if "specify_weight_features" not in settings
            else settings["specify_weight_features"]
        )
        remaining_inputs = list(
            set(self.signal_weight_inputs) - set(settings["features"])
        )
        self.num_signal_weight_inputs = len(self.signal_weight_inputs)
        self.weight_fields = (
            self.signal_weight_inputs
        )  # TODO cleanup - this is a duplicate but needed later

        data = (
            data[
                settings["reference_cols"]
                + settings["features"]
                + list(set(extra_cols) - {"ticker_id", "transaction_cost"})
                + [settings["target"]]
                + remaining_inputs
            ]
            .dropna()
            .copy()
        )

        self.ticker_ref = ticker_ref

        if tickers_dict is None:
            data = data.merge(
                pd.DataFrame({"ticker": data["ticker"].unique()})
                .reset_index()
                .rename(columns={"index": "ticker_id"})
            )
            self.tickers_dict = (
                data[["ticker", "ticker_id"]]
                .drop_duplicates()
                .set_index("ticker")["ticker_id"]
            )
        else:
            self.tickers_dict = tickers_dict
            data = data.merge(tickers_dict.to_frame(), on="ticker")

        data = data.merge(ticker_ref[["ticker", "transaction_cost"]], on="ticker")

        data = (
            data[
                settings["reference_cols"]
                + settings["features"]
                + extra_cols
                + [settings["target"]]
                + remaining_inputs
            ]
            # .dropna()
            .reset_index(drop=True).copy()
        )

        self.tickers = data["ticker_id"].values
        # self.num_tickers = len(self.tickers_dict)
        self.num_tickers = data["ticker_id"].nunique()

        self.inputs = torch.Tensor(data[settings["features"]].values)   # Only the feature numerical values and without the target value
        if len(self.signal_weight_inputs) > 0:
            self.inputs_to_weight = torch.Tensor(data[self.signal_weight_inputs].values)
            if settings["sigcom_sgn"]:
                self.inputs_to_weight = torch.sign(self.inputs_to_weight)
        else:
            self.inputs_to_weight = None

        self.outputs = torch.Tensor(data[settings["targets"]].values)
        self.date_mask = torch.Tensor(
            (
                (data["date"].dt.year >= first_year) & (data["date"].dt.year < end_year)
            ).values
        ).bool()

        self.dates = data["date"].dt.strftime("%Y-%m-%d").values

        self.vs_factor = data["vs_factor"].values.astype(np.float32)
        self.vs_factor_prev = data["vs_factor_prev"].values.astype(np.float32)

        # TODO revistit this later... was too small

        self.trans_cost_bp = data["transaction_cost"].values.astype(np.float32)

        data["cum_count"] = data.groupby("ticker").cumcount()

        # split based on cumcount

        data["count"] = data.groupby(["ticker"])["cum_count"].transform("count")

        prediction_day = data[data["cum_count"] >= self.pre_loss_steps][
            ["date", "ticker", "cum_count", "count"]
        ].copy()

        # if keep_first_perc or drop_first_perc:
        prediction_day["pred_cum_count"] = (
            prediction_day["cum_count"] - self.pre_loss_steps
        )
        prediction_day["pred_count"] = prediction_day["count"] - self.pre_loss_steps

        self.all_prediction_day = prediction_day

        # prediction_day["num_seqs"] = (prediction_day["count"] - pre_loss_steps + 1)//pred_steps
        prediction_day["num_seqs"] = prediction_day["pred_count"] // pred_steps

        offset = (prediction_day.groupby("ticker").transform("max")["pred_count"]) % (
            prediction_day["num_seqs"] * pred_steps
        )

        prediction_day["pred_cum_count"] = (
            prediction_day["cum_count"] - self.pre_loss_steps + 1
        )

        prediction_day["seq_num"] = (
            prediction_day["pred_cum_count"] - offset - 1
        ) // pred_steps

        prediction_day = prediction_day[prediction_day["seq_num"] >= 0].copy()

        assert not (len(prediction_day) % pred_steps)

        prediction_day["portion"] = prediction_day["seq_num"] / (
            prediction_day["num_seqs"] - 1
        )

        if keep_first_perc:
            prediction_day = prediction_day[prediction_day["portion"] < keep_first_perc]
        if drop_first_perc:
            prediction_day = prediction_day[
                prediction_day["portion"] >= drop_first_perc
            ]

        seq_indexes = (
            prediction_day.reset_index()
            .groupby(["ticker", "seq_num"])["index"]
            .last()
            .map(lambda i: list(range(i - self.seq_len + 1, i + 1)))
        )
        self.seq_indexes = seq_indexes.tolist()
        # self.seq_indexes_end = np.array(self.seq_indexes)[:, seq_len - 1].tolist()

        # self.datetime_embedding = date_time_embedding
        if date_time_embedding is not None:
            # date_time_embedding.index = date_time_embedding.index.strftime("%Y-%m-%d")
            self.date_time_embedding_index = date_time_embedding.loc[
                self.dates.tolist()
            ].values
        else:
            self.date_time_embedding_index = None

    def __getitem__(self, index):
        return (
            self.inputs[self.seq_indexes[index]],
            self.outputs[self.seq_indexes[index]],
            self.date_mask[self.seq_indexes[index]],
            self.dates[self.seq_indexes[index]].tolist(),
            self.tickers[self.seq_indexes[index][0]],
            self.vs_factor[self.seq_indexes[index]],  # [self.pre_loss_steps :],
            self.vs_factor_prev[self.seq_indexes[index]],  # [self.pre_loss_steps :],
            self.trans_cost_bp[self.seq_indexes[index]],  # [self.pre_loss_steps :],
            (
                []
                if self.date_time_embedding_index is None
                else self.date_time_embedding_index[self.seq_indexes[index]]
            ),
            (
                self.inputs_to_weight[self.seq_indexes[index]]
                if self.inputs_to_weight is not None
                else []
            ),
        )

    def __len__(self):
        return len(self.seq_indexes)

    @property
    def tickers_from_idx_dict(self):
        return {value: key for key, value in self.tickers_dict.items()}


def cat_tensors(x):
    return torch.cat([x[i] for i in range(x.shape[0])], dim=0)


def unpack_torch_dataset(
    samples,
    dataset: Union[MomentumDataset,],
    device: torch.device,
    use_dates_mask=False,
    live_mode: bool = False,
):
    (
        target_x,
        target_y,
        date_mask,
        dates,
        target_tickers,
        vol_scaling_amount,
        vol_scaling_amount_prev,
        trans_cost_bp,
        date_time_embedding_index,
        to_weight_x,
    ) = samples[:10]

    target_x = target_x.to(device)
    target_tickers = target_tickers.to(device)

    # TODO maybe don't use this unless in test mode if not using trans costs
    vol_scaling_amount = vol_scaling_amount.to(device)
    vol_scaling_amount_prev = vol_scaling_amount_prev.to(device)
    trans_cost_bp = trans_cost_bp.to(device)

    if live_mode:
        target_y = None
    else:
        target_y = target_y.to(device)

    if use_dates_mask:
        date_mask = date_mask.to(device)
    else:
        date_mask = None

    if date_time_embedding_index == []:
        date_time_embedding_index = None

    if to_weight_x == []:
        to_weight_x = None
    else:
        to_weight_x = to_weight_x.to(device)

    # actually move to gpu later once filtered...
    # else:
    #     date_time_embedding_index = date_time_embedding_index.to(device)

    extra_data = {}

    return {
        "target_x": target_x,
        "target_y": target_y,
        "date_mask": date_mask,
        "dates": dates,
        "target_tickers": target_tickers,
        "vol_scaling_amount": vol_scaling_amount,
        "vol_scaling_amount_prev": vol_scaling_amount_prev,
        "trans_cost_bp": trans_cost_bp,
        "date_time_embedding_index": date_time_embedding_index,
        "to_weight_x": to_weight_x,
        **extra_data,
    }
