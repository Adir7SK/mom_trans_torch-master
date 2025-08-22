"""Job for running live predictions"""

import argparse
import getpass
import importlib
import os
import time
from copy import deepcopy
from typing import List

from empyrical import sharpe_ratio

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import yaml

from mom_trans_torch.data.make_torch_dataset import get_transaction_costs
from mom_trans_torch.utils.logging_utils import get_logger
from mom_trans_torch.data.build_features import prepare_features

from mom_trans_torch.data.utils import atomic_save_parquet

from empyrical import sharpe_ratio, calmar_ratio

TARGET_VOL = 0.10

def perf_metrics(returns):
    """
    Calculate performance metrics for a given time-series of returns.

    :param returns: pd.Series, time-series of returns
    :return: dict
    """
    metrics = {}
    metrics["annual_return"] = returns.mean() * 252
    metrics["annual_vol"] = returns.std() * np.sqrt(252)
    metrics["sharpe_ratio"] = sharpe_ratio(returns)
    metrics["calmar_ratio"] = calmar_ratio(returns)
    # TODO more
    return metrics

class MakePredictionsJob:
    """Make predictions using a model for a list of tickers and optionally run diagnostics."""

    logger = get_logger(__name__)

    def __init__(
        self,
        name: str,
        tickers: List[str],
        start_date: str,
        data_parquet: str,
        save_dir: str,
        cfg_model: dict,
        transaction_cost_path: str,
        variable_importance_mode: bool = False,
        output_signal_weights: bool = False,
        directory_already_altered: bool = False,
        end_date: str = None,
        data_subfolder: str = None,
        # live=False,
        # x_curr_pairs_to_usd={},
    ):
        """
        Make predictions using a model for a list of tickers and optionally run diagnostics.

        :param name: The name of the model.
        :type name: str
        :param tickers: A list of stock tickers for which predictions will be made.
        :type tickers: list of str
        :param start_date: The date from which to start making predictions.
        :type start_date: str
        :param cfg_data: A dictionary containing configuration parameters for the data.
        :type cfg_data: dict
        :param data_yaml_name: The name of the data yaml file.
        :type data_yaml_name: str
        :param save_dir: The directory where the predictions will be saved.
        :type save_dir: str
        :param cfg_model: A dictionary containing configuration parameters for the machine
                        learning model.
        :type cfg_model: dict
        :param transaction_cost_path: The path to the transaction cost file.
        :type transaction_cost_path: str
        :param variable_importance_mode: Whether to run in variable importance mode. Defaults to False.
        :type variable_importance_mode: bool, optional
        :param output_signal_weights: Whether to output signal weights. Defaults to False.
        :type output_signal_weights: bool, optional
        :param directory_already_altered: Whether the directory has already been altered. Defaults to False.
        :type directory_already_altered: bool, optional
        :param end_date: The date at which to end making predictions. Defaults to None.
        :type end_date: str, optional
        :param data_subfolder: The subfolder of the data directory. Defaults to None.
        :type data_subfolder: str, optional

        :return: None
        :rtype: None
        """
        self.name = name
        # self.live = live
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data_parquet = data_parquet
        self.save_dir = save_dir
        self.cfg_model = deepcopy(cfg_model)
        self.transaction_cost_path = transaction_cost_path
        self.variable_importance_mode = variable_importance_mode
        # self.x_curr_pairs_to_usd = x_curr_pairs_to_usd
        if self.variable_importance_mode and not directory_already_altered:
            self.save_dir = os.path.join(self.save_dir, "variable_importance")
        self.output_signal_weights = output_signal_weights

        self.data_subfolder = data_subfolder

        if self.output_signal_weights and not directory_already_altered:
            self.save_dir = os.path.join(self.save_dir, "signal_weights")

        if "is_fine_tune_model" in self.cfg_model:
            self.is_fine_tune_model = self.cfg_model.pop("is_fine_tune_model")
        else:
            self.is_fine_tune_model = False

        self.load_data_path = os.path.join("data", data_parquet)

        if self.data_subfolder:
            self.load_data_path = os.path.join(self.load_data_path, self.data_subfolder)

    def make_predictions(
        self,
        clear_cache: bool = False,
        save: bool = True,
    ):
        """
        Make predictions using a model for a list of tickers.

        :param clear_cache: Whether to clear the cache. Defaults to False.
        :type clear_cache: bool, optional

        :return: None
        :rtype: None
        """
        os.makedirs(self.save_dir, exist_ok=True)
        # instantiate model
        cfg_model = deepcopy(self.cfg_model)
        cfg_model["prediction_folder"] = self.save_dir

        if self.is_fine_tune_model:
            cfg_model["is_fine_tune_model"] = True

        module_name = cfg_model.pop("module")
        class_name = cfg_model.pop("class")
        smooth_signal_span = cfg_model.pop("smooth_signal_span", None)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        model = class_(**cfg_model)
        # load features

        dataframe = pd.read_parquet(self.load_data_path)
        dataframe = prepare_features(dataframe)
        self.dataframe = dataframe

        # moved ticker filter inside predict due to edge cases
        # make predictions
        extra_params = {}
        if self.variable_importance_mode:
            extra_params["variable_importance_mode"] = True
        if self.output_signal_weights:
            extra_params["output_signal_weights"] = True
        predictions = model.predict(
            dataframe, self.tickers, self.start_date, **extra_params
        )
        predictions.index = pd.to_datetime(predictions.index)
        if self.end_date is not None:
            predictions = predictions.loc[: self.end_date]

        if self.output_signal_weights:
            if predictions.index.name != "date":
                predictions.index.name = "date"
            predictions = predictions.reset_index().set_index(["date", "ticker"])
        if not clear_cache:
            existing_predictions = [
                pd.read_parquet(os.path.join(self.save_dir, f"{ticker}.parquet"))
                for ticker in self.tickers
                if os.path.exists(os.path.join(self.save_dir, f"{ticker}.parquet"))
            ]
            if len(existing_predictions) > 0:
                existing_predictions = pd.concat(existing_predictions, axis=1)
                existing_predictions.index = pd.to_datetime(existing_predictions.index)
                existing_predictions = existing_predictions.loc[
                    existing_predictions.index < self.start_date
                ].sort_index()  # soort just in case
                predictions = pd.concat([existing_predictions, predictions])
                if smooth_signal_span:
                    self.logger.warning(
                        "Cannot correctly smooth signal directly after cache date"
                    )
        if smooth_signal_span:
            predictions = predictions.ewm(span=smooth_signal_span).mean()
        

        if save:
            self.save_predictions(
                predictions,
                self.save_dir,
                self.variable_importance_mode or self.output_signal_weights,
            )

        return predictions

    @classmethod
    def save_predictions(
        cls,
        dataframe: pd.DataFrame,
        save_dir: str,
        stacked_mode: bool = False,
    ):
        """
        Save predictions to a directory.

        :param dataframe: The DataFrame containing the predictions.
        :type dataframe: pd.DataFrame
        :param save_dir: The directory where the predictions will be saved. The
                        predictions will be saved as parquet files in this directory.


        :type save_dir: str


        :return: None
        :rtype: None
        """
        os.makedirs(save_dir, exist_ok=True)
        if stacked_mode:
            for ticker, df_slice in dataframe.groupby("ticker"):
                atomic_save_parquet(
                    df_slice, os.path.join(save_dir, f"{ticker}.parquet")
                )
        else:
            for ticker in dataframe.columns:
                atomic_save_parquet(
                    dataframe[[ticker]], os.path.join(save_dir, f"{ticker}.parquet")
                )

    @classmethod
    def calc_returns(
        cls,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
        scale_to_target_vol: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate returns.

        :param dataframe: The DataFrame containing the returns.
        :type dataframe: pd.DataFrame
        :param transaction_costs: The DataFrame containing the transaction costs.
        :type transaction_costs: pd.DataFrame
        :param price_series: The DataFrame containing the price series.
        :type price_series: pd.DataFrame
        :param scale_to_target_vol: Whether to scale to target vol (individual tickers, not portfolio vol). Defaults to True.
        :type scale_to_target_vol: bool, optional


        :return: The DataFrame containing the returns.
        :rtype: pd.DataFrame
        """
        reference = (
            price_series.reset_index()
            .rename(columns={"index": "date"})
            .merge(transaction_costs, on="ticker")
        )
        reference["date"] = pd.to_datetime(reference["date"].dt.date)
        positions = (
            dataframe.stack()
            .reset_index()
            .rename(columns={"level_0": "date", 0: "position"})
        )
        positions["date"] = pd.to_datetime(positions["date"])
        positions = positions.merge(reference, on=["ticker", "date"]).dropna()
        positions["holdings_x_transaction"] = (
            positions["position"]
            * positions["vs_factor"]
            * positions["transaction_cost"]
            * 1e-4
        )
        # assume position was already held
        positions["cost"] = (
            positions.groupby("ticker")["holdings_x_transaction"]
            .diff()
            .fillna(0.0)
            .abs()
        )
        positions["gross_return"] = positions["position"] * positions["target"]
        positions["net_return"] = positions["gross_return"] - positions["cost"]

        if scale_to_target_vol:
            positions[
                [
                    "target",
                    "vs_factor",
                    "holdings_x_transaction",
                    "cost",
                    "gross_return",
                    "net_return",
                ]
            ] *= TARGET_VOL / np.sqrt(252)
        return positions

    @classmethod
    def calc_portfolio_returns(
        cls,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
        scale_to_target_vol: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate portfolio returns.

        :param dataframe: The DataFrame containing the predictions.
        :type dataframe: pd.DataFrame
        :param transaction_costs: The DataFrame containing the transaction costs.
        :type transaction_costs: pd.DataFrame
        :param price_series: The DataFrame containing the price series.
        :type price_series: pd.DataFrame
        :param scale_to_target_vol: Whether to scale to target vol (individual tickers, not portfolio vol). Defaults to True.
        :type scale_to_target_vol: bool, optional


        :return: The DataFrame containing the portfolio returns.
        :rtype: pd.DataFrame
        """
        positions = cls.calc_returns(
            dataframe, transaction_costs, price_series, scale_to_target_vol=False
        )

        portfolio_returns = (
            positions.groupby("date")[["gross_return", "net_return"]].sum()
            / positions["ticker"].nunique()
        )

        if scale_to_target_vol:
            portfolio_returns = (
                portfolio_returns / portfolio_returns.std() * TARGET_VOL / np.sqrt(252)
            )
            return portfolio_returns
        # portfolio_returns.groupby(portfolio_returns.index.year).apply(sharpe_ratio)

    def calc_group_returns(
        self,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
        ticker_groups: dict,
        scale_to_target_vol: bool = True,
    ) -> pd.DataFrame:

        positions = self.calc_returns(
            dataframe, transaction_costs, price_series, scale_to_target_vol=False
        )

        group_returns = (
            positions.groupby(["date", "ticker"])[["gross_return", "net_return"]]
            .sum()
            .reset_index()
            .merge(
                pd.Series(ticker_groups, name="group")
                .to_frame()
                .reset_index()
                .rename(columns={"index": "ticker"}),
                on="ticker",
                how="left",
            )
            .groupby(["date", "group"])[["gross_return", "net_return"]]
            .sum()
        )

        if scale_to_target_vol:
            combined_returns = group_returns.groupby("date")[
                ["gross_return", "net_return"]
            ].sum()
            group_returns = (
                group_returns / combined_returns.std() * TARGET_VOL / np.sqrt(252)
            )
            return group_returns

    def asset_diagnostics(
        self,
        dataframe: pd.DataFrame,
        transaction_costs: pd.DataFrame,
        price_series: pd.DataFrame,
    ):
        positions = self.calc_returns(
            dataframe, transaction_costs, price_series, scale_to_target_vol=False
        )

        # TKR = "QC1 Comdty"
        # positions[positions.ticker == TKR].set_index("date")["net_return"].cumsum().plot(title=TKR)
        # positions[positions.ticker == TKR].set_index("date")["target"].cumsum().plot(title=TKR)

        positions["turnover"] = positions.groupby("ticker")["position"].diff().abs()

        # turnover = data.groupby("ticker")["turnover"].mean()
        # holding_time = 1 / turnover

        # alpha = net_pnl / turnover

    @classmethod
    def max_drawdown(cls, levels, in_percent=False):
        dd = cls.drawdown(levels, in_percent=in_percent)
        maxdd = dd.min()
        return maxdd

    @classmethod
    def drawdown(cls, series, in_percent=False):
        """
        Calculates drawdown given a time-series of price levels. When the price is at all time highs, the drawdown is 0.
        When prices are below high water marks (HWM), the drawdown is calculated as
        DD_t = S_t - HWM_t. If in_percent=True, then drawdown is computed as DD_t = S_t/HWM_t - 1.

        :param series: pd.Series, price levels
        :param in_percent: bool, default is False
        :return: pd.Series
        """

        # make a copy so that we don't modify original data
        drawdown = series.copy()

        # Fill NaN's with previous values
        drawdown = drawdown.fillna(method="ffill")

        # Ignore problems with NaN's in the beginning
        drawdown[np.isnan(drawdown)] = -np.Inf

        # Rolling maximum
        roll_max = np.maximum.accumulate(drawdown)
        drawdown = drawdown - roll_max

        # if need to display in drawdown in percent
        if in_percent:
            drawdown = drawdown / roll_max

        return drawdown

    def _plot_diagnostics(
        self,
        dataframe: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        diagnostics_dir: str,
        group_returns: pd.DataFrame = None,
    ):
        """
        Plot diagnostic visualizations for a given DataFrame.

        :param dataframe: The DataFrame containing the data to be analyzed.
        :type dataframe: pd.DataFrame
        :param portfolio_returns: The DataFrame containing the portfolio returns.
        :type portfolio_returns: pd.DataFrame
        :param diagnostics_dir: The directory where the diagnostic plots should be saved. The
                        plots will be saved as image files (e.g., PNG or JPEG) in this directory.
        :type diagnostics_dir: str
        :param group_returns: The DataFrame containing the group returns. Defaults to None.
        :type group_returns: pd.DataFrame, optional



        :return: None
        :rtype: None
        """
        plt.ioff()
        os.makedirs(os.path.join(diagnostics_dir), exist_ok=True)
        # lineplots
        fig, axis = plt.subplots(tight_layout=True)
        dataframe.plot(ax=axis, grid=True, legend=False)
        axis.set_xlabel(None)
        axis.set_title(self.name)
        fig.savefig(os.path.join(diagnostics_dir, "positions_lineplots.png"))
        plt.close(fig)
        # boxplots
        fig, axis = plt.subplots(figsize=(20, 30), tight_layout=True)
        sns.boxplot(data=dataframe, ax=axis, showfliers=False, orient="h")
        axis.grid()
        fig.savefig(os.path.join(diagnostics_dir, "positions_boxplots.png"))
        plt.close(fig)
        # missing data chart
        # fig, axis = plt.subplots(figsize=(20, 10), tight_layout=True)
        # matrix(dataframe, ax=axis, labels=True, label_rotation=90)
        # axis.set_title(self.name)
        # fig.savefig(os.path.join(diagnostics_dir, "missing_data.png"))
        # plt.close(fig)
        # PnL
        fig, axis = plt.subplots(tight_layout=True)
        annualised_sharpe = pd.concat(
            [
                portfolio_returns.groupby(portfolio_returns.index.year)[
                    "gross_return"
                ].apply(sharpe_ratio),
                portfolio_returns.groupby(portfolio_returns.index.year)[
                    "net_return"
                ].apply(sharpe_ratio),
            ],
            axis=1,
        ).rename(
            columns={
                "gross_return": "Gross Sharpe",
                "net_return": "Net Sharpe",
            }
        )
        annualised_sharpe.plot.bar(ax=axis, grid=True)
        axis.set_xlabel(None)
        axis.set_title(self.name)
        fig.savefig(os.path.join(diagnostics_dir, "annual_sharpe.png"))
        plt.close(fig)
        # annual Sharpe

        fig, axis = plt.subplots(tight_layout=True)

        gross_metrics = perf_metrics(portfolio_returns["gross_return"])
        net_metrics = perf_metrics(portfolio_returns["net_return"])
        portfolio_returns.cumsum().rename(
            columns={
                "gross_return": f"Gross PnL (Sharpe: {gross_metrics['sharpe_ratio']:.2f}, Calmar: {gross_metrics['calmar_ratio']:.2f})",
                "net_return": f"Net PnL (Sharpe: {net_metrics['sharpe_ratio']:.2f}, Calmar: {net_metrics['calmar_ratio']:.2f})",
            }
        ).plot(ax=axis, grid=True)
        axis.set_xlabel(None)
        axis.set_title(self.name)
        fig.savefig(os.path.join(diagnostics_dir, "pnl.png"))
        plt.close(fig)

        if group_returns is not None:
            fig, axis = plt.subplots(tight_layout=True)
            net_group_returns = group_returns.reset_index().pivot(
                index="date", columns="group", values="net_return"
            )
            net_group_returns.cumsum().plot(
                ax=axis,
                grid=True,
            )
            axis.legend(
                [
                    f"{ac} ({s:.2f})"
                    for ac, s in zip(
                        net_group_returns.columns, sharpe_ratio(net_group_returns)
                    )
                ]
            )
            axis.set_xlabel(None)
            axis.set_title(self.name + " -- net PnL (Sharpe)")
            fig.savefig(os.path.join(diagnostics_dir, "group_net_pnl.png"))
            plt.close(fig)


        if group_returns is not None:
            fig, axis = plt.subplots(tight_layout=True)
            gross_group_returns = group_returns.reset_index().pivot(
                index="date", columns="group", values="gross_return"
            )
            gross_group_returns.cumsum().plot(
                ax=axis,
                grid=True,
            )
            axis.legend(
                [
                    f"{ac} ({s:.2f})"
                    for ac, s in zip(
                        gross_group_returns.columns, sharpe_ratio(gross_group_returns)
                    )
                ]
            )
            axis.set_xlabel(None)
            axis.set_title(self.name + " -- gross PnL (Sharpe)")
            fig.savefig(os.path.join(diagnostics_dir, "group_gross_pnl.png"))
            plt.close(fig)

        plt.ion()

    def push_to_dashboard(
        self,
        asset_returns: pd.DataFrame,
        portfolio_returns: pd.DataFrame,
        dashboard_dir: str,
    ):
        """
        Push run results to a dashboard.

        :param asset_returns: The DataFrame containing the asset returns.
        :type asset_returns: pd.DataFrame
        :param portfolio_returns: The DataFrame containing the portfolio returns.
        :type portfolio_returns: pd.DataFrame
        :param dashboard_dir: The directory where the dashboard data should be saved.
        :type dashboard_dir: str


        :return: None
        """

        raise NotImplementedError("Dashboard functionality not yet implemented")

    # @job_helper.email_job_errors(
    #     os.path.basename(__file__), email_to=EMAIL_TO, email_success=True
    # )
    def main(
        self,
        diagnostics: bool = False,
        to_dashboard: bool = False,
        clear_cache: bool = False,
        skip_predictions: bool = False,
        ticker_groups: dict = {},
    ):
        """
        Make predictions using a model for a list of tickers and optionally run diagnostics.

        :param diagnostics: The directory where the predictions will be saved. Defaults to False.
        :type save_dir: bool, optional
        :param to_dashboard: Whether to push the results to the dashboard. Defaults to False.
        :type to_dashboard: bool, optional
        :param clear_cache: Whether to clear the cache. Defaults to False.
        :type clear_cache: bool, optional
        :param skip_predictions: Whether to skip the predictions. Defaults to False.
        :type skip_predictions: bool, optional


        :return: None
        :rtype: None
        """

        if skip_predictions:
            dataframe = pd.concat(
                [
                    pd.read_parquet(os.path.join(self.save_dir, f"{ticker}.parquet"))
                    for ticker in self.tickers
                    if os.path.exists(os.path.join(self.save_dir, f"{ticker}.parquet"))
                ],
                axis=1,
            )
        else:
            dataframe = self.make_predictions(
                clear_cache=clear_cache, save=True
            )

        if diagnostics:
            diagnostics_dir = os.path.abspath(
                os.path.join(self.save_dir, os.pardir, "diagnostics", self.name)
            )
            if self.data_subfolder:
                reference = pd.read_parquet(
                    os.path.join(self.load_dir, self.data_subfolder, "all")
                )[["ticker", "target", "vs_factor"]]
            else:
                reference = self.dataframe[
                    ["ticker", "target", "vs_factor"]
                ]

            transaction_costs = get_transaction_costs(self.transaction_cost_path)
            portfolio_returns = self.calc_portfolio_returns(
                dataframe, transaction_costs, reference
            )
            group_returns = self.calc_group_returns(
                dataframe, transaction_costs, reference, ticker_groups=ticker_groups
            )

            self._plot_diagnostics(
                dataframe, portfolio_returns, diagnostics_dir, group_returns
            )

        if to_dashboard:
            raise NotImplementedError("Dashboard functionality not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="lstm-simple-pinnacle-gross-futs-and-fx")
    parser.add_argument("--start_date", type=str, default="2000-01-01")
    parser.add_argument("--tickers", type=list, default=None)
    parser.add_argument("--diagnostics", type=bool, default=True)
    parser.add_argument("--dashboard", action="store_true", default=False)
    parser.add_argument("--clear_cache", action="store_true", default=False)
    parser.add_argument("--skip_predictions", action="store_true", default=False)
    parser.add_argument(
        "--variable_importance_mode", action="store_true", default=False
    )
    parser.add_argument("--output_signal_weights", action="store_true", default=False)
    parser.add_argument("--end_date", type=str, default=None)
    args, _ = parser.parse_known_args()

    t0 = time.time()
    cfg_path = os.path.join("mom_trans_torch", "configs", "backtest_settings", f"{args.name}.yaml")

    # load config
    with open(cfg_path, encoding="utf-8") as f:
        configs = yaml.safe_load(f)
    # is_fine_tune_model_settings = (
    #     "is_fine_tune_model" in configs and configs["is_fine_tune_model"]
    # )

    data_parquet = configs['data_parquet']
    

    # prepare list of tickers
    cfg_tickers = configs["universe"]
    if args.tickers is None:
        ticker_list = []
        ticker_groups = {}
        for tkr in cfg_tickers:
            ticker_list.append(tkr)
            ticker_groups[tkr] = cfg_tickers[tkr].get("group", None)
    else:
        ticker_list = args.tickers
        ticker_groups = {}


    save_directory = os.path.join(configs["save_path"], args.name)
    config_model = configs["model"]

    # run process
    job = MakePredictionsJob(
        args.name,
        ticker_list,
        args.start_date,
        data_parquet=data_parquet,
        save_dir=save_directory,
        cfg_model=config_model,
        transaction_cost_path=configs["ticker_reference_file"],
        variable_importance_mode=args.variable_importance_mode,
        output_signal_weights=args.output_signal_weights,
        end_date=args.end_date,
        data_subfolder=configs.get("data_subfolder", None),
        # live=configs.get("live", False),
        # x_curr_pairs_to_usd=x_curr_pairs_to_usd,
        # is_fine_tune_model=is_fine_tune_model_settings,
    )
    job.main(
        diagnostics=args.diagnostics and not args.output_signal_weights,
        to_dashboard=args.dashboard,
        clear_cache=args.clear_cache,
        skip_predictions=args.skip_predictions,
        ticker_groups=ticker_groups,
    )

    t1 = time.time()
    dt = t1 - t0
    print(f"Done! Elapsed: {round(t1 - t0, 2)} seconds.")
