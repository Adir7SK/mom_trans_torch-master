"""Job for tuning hyperparameters of DMN models."""

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
from colorama import Fore, Style
from mom_trans_torch.data.build_features import deep_momentum_strategy_features, prepare_features
from mom_trans_torch.configs.load import (
    ARCHITECTURES,
    # load_data_settings,
    load_settings_for_architecture,
    load_sweep_settings,
)
from mom_trans_torch.data.make_torch_dataset import (
    # MomentumDatasetWithRandomContexts,
    MomentumDataset,
    calc_vs_factor_scaler,
    calc_trans_cost_scaler,
    get_transaction_costs,
    datetime_embedding_global,
)

from mom_trans_torch.training.hp_tuner import Tuner
from mom_trans_torch.utils.logging_utils import get_logger

from joblib import Parallel, delayed


def main(
    settings: dict,
    data_parquet: str,
    sweep_settings: dict,
    architecture: str,
    valid_end_optional: int = None,
    end_date: str = None,
    compile: bool = False,
    skip_first_window: bool = False,
    filter_start_years: list = None,
    num_workers: int = 1,
):
    """Main function for hyperparameter tuning.

    :param settings: settings dictionary
    :param data_parquet: data parquet
    :param sweep_settings: sweep settings dictionary
    :param architecture: architecture name
    :param valid_end_optional: optionally change the end of the valid period to earlier
    :param compile: compile model
    :param skip_first_window: skip first window
    :param filter_start_years: filter start years
    :param num_workers: number of workers
    """
    settings = copy.deepcopy(settings)
    # data_settings = copy.deepcopy(data_settings)
    data_file_path = os.path.join("data", data_parquet)

    save_directory = settings["save_directory"]

    logger = get_logger(__name__)

    if settings["test_run"]:
        logger.info(
            Fore.RED
            + "WARNING: In test mode and only five assets will be loaded"
            + Style.RESET_ALL
        )
    data_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
        data_file_path)
    data = pd.read_parquet(data_file_path)
    if end_date:
        data = data.loc[:end_date]

    if settings["test_run"]:
        data = data[data.ticker.isin(data.ticker.unique()[:5])]

    data = prepare_features(data)

    if "ticker_subset" in settings.keys() and len(settings["ticker_subset"]) > 0:
        data = data[data.ticker.isin(settings["ticker_subset"])]

    # for version in versions:
    # test_period_sharpes = []
    # test_period_sharpes_net = []
    windows = [
        (settings["first_train_year"], s, e)
        for s, e in zip(
            settings["test_start_years"],
            settings["test_start_years"][1:] + [settings["final_test_year"] + 1],
        )
        if (filter_start_years is None or s in filter_start_years)
    ][::-1]

    if skip_first_window:
        windows = windows[1:]

    def run_tuner(window):
        train_start, test_start, test_end = window
        logger.info("Configuration: %s", settings["description"])
        logger.info("Test window: %s-%s", test_start, test_end)
        logger.info("Creating datasets...")

        train_extra_data = []

        data_ex_test = data[
            (pd.to_datetime(data.index).year >= train_start)
            & (pd.to_datetime(data.index).year < test_start)
        ]

        vs_factor_scaler = calc_vs_factor_scaler(data_ex_test)
        ticker_ref = get_transaction_costs(settings["ticker_reference_file"])
        trans_cost_scaler = calc_trans_cost_scaler(
            ticker_ref[ticker_ref["ticker"].isin(data_ex_test.ticker.unique())]
        )
        scalers = {
            "vs_factor_scaler": vs_factor_scaler,
            "trans_cost_scaler": trans_cost_scaler,
        }

        shift_valid = valid_end_optional and valid_end_optional < test_start

        dataset_class = MomentumDataset
        extra_dataset_params = {}

        if settings["date_time_embedding"]:
            datetime_embedding_global_max_length = settings[
                "datetime_embedding_global_max_length"
            ]
            extra_dataset_params["date_time_embedding"] = datetime_embedding_global(
                train_start, data.index.year.max()
            )

            if (
                len(extra_dataset_params["date_time_embedding"])
                > datetime_embedding_global_max_length
            ) and not settings["local_time_embedding"]:
                raise ValueError(
                    f"Date time embedding too long: {len(extra_dataset_params['date_time_embedding'])} "
                    + "- please train with a higher max length or reduce the number of years in the dataset."
                )

            # extra_dataset_params["date_time_embedding"] = True

        valid_data = dataset_class(
            settings=settings,
            data=data,
            first_year=train_start,
            end_year=valid_end_optional if shift_valid else test_start,
            drop_first_perc=settings["train_valid_split"],
            target_override=settings["valid_target_override"],
            **extra_dataset_params,
        )

        if settings["signal_combine"]:
            extra_dataset_params["input_fields"] = valid_data.input_fields
            extra_dataset_params["weight_fields"] = valid_data.weight_fields

        if settings["use_contexts"] and settings["cross_section"]:
            settings["num_context"] = valid_data.num_contexts
        # elif settings["cross_section"]:
        #     settings["num_context"] = len(valid_data.tickers_dict)

        # TODO - input_features
        train_data = dataset_class(
            settings=settings,
            data=data,
            first_year=train_start,
            end_year=valid_end_optional if shift_valid else test_start,
            keep_first_perc=settings["train_valid_split"],
            tickers_dict=valid_data.tickers_dict,
            target_override=settings["train_target_override"],
            **extra_dataset_params,
        )

        if shift_valid:
            train_extra_data.append(
                dataset_class(
                    settings=settings,
                    data=data,
                    first_year=valid_end_optional,
                    end_year=test_start,
                    tickers_dict=valid_data.tickers_dict,
                    target_override=settings["train_target_override"],
                    **extra_dataset_params,
                )
            )

        test_data = dataset_class(
            settings=settings,
            data=data,
            first_year=test_start,
            end_year=test_end,
            tickers_dict=valid_data.tickers_dict,
            test_set=True,
            **extra_dataset_params,
        )

        data_params = {}
        if settings["use_contexts"] and not settings["cross_section"]:
            data_params["context_random_state"] = test_data.context_random_state

        logger.info("Done.")

        tuner = Tuner(
            {**settings, **scalers},
            sweep_settings,
            architecture,
            train_data,
            valid_data,
            test_data,
            # version,
            test_start,
            test_end,
            train_extra_data,
            data_params,
            valid_end_optional,
            compile=compile,
        )

        tuner.hyperparameter_optimisation()

        # (
        #     best_test_sharpe,
        #     best_test_sharpe_net,
        #     _,
        # ) = tuner.hyperparameter_optimisation()

    if num_workers > 1:
        Parallel(n_jobs=num_workers)(delayed(run_tuner)(window) for window in windows)
    else:
        for window in windows:
            run_tuner(window)

        # test_period_sharpes.append(best_test_sharpe)
        # test_period_sharpes_net.append(best_test_sharpe_net)

    # with open(
    #     os.path.join(
    #         save_directory,
    #         settings["description"],
    #         settings["run_name"],
    #         "results-all.json",
    #     ),
    #     "w",
    #     encoding="utf-8",
    # ) as file:
    #     json.dump(
    #         {
    #             "test_sharpe_entire": np.mean(test_period_sharpes),
    #             "test_sharpe_net_entire": np.mean(test_period_sharpes_net),
    #         },
    #         file,
    #         indent=4,
    #     )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argument Parser Hyperparameter Tuning"
    )
    parser.add_argument(
        "-r",
        "--run_file_name",
        type=str,
        default="pinnacle-gross-futs-and-fx",
        help="Name of YAML file",
    )
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="PatchTST", # , iTransformer, X_MOM_TRANS, PatchCopy,TimeMixer, iTransformer, xLSTM, PxLSTM, MOM_TRANS, LSTM_SIMPLE, LSTM, TemporalFusionTransformer, TemporalFusionTransformer_Base, PatchTST, NLinear, DLinear, Informer
        help="Architecture name",
        choices=ARCHITECTURES,
    )
    parser.add_argument(
        "-rs",
        "--restart",
        type=str,
        default=None,
        help='Restart format "<VERSION>_<TEST_START_YEAR>_<TEST_END_YEAR>"',
    )

    parser.add_argument(
        "-veo",
        "--valid_end_optional",
        type=int,
        default=None,
        help="Optionally change the end of the valid period to earlier",
    )

    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        default=None,
        help="Specify an end-date for backtest",
    )
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Compile model",
        default=False,
    )

    parser.add_argument(
        "-sfw",
        "--skip_first_window",
        action="store_true",
        help="Skip first window",
        default=False,
    )

    parser.add_argument(
        "-fsy",
        "--filter_start_years",
        type=int,
        nargs="+",
        default=None,
        help="Filter start years",
    )

    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=1,
        help="Number of workers",
    )

    args = parser.parse_args()

    loaded_settings = load_settings_for_architecture(args.run_file_name, args.arch)

    data_parquet = loaded_settings["data_parquet"]

    loaded_sweep_settings = load_sweep_settings(loaded_settings["sweep_yaml"])

    main(
        loaded_settings,
        data_parquet,
        loaded_sweep_settings,
        args.arch,
        # args.restart,
        args.valid_end_optional,
        args.end_date,
        compile=args.compile,
        skip_first_window=args.skip_first_window,
        filter_start_years=args.filter_start_years,
        num_workers=args.num_workers,
    )
