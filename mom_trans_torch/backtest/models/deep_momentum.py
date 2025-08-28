"""Predictions for Deep Momentum model"""

import os
import re

from typing import List

import torch
import pandas as pd

from mom_trans_torch.configs.load import (
    load_settings_for_architecture,
    load_settings_for_finetune,
)
from mom_trans_torch.data.make_torch_dataset import (
    # prepare_segmented_dataset,
    MomentumDataset,
    # MomentumDatasetWithSegmentedContexts,
    # MomentumDatasetCrossSectionContexts,
    # CrossSectionDataset,
    # SignalCombineDataset,
    # correlation_features,
    # pca_features,
    datetime_embedding_global,
)

from mom_trans_torch.backtest.models.benchmark import AbstractModel
from mom_trans_torch.predict.trading_signal import (
    TradingSignalDmn,
    FIELDS_REQUIRED_FOR_PREDICT,
)


from mom_trans_torch.training.train import TrainMomentumTransformer
# from mom_trans_torch.training.fine_tune_training import TrainFineTune, SOURCE_MODEL_PARAMS


FINETUNE_KEEP_ORIGINAL_PARAMS = [
    "num_tickers",
    "num_tickers_full_univ",
]

YEAR_NO_END = 9999


class DeepMomentum(AbstractModel):
    """Deep Momentum model"""

    def __init__(
        self,
        train_yaml: str,
        architecture: str,
        top_n_seeds: int,
        seq_len: int,
        pre_loss_steps: int,
        # start_year,
        batch_size: int,
        # backtest_historical,
        prediction_folder: str = None,
        is_fine_tune_model: bool = False,
        drop_n_seeds_before_top_n: int = 0,
        use_first_n_valid_seeds: int = None,
    ):
        """
        :param train_yaml: Name of the yaml file containing the training settings
        :type train_yaml: str
        :param architecture: Name of the architecture
        :type architecture: str
        :param top_n_seeds: Number of seeds to use
        :type top_n_seeds: int
        :param seq_len: Sequence length
        :type seq_len: int
        :param pre_loss_steps: Number of pre-loss steps
        :type pre_loss_steps: int
        :param batch_size: Batch size
        :type batch_size: int
        :param prediction_folder: Folder to save predictions to, defaults to None
        :type prediction_folder: str, optional
        :param is_fine_tune_model: Whether the model is a fine-tuned model, defaults to False
        :type is_fine_tune_model: bool, optional
        """
        self.finetune_settings = {}
        self.finetune_yaml = train_yaml if is_fine_tune_model else None
        if is_fine_tune_model:
            self.finetune_settings = load_settings_for_finetune(self.finetune_yaml)
            train_yaml = self.finetune_settings["transfer_source_name"]
            self.train_settings = load_settings_for_architecture(
                train_yaml, architecture
            )
            self.model_path = os.path.join(
                self.train_settings["save_directory"],
                train_yaml,
                architecture,
                "fine-tuned",
                self.finetune_yaml,
            )
            self.source_save_path = os.path.join(
                self.train_settings["save_directory"],
                train_yaml,
                architecture,
            )

        else:
            self.train_settings = load_settings_for_architecture(
                train_yaml, architecture
            )
            self.model_path = os.path.join(
                self.train_settings["save_directory"], train_yaml, architecture
            )

        self.train_settings["seq_len"] = seq_len
        self.train_settings["pre_loss_steps"] = pre_loss_steps
        self.train_settings["batch_size"] = batch_size
        self.architecture = architecture
        self.top_n_seeds = top_n_seeds
        self.drop_n_seeds_before_top_n = drop_n_seeds_before_top_n
        self.seq_len = seq_len
        self.pre_loss_steps = pre_loss_steps
        self.batch_size = batch_size
        self.prediction_folder = prediction_folder
        self.use_first_n_valid_seeds = use_first_n_valid_seeds

        files = pd.Series(os.listdir(self.directory))

        self.model_start_years = (
            files[files.map(self.is_valid_year)].astype(int).sort_values()
        )

        # this is probably a bit dangerous...
        if not "extra_data_pre_steps" in self.train_settings.keys():
            self.train_settings["extra_data_pre_steps"] = 0

        # data_settings = load_data_settings(train_settings["data_settings"])

        super().__init__()

    @property
    def directory(self):
        return os.path.join(
            self.train_settings["save_directory"],
            self.train_settings["description"],
            self.train_settings["run_name"],
        )

    def get_model_directory(self, start_year):
        return os.path.join(
            self.directory,
            f"{start_year}",
        )

    def predict(
        self,
        data: pd.DataFrame,
        tickers: List[str],
        start_date: str,
        variable_importance_mode: bool = False,
        output_signal_weights: str = None,
    ):
        if variable_importance_mode:
            variable_importances_by_version = []
        elif output_signal_weights:
            positions_by_version = []
            # if not self.train_settings["signal_combine"]:
            #     raise ValueError(
            #         "Output signal weights only available for signal combine models"
            #     )
        else:
            positions_by_version = []

        start_years = self.model_start_years[
            self.model_start_years >= int(start_date[:4])
        ]
        # 9999 is to mimic None for end_year
        end_years = (
            self.model_start_years[self.model_start_years >= int(start_date[:4])]
            .shift(-1)
            .fillna(YEAR_NO_END)
            .astype(int)
        )

        # date_time_embedding = None
        # if (
        #     "use_correlation_features" in self.train_settings.keys()
        #     and self.train_settings["use_correlation_features"]
        # ):

        #     if (
        #         "use_principal_components" in self.train_settings.keys()
        #         and self.train_settings["use_principal_components"]
        #     ):
        #         corr_feat = pca_features(
        #             data,
        #             self.train_settings["use_correlation_features"],
        #             self.train_settings["use_principal_components"],
        #         )
        #     else:
        #         corr_feat = correlation_features(
        #             data,
        #             self.train_settings["use_correlation_features"],
        #             self.train_settings["correlation_span"],
        #         )

        if self.train_settings["date_time_embedding"]:
            datetime_embedding_global_max_length = self.train_settings[
                "datetime_embedding_global_max_length"
            ]
            date_time_embedding = datetime_embedding_global(
                self.train_settings["first_train_year"], data.index.year.max()
            )

            if (
                len(date_time_embedding) > datetime_embedding_global_max_length
            ) and not self.train_settings["local_time_embedding"]:
                raise ValueError(
                    f"Date time embedding too long: {len(date_time_embedding)} "
                    + "- please train with a higher max length or reduce the number of years in the dataset."
                )

        for start_year, end_year in zip(start_years, end_years):
            all_runs = pd.read_csv(
                os.path.join(self.get_model_directory(start_year), "all_runs.csv"),
                index_col=0,
            )
            if self.use_first_n_valid_seeds:
                all_runs = all_runs.dropna()
                run_number = all_runs.index.map(lambda s: int(s.split("-")[-1])) 
                last_valid_seed = run_number.sort_values()[:self.use_first_n_valid_seeds][-1]
                all_runs = all_runs[run_number <= last_valid_seed]

            run_settings = pd.read_json(
                os.path.join(
                    self.get_model_directory(start_year),
                    "settings",
                    f"{all_runs.index[0]}.json",
                ),
                typ="series",
            )

            if self.drop_n_seeds_before_top_n:
                all_runs = all_runs.iloc[self.drop_n_seeds_before_top_n :]

            # TODO: this is a bit of a hack - fix later (issue with transfer learning)
            run_settings["num_tickers"] = len(run_settings["tickers_dict"])

            tickers_dict = pd.Series(run_settings["tickers_dict"])
            tickers_dict.name = "ticker_id"
            tickers_dict.index.name = "ticker"

            # if self.train_settings["run_name"] == "X_TREND":
            #     if end_year == YEAR_NO_END:
            #         cache_folder = os.path.join(
            #             os.path.dirname(self.prediction_folder),
            #             "CACHED-SEGMENTED",
            #             os.path.basename(os.path.normpath(self.prediction_folder)),
            #         )

            #         if not os.path.exists(cache_folder):
            #             os.makedirs(cache_folder)

            #         if not os.path.exists(cache_folder):
            #             os.makedirs(cache_folder)

            #         ## ticker dict used to check is the same dataset...
            #         if (
            #             os.path.exists(
            #                 os.path.join(cache_folder, f"ticker_dict_{start_year}.pkl")
            #             )
            #             and tickers_dict.equals(
            #                 pd.read_pickle(
            #                     os.path.join(
            #                         cache_folder, f"ticker_dict_{start_year}.pkl"
            #                     )
            #                 )
            #             )
            #             and os.path.exists(
            #                 os.path.join(
            #                     cache_folder, f"segmented_data_{start_year}.pkl"
            #                 )
            #             )
            #         ):
            #             segmented_data = pd.read_pickle(
            #                 os.path.join(
            #                     cache_folder, f"segmented_data_{start_year}.pkl"
            #                 )
            #             )
            #         else:
            #             segmented_data = prepare_segmented_dataset(
            #                 self.train_settings,
            #                 data[data.ticker.isin(tickers_dict.index)],
            #                 min_seq_size=self.train_settings["contexts_min_seg_size"],
            #                 drop_last_observation=True,
            #                 # use_transaction_costs=use_transaction_costs,
            #                 cp_threshold=self.train_settings["cp_threshold"],
            #                 max_seq_size=self.train_settings["context_seq_len"],
            #             )
            #             segmented_data.to_pickle(
            #                 os.path.join(
            #                     cache_folder, f"segmented_data_{start_year}.pkl"
            #                 )
            #             )
            #             tickers_dict.to_pickle(
            #                 os.path.join(cache_folder, f"ticker_dict_{start_year}.pkl")
            #             )

            #     else:
            #         segmented_data = prepare_segmented_dataset(
            #             self.train_settings,
            #             data[data.ticker.isin(tickers_dict.index)],
            #             min_seq_size=self.train_settings["contexts_min_seg_size"],
            #             drop_last_observation=True,
            #             # use_transaction_costs=use_transaction_costs,
            #             cp_threshold=self.train_settings["cp_threshold"],
            #             max_seq_size=self.train_settings["context_seq_len"],
            #         )

            #     data_filtered = data[data.ticker.isin(tickers)]

            #     extra_dataset_params = {
            #         "segmented_data": segmented_data,
            #         "context_first_year": self.train_settings["first_train_year"],
            #         "context_end_year": start_year,
            #     }

            #     test_data = MomentumDatasetWithSegmentedContexts(  # pylint: disable=no-value-for-parameter
            #         self.train_settings,
            #         data_filtered[data_filtered.ticker.isin(tickers_dict.index)],
            #         first_year=start_year,
            #         end_year=end_year,
            #         tickers_dict=tickers_dict,
            #         test_set=True,
            #         live_mode=True,
            #         date_time_embedding=date_time_embedding**extra_dataset_params,
            #     )
            # elif self.train_settings["signal_combine"]:
            #     extra_dataset_params = {}
            #     test_data = SignalCombineDataset(
            #         self.train_settings,
            #         data[data.ticker.isin(tickers_dict.index.tolist())],
            #         first_year=start_year,
            #         end_year=end_year,
            #         tickers_dict=tickers_dict,
            #         test_set=True,
            #         live_mode=True,
            #         date_time_embedding=date_time_embedding,
            #         **extra_dataset_params,
            #     )
            # elif self.train_settings["cross_section"]:
            #     extra_dataset_params = {}
            #     # data_filtered = data[data.ticker.isin(tickers)]
            #     if (
            #         "use_correlation_features" in self.train_settings.keys()
            #         and self.train_settings["use_correlation_features"]
            #     ):
            #         extra_dataset_params["corr_features"] = corr_feat
            #         if (
            #             "use_principal_components" in self.train_settings
            #             and self.train_settings["use_principal_components"]
            #         ):
            #             extra_dataset_params["use_principal_components"] = True
            #     test_data = CrossSectionDataset(
            #         self.train_settings,
            #         data[data.ticker.isin(tickers_dict.index.tolist())],
            #         first_year=start_year,
            #         end_year=end_year,
            #         tickers_dict=tickers_dict,
            #         test_set=True,
            #         live_mode=True,
            #         date_time_embedding=date_time_embedding,
            #         **extra_dataset_params,
            #     )

            # else:

            # NOTE: only filtering for target tickers
            data_filtered = data[data.ticker.isin(tickers)]
            test_data = MomentumDataset(
                self.train_settings,
                data_filtered,
                first_year=start_year,
                end_year=end_year,
                tickers_dict=tickers_dict,
                test_set=True,
                live_mode=True,
                # date_time_embedding=date_time_embedding,
            )

            for i in range(self.top_n_seeds):
                # model_save_path = os.path.join(
                #     self.get_model_directory(start_year),
                #     "models",
                #     f"{all_runs.index[i]}",
                # )

                # if self.train_settings["run_name"] == "X_TREND":
                #     # raise NotImplementedError("Contexts not implemented yet")
                #     data_params_file_path = os.path.join(
                #         self.model_path,
                #         str(start_year),
                #         "data-params",
                #         all_runs.index[i] + ".pkl",
                #     )

                #     data_params = pd.read_pickle(data_params_file_path)
                #     test_data.shuffle_context(data_params["context_random_state"])

                run_settings = pd.read_json(
                    os.path.join(
                        self.get_model_directory(start_year),
                        "settings",
                        f"{all_runs.index[i]}.json",
                    ),
                    typ="series",
                )
                # TODO: this is a bit of a hack - fix later (issue with transfer learning)
                run_settings["num_tickers"] = len(run_settings["tickers_dict"])
                extra_settings = {}

                fields_required_for_predict = list(
                    set(FIELDS_REQUIRED_FOR_PREDICT).intersection(run_settings.keys())
                )
                if self.finetune_yaml:
                    raise NotImplementedError("Finetuning not implemented yet")
                    # source_save_path = os.path.join(
                    #     self.source_save_path,
                    #     f"v{version}",
                    # )
                    # source_model_kwargs = (
                    #     pd.read_json(
                    #         os.path.join(
                    #             source_save_path,
                    #             f"results-{start_year}.json",
                    #         ),
                    #         typ="series",
                    #     )
                    #     .loc[SOURCE_MODEL_PARAMS]
                    #     .to_dict()
                    # )
                    # settings_to_check = {**run_settings, **source_model_kwargs}

                    # if "original_override" in self.finetune_settings.keys():
                    #     for k, v in self.finetune_settings["original_override"].items():
                    #         if k in self.finetune_settings.keys():
                    #             assert settings_to_check[k] == v

                    # extra_settings["source_model_kwargs"] = source_model_kwargs
                    # extra_settings["source_save_path"] = os.path.join(
                    #     source_save_path, str(start_year)
                    # )
                    # fields_required_for_predict = list(
                    #     set(fields_required_for_predict) - set(SOURCE_MODEL_PARAMS)
                    # )

                kwargs = {k: run_settings[k] for k in fields_required_for_predict}

                # if self.train_settings["signal_combine"]:
                #     kwargs["input_fields"] = test_data.input_fields
                #     kwargs["weight_fields"] = test_data.weight_fields
                #     kwargs["num_inputs_ticker"] = test_data.num_inputs_ticker
                #     kwargs["num_to_weight"] = test_data.num_to_weight

                # if output_signal_weights:
                #     kwargs["output_signal_weights"] = output_signal_weights

                # if (
                #     isinstance(test_data, MomentumDatasetCrossSectionContexts)
                #     and "target_tickers" in kwargs
                # ):
                #     kwargs["target_tickers"] = test_data.target_ticker_embed_mapping # pylint: disable=no-member

                # if (
                #     isinstance(test_data, MomentumDatasetCrossSectionContexts)
                #     and "context_set_filter" in kwargs
                # ):
                #     kwargs["context_ticker_embed_mapping"] = (
                #         test_data.context_ticker_embed_mapping # pylint: disable=no-member
                #     )

                dmn = TradingSignalDmn(
                    self.architecture,
                    test_data,
                    num_features=len(self.train_settings["features"]),
                    model_save_path=self.get_model_directory(start_year),
                    parent_class=(
                        TrainMomentumTransformer
                        # if not self.finetune_settings
                        # else TrainFineTune
                    ),
                    extra_settings=extra_settings,
                    **self.train_settings,
                    **kwargs,
                )
                # if variable_importance_mode:
                #     variable_importances_by_version.append(
                #         dmn.variable_importance(self.batch_size).assign(version=version)
                #     )
                # elif output_signal_weights:
                #     positions_by_version.append(
                #         dmn.predict(all_runs.index[i], self.batch_size).assign(
                #             version=i
                #         )
                #     )

                # else:
                positions_by_version.append(
                    dmn.predict(all_runs.index[i], self.batch_size).assign(
                        version=i
                    )
                )

        # if variable_importance_mode:
        #     if self.train_settings["cross_section"]:
        #         raise NotImplementedError("TODO")
        #     return (
        #         pd.concat(variable_importances_by_version)
        #         .rename(
        #             columns=dict(
        #                 zip(
        #                     range(len(self.train_settings["features"])),
        #                     self.train_settings["features"],
        #                 )
        #             )
        #         )
        #         .drop(columns="version")
        #         .groupby(["date", "ticker"])
        #         .mean()
        #         .reset_index()
        #         .set_index("date")
        #     )

        positions = pd.concat(positions_by_version)

        if self.train_settings["cross_section"]:
            positions = positions[positions.ticker.isin(tickers)]

        if output_signal_weights:
            return (
                positions.drop(columns="version")
                .reset_index()
                .rename(columns={"index": "date"})
                .groupby(["date", "ticker",])
                .mean()
                .reset_index()
                # .rename(columns={"level_0": "date"})
                .set_index("date")
            )

        return (
            positions.groupby([positions.index, positions.ticker])["position"]
            .mean()
            .unstack()
            # .reset_index()
            # .set_index("")
        )

    @classmethod
    def is_valid_year(cls, year: int) -> bool:
        """Check if year is valid

        :param year: Year to check
        :type year: str
        :return: Whether the year is valid
        :rtype: bool
        """
        return bool(re.match(r"^(19|20)\d{2}$", year))
