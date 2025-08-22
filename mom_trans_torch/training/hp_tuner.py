
import json
import os
import pickle
import copy

import numpy as np
import pandas as pd

# import optuna
import wandb

from mom_trans_torch.models.common import LossFunction
from mom_trans_torch.training.train import TrainMomentumTransformer
from mom_trans_torch.utils.logging_utils import get_logger
from mom_trans_torch.configs.settings import WANDB_ENTITY


CONFIGS_IGNORE_WANDB = [
    "project",
    "data_yaml",
    # "save_directory",
    "ticker_reference_file",
    "test_run",
    # "train_valid_split",
    # "seq_len",
    # "pre_loss_steps",
    # "date_time_embedding",
    # "early_stopping",
    # "iterations",
    # "random_search_max_iterations",
    # "optimise_loss_function",
    # "use_transaction_costs",
    # "fixed_trans_cost_bp_loss",
    # "volscale_tc_loss",
    # "assume_same_leverage_for_prev",
    # "turnover_regulariser_scaler",
    # "use_static_ticker",
    # "num_context",
    # "context_seq_len",
    # "cp_threshold",
    # "contexts_min_seg_size",
    # "cpd_max_segment_len",
    # "cpd_segment_threshold_lbw",
    # "features",
    "feature_prepare",
    "versions",
    "first_train_year",
    "final_test_year",
    "first_test_year",
    "reference_cols",
    "target",
    "targets",
    "target_vol",
    "description",
    "run_name",
    # "use_contexts",
    "extra_data_pre_steps",
    "vs_factor_scaler",
    "trans_cost_scaler",
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class Tuner:
    logger = get_logger(__name__)

    def __init__(
        self,
        settings,
        sweep_settings,
        architecutre,
        train_data,
        valid_data,
        test_data,
        # version,
        test_start,
        test_end,
        train_extra_data,
        data_params,
        valid_end_optional: None,
        compile=False,
    ) -> None:
        self.best_test_sharpe, self.best_valid_sharpe = -np.inf, -np.inf
        self.iteration = 0

        self.settings = settings
        self.architecture = architecutre
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        # self.version = version
        self.test_start = test_start
        self.test_end = test_end
        self.data_params = data_params
        self.valid_end_optional = valid_end_optional
        # (
        #     self.best_test_sharpe,
        #     self.best_valid_sharpe,
        #     self.best_test_sharpe_net,
        # ) = (np.NINF, np.NINF, np.NINF)
        self.sweep_settings = copy.deepcopy(sweep_settings)
        self.sweep_settings = {
            "name": f"{self.settings['description']}_{self.architecture}_{self.test_start}",
            **sweep_settings,
        }
        # default_sweep_settings = {
        #     "name": f"{self.settings['description']}_{self.architecture}_{self.test_start}",
            # Add other default keys and values here
        # }
        # self.sweep_settings = {**default_sweep_settings, **sweep_settings}
        self.save_path = os.path.join(
            self.settings["save_directory"],
            self.settings["description"],
            self.settings["run_name"],
            # f"v{version}",
            f"{test_start}",
        )
        self.model_training = TrainMomentumTransformer(
            train_data,
            valid_data,
            test_data,
            self.settings["seq_len"],
            len(self.settings["features"]),
            save_path=self.save_path,
            train_extra_data=train_extra_data,
            # test_sharpe=self.test_sharpe,
            compile=compile,
        )

    def hyperparameter_optimisation(self):
        # TODO make abstract and include version for optuna search instead of random grid
        # (
        #     self.best_test_sharpe,
        #     self.best_valid_sharpe,
        #     self.best_test_sharpe_net,
        # ) = (np.NINF, np.NINF, np.NINF)

        self.iteration = 0
        # optuna_path = self.settings["save_directory"].replace("\\", "/")

        #### THE FOLLOWING MUST NOT BE IN PRODUCTION !!
        os.environ["WANDB_API_KEY"] = "998fed3425bcbcfc5bf7eba464e4b35d40d4624c"

        if not wandb.api.api_key:
            wandb_key = (
                {"key": os.environ["WANDB_API_KEY"]}
                if "WANDB_API_KEY" in os.environ
                else {}
            )
            wandb.login(**wandb_key)

        api = wandb.Api()
        runs = api.runs(
            f"{WANDB_ENTITY}/{self.settings['project']}",
            filters={"group": self.sweep_settings["name"]},
            per_page=10000,
        )
        if len(runs) > 0:
            # TODO logic to check if sweep is finished or num runs to go
            # runs_df = pd.DataFrame([run.config for run in runs])
            # names = pd.Series([run.name for run in runs])
            # is_finished = [run.state == "finished" for run in runs]
            # finished_names = names[is_finished]
            sweep_id = runs[0].sweep.id
            assert all(x.sweep.id == sweep_id for x in runs)
            num_completed_runs = len([x for x in runs if x.state == "finished"])
            if num_completed_runs >= self.settings["random_search_max_iterations"]:
                return self.save_results_to_drive(sweep_id)
        else:

            sweep_id = wandb.sweep(
                sweep=self.sweep_settings,
                project=self.settings["project"],
                entity=WANDB_ENTITY,
            )
            num_completed_runs = 0

        # wandb.init(
        #     # project=self.settings["description"] + "_" + self.architecture,
        #     project=self.settings["project"],
        #     entity=ENTITY,
        #     group=self.sweep_settings["name"],
        #     # track hyperparameters and run metadata
        # )
        
        # TODO - input_features
        fixed_config = {
            "model": self.settings["description"],
            # "dataset": self.settings["data_yaml"],
            "architecture": self.architecture,
            "test_start": self.test_start,
            "train_start": self.settings["first_train_year"],
            "valid_end_optional": self.valid_end_optional,
            "train_tickers": self.valid_data.tickers_dict.index.tolist(),
            **{k: v for k, v in self.settings.items() if k not in CONFIGS_IGNORE_WANDB},
        }

        if self.settings["signal_combine"]:
            fixed_config["input_fields"] = self.valid_data.input_fields
            fixed_config["weight_fields"] = self.valid_data.weight_fields
            self.settings["num_inputs_ticker"] = self.valid_data.num_inputs_ticker
            self.settings["num_to_weight"] = self.valid_data.num_to_weight

    
        # for it, hp in random_search.iterrows():
        def objective(config: dict = None) -> float:
            with wandb.init(
                group=self.sweep_settings["name"],
                config=config,
            ):
                # If called by wandb.agent, as below,
                # this config will be set by Sweep Controller

                config = wandb.config
                # if "pair" in config:
                #     pair = config["pair"]
                #     config["patch_len"] = pair["patch_len"]
                #     config["stride"] = pair["stride"]
                name = wandb.run.name

                hp = dict(config.items())

                if "max_gradient_norm" in hp:
                    hp["max_gradient_norm"] = float(hp["max_gradient_norm"])

                for k, v in fixed_config.items():
                    config[k] = v

                # comment = (
                #     f"{self.architecture} {self.test_start}-{self.test_end} "
                #     + " ".join(f"{k}={v}" for k, v in hp.items())
                # )

                self.iteration = self.iteration + 1
                self.logger.info(
                    "----Random grid search iteration %s----", self.iteration
                )
                self.logger.info("----%s----", self.settings["description"])
                # self.logger.info(comment)

                (
                    test_sharpe,
                    valid_sharpe,
                    test_sharpe_net,
                ) = self.model_training.run(
                    architecture=self.architecture,
                    log_wandb=True,
                    wandb_run_name=name,
                    **self.settings,
                    **hp,
                )

                # if valid_sharpe > self.best_valid_sharpe:
                #     (
                #         self.best_valid_sharpe,
                #         self.best_test_sharpe,
                #         self.best_test_sharpe_net,
                #     ) = (
                #         valid_sharpe,
                #         test_sharpe,
                #         test_sharpe_net,
                #     )
                with open(
                    self.model_training.settings_path(name),
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        {
                            "test_sharpe": test_sharpe,
                            "test_sharpe_net": test_sharpe_net,
                            (
                                "valid_sharpe"
                                if self.settings["optimise_loss_function"]
                                == LossFunction.SHARPE.value
                                else "valid_loss"
                            ): valid_sharpe,
                            **hp,
                            # **COMMON_PARAMS,
                            **self.settings,
                            **{
                                "tickers_dict": self.valid_data.tickers_dict.to_dict(),
                                # "vs_scaling_factor_scaler": self.valid_data.vs_factor_scaler.data_max_[
                                #     0
                                # ],
                                # "trans_cost_scaler": self.valid_data.trans_cost_scaler.data_max_[
                                #     0
                                # ],
                                "num_tickers": self.train_data.num_tickers,
                                "num_tickers_full_univ": self.valid_data.num_tickers_full_univ,
                            },
                            # **{"features": FEATURES},
                            # **data_params,
                        },
                        file,
                        indent=4,
                        cls=NpEncoder,
                    )

                with open(
                    self.model_training.data_params_path(name),
                    "wb",
                ) as handle:
                    pickle.dump(
                        self.data_params,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                # do this
                wandb.log(
                    {
                        "test_sharpe_gross": test_sharpe,
                        "test_sharpe_net": test_sharpe_net,
                        "valid_loss_best": valid_sharpe,
                    },
                    step=None,
                )

        # # return valid_sharpe
        wandb.agent(
            f"{WANDB_ENTITY}/{self.settings['project']}/{sweep_id}",
            function=objective,
            count=self.settings["random_search_max_iterations"] - num_completed_runs,
        )

        return self.save_results_to_drive(sweep_id)

    def save_results_to_drive(self, sweep_id):
        sweep = wandb.Api().sweep(f"{WANDB_ENTITY}/{self.settings['project']}/{sweep_id}")
        runs = [r for r in sweep.runs if r.state == "finished"]

        def check_keys_in_dict(keys, dictionary):
            for key in keys:
                if key not in dictionary:
                    return False
            return True

        all_runs = pd.concat(
            [
                pd.Series(
                    run.summary._json_dict,  # pylint: disable=protected-access
                    name=run.name,
                ).loc[["valid_loss_best", "test_sharpe_gross", "test_sharpe_net"]]
                for run in runs
                if check_keys_in_dict(
                    ["valid_loss_best", "test_sharpe_gross", "test_sharpe_net"],
                    run.summary._json_dict,  # pylint: disable=protected-access
                )
            ],
            axis=1,
        ).T.sort_values("valid_loss_best", ascending=False)
        best_runs = all_runs.head(self.settings["top_n_seeds"])

        all_runs.to_csv(
            os.path.join(
                self.save_path,
                "all_runs.csv",
            )
        )

        best_runs.to_csv(
            os.path.join(
                self.save_path,
                "best_runs.csv",
            )
        )

        best_means = best_runs.mean()

        best_means.to_json(
            os.path.join(
                self.save_path,
                "best_runs_mean.json",
            ),
            indent=4,
        )

        wandb.finish()
        return (
            best_means.loc["test_sharpe_gross"],
            best_means.loc["test_sharpe_net"],
            best_means.loc["valid_loss_best"],
        )
