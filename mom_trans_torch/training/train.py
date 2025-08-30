

import argparse
import datetime as dt
import itertools
import logging
import os

from typing import List
import numpy as np
import pandas as pd
import torch
import wandb
import torch.utils.data
from empyrical import sharpe_ratio
from torch import nn
from mom_trans_torch.utils.logging_utils import get_logger


from mom_trans_torch.data.make_torch_dataset import unpack_torch_dataset, MomentumDataset
from mom_trans_torch.models.common import LossFunction
from mom_trans_torch.models.dmn import QUANTILES, DeepMomentumNetwork, DmnMode
from mom_trans_torch.models.mom_trans import MomentumTransformer
# from mom_trans_torch.models.x_mom_trans import XMomentumTransformer
from mom_trans_torch.models.lstm_dmn import LstmBaseline, LstmSimple
from mom_trans_torch.models.temporal_fusion_transformer import TFTStyleTransformer, TFTBaseline
from mom_trans_torch.models.PatchTST import PatchTST
from mom_trans_torch.models.PatchTST2 import PatchTST2
# from mom_trans_torch.models.PatchCopy import PatchTSTTS
from mom_trans_torch.models.PatchCopy2 import PatchTSTTS2
# from mom_trans_torch.models.PsLSTM import Patch_xLSTM
# from mom_trans_torch.models.xLSTM import xLSTM
# from mom_trans_torch.models.Informer import CausalInformerBaseline
from mom_trans_torch.models.DLinear import CausalDLinearBaseline
from mom_trans_torch.models.NLinear import NLinearBaseline
from mom_trans_torch.models.mamba import MambaBaseline, Mamba2Baseline
from mom_trans_torch.models.iTransformer import iTransformer
from mom_trans_torch.models.iTransformer2 import iTransformer2
# from mom_trans_torch.models.TimeMixer import TimeMixer
# from mom_trans_torch.models.FEDformer import FedFormer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_PREDICTIONS = False


class TrainMomentumTransformer:
    logger = get_logger(__name__)

    def __init__(
        self,
        train_data: MomentumDataset,
        valid_data: MomentumDataset,
        test_data: MomentumDataset,
        seq_len: int,
        num_features: int,
        save_path: int,
        # test_sharpe: float = np.NINF,
        # best_valid_sharpe: float = np.NINF,
        # test_sharpe_net: float = np.NINF,
        train_extra_data: List[MomentumDataset] = None,  # List - typing
        compile=False,
        **kwargs,
    ):
        """
        :param train_data: Training dataset
        :type train_data: MomentumDataset
        :param valid_data: Validation dataset
        :type valid_data: MomentumDataset
        :param test_data: Test dataset
        :type test_data: MomentumDataset
        :param seq_len: Sequence length
        :type seq_len: int
        :param num_features: Number of features
        :type num_features: int
        :param save_path: Model save path
        :type save_path: str
        :param train_extra_data: Additional training datasets
        :type train_extra_data: List[MomentumDataset]
        :param compile: Compile model
        :type compile: bool
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        """

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_extra_data = train_extra_data

        self.seq_len = seq_len
        self.num_features = num_features

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(os.path.join(save_path, "models")):
            os.makedirs(os.path.join(save_path, "models"))
        if not os.path.exists(os.path.join(save_path, "returns")):
            os.makedirs(os.path.join(save_path, "returns"))
        if not os.path.exists(os.path.join(save_path, "settings")):
            os.makedirs(os.path.join(save_path, "settings"))
        if not os.path.exists(os.path.join(save_path, "data-params")):
            os.makedirs(os.path.join(save_path, "data-params"))
        self._save_path = save_path

        self.compile = compile
        # self.test_sharpe = test_sharpe
        # self.best_valid_sharpe = best_valid_sharpe
        # self.test_sharpe_net = test_sharpe_net

    def model_save_path(self, run_name: str) -> str:
        """
        Current model save path.

        :param run_name: Run name
        :type run_name: str
        :return: Current save path
        :rtype: str

        """
        return os.path.join(self._save_path, "models", run_name)

    def settings_path(self, run_name: str) -> str:
        """
        Settings path.

        :param run_name: Run name
        :type run_name: str
        :return: Settings path
        :rtype: str

        """
        return os.path.join(self._save_path, "settings", run_name + ".json")

    def data_params_path(self, run_name: str) -> str:
        """
        Data params path.

        :param run_name: Run name
        :type run_name: str
        :return: Data params path
        :rtype: str

        """
        return os.path.join(self._save_path, "data-params", run_name + ".pkl")

    def returns_save_path(self, run_name: str) -> str:
        """
        Returns save path.

        :param run_name: Run name
        :type run_name: str
        :return: Returns save path
        :rtype: str
        """
        return os.path.join(self._save_path, "returns", run_name + ".csv")

    def _load_architecture(
        self,
        architecture: str,
        input_dim: int,
        num_tickers: int,
        optimise_loss_function: int,
        **kwargs,
    ) -> torch.nn.Module:
        """
        Load the specified architecture.

        :param architecture: Architecture name
        :type architecture: str
        :param input_dim: Input dimension
        :type input_dim: int
        :param num_tickers: Number of tickers
        :type num_tickers: int
        :param optimise_loss_function: Loss function to optimise
        :type optimise_loss_function: int
        :return: Model
        :rtype: nn.Module
        """
        # if architecture == "MOM_TRANS_SIMPLE":
        #     model = MomentumTransformerSimple(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )
        if architecture == "LSTM":
            model = LstmBaseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )
        elif architecture == "LSTM_SIMPLE":
            model = LstmSimple(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "TemporalFusionTransformer":
            model = TFTStyleTransformer(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "MOM_TRANS":
            model = MomentumTransformer(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        # elif architecture == "X_MOM_TRANS":
        #     model = XMomentumTransformer(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        elif architecture == "TemporalFusionTransformer_Base":
            model = TFTBaseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "PatchTST":
            model = PatchTST(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "PatchTST2":
            model = PatchTST2(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "PatchCopy":
            model = PatchTSTTS(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "PatchCopy2":
            model = PatchTSTTS2(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        # elif architecture == "PxLSTM":
        #     model = Patch_xLSTM(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        # elif architecture == "xLSTM":
        #     model = xLSTM(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )
        
        elif architecture == "PxLSTM":
            model = Patch_xLSTM(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "xLSTM":
            model = xLSTM(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )
            
        elif architecture == "iTransformer":
            model = iTransformer(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "iTransformer2":
            model = iTransformer2(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        # elif architecture == "TimeMixer":
        #     model = TimeMixer(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        # elif architecture == "Informer":
        #     model = CausalInformerBaseline(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        # elif architecture == "FEDFormer":
        #     model = FedFormer(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        elif architecture == "DLinear":
            model = CausalDLinearBaseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "NLinear":
            model = NLinearBaseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "Mamba":
            model = MambaBaseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        elif architecture == "Mamba2":
            model = Mamba2Baseline(
                input_dim=input_dim,
                num_tickers=num_tickers,
                optimise_loss_function=optimise_loss_function,
                **kwargs,
            )

        # elif architecture == "TimeMixer":
        #     model = TimeMixerBaseline(
        #         input_dim=input_dim,
        #         num_tickers=num_tickers,
        #         optimise_loss_function=optimise_loss_function,
        #         **kwargs,
        #     )

        else:
            raise ValueError("Architecture not recognised")

        return model.to(device)

    def run(
        self,
        architecture: str,
        lr: float,
        batch_size: int,
        optimise_loss_function: int,
        max_gradient_norm: int,
        iterations: int,
        early_stopping: int,
        wandb_run_name: str,
        # contexts_segmented=False,
        fineturn_after_train=False,
        log_wandb: bool = False,
        **kwargs,
    ):
        """
        Run training.

        :param architecture: Architecture name
        :type architecture: str
        :param lr: Learning rate
        :type lr: float
        :param batch_size: Batch size
        :type batch_size: int
        :param optimise_loss_function: Loss function to optimise
        :type optimise_loss_function: int
        :param max_gradient_norm: Maximum gradient norm
        :type max_gradient_norm: float
        :param iterations: Number of iterations
        :type iterations: int
        :param early_stopping: Early stopping
        :type early_stopping: int
        :param wandb_run_name: Run name
        :type wandb_run_name: str
        :param fineturn_after_train: Finetune after training
        :type fineturn_after_train: bool
        :param kwargs: Additional keyword arguments
        :type kwargs: dict
        :return: Test sharpe, best validation sharpe, test sharpe net
        :rtype: float, float, float
        """
        assert optimise_loss_function in [
            LossFunction.SHARPE.value,
            LossFunction.JOINT_GAUSS.value,
            LossFunction.JOINT_QRE.value,
        ]
        assert isinstance(iterations, int)
        assert isinstance(max_gradient_norm, float)
        assert isinstance(early_stopping, int)
        assert isinstance(batch_size, int)
        input_dim = self.num_features
        kwargs = kwargs.copy()
        if "num_tickers" not in kwargs:
            kwargs["num_tickers"] = self.valid_data.num_tickers

        model = self._load_architecture(
            architecture=architecture,
            input_dim=input_dim,
            optimise_loss_function=optimise_loss_function,
            **kwargs,
        )

        best_valid_sharpe = -np.inf
        test_sharpe = -np.inf

        if self.compile:
            model = torch.compile(model, dynamic=True)
            # model = torch.compile(model, backend="eager")
            # model = torch.compile(model)

        best_valid_sharpe = self.train_and_validate(
            model,
            lr,
            batch_size,
            max_gradient_norm,
            optimise_loss_function,
            iterations,
            early_stopping,
            log_wandb,
            wandb_run_name,
            best_valid_sharpe,
            **kwargs,
        )

        # TODO remove this
        if fineturn_after_train:
            self.logger.info(
                "----Finetune---- (from best valid = %.2f)", best_valid_sharpe
            )
            model.load_state_dict(torch.load(self.model_save_path(wandb_run_name)))

            # unfreeze source dmn
            for param in model.parameters():
                param.requires_grad = True

            best_valid_sharpe = self.train_and_validate(
                model,
                kwargs["extra_fine_tune_lr"],
                batch_size,
                max_gradient_norm,
                optimise_loss_function,
                iterations,
                # NOTE: 2x early stopping
                2 * early_stopping,
                log_wandb,
                wandb_run_name,
                best_valid_sharpe,
                **kwargs,
            )

        # TODO include mode for taining with no test
        predictions, test_sharpe, test_sharpe_net = self.predict(
            model,
            self.model_save_path(wandb_run_name),
            batch_size,
            # use_contexts,
            optimise_loss_function,
            DmnMode.INFERENCE,
        )

        train_loss_metric = (
            "Sharpe" if optimise_loss_function == LossFunction.SHARPE.value else "Loss"
        )

        self.logger.info("----Iteration update----")
        self.logger.info("Best Valid %s: %.3f", train_loss_metric, best_valid_sharpe)
        self.logger.info("Test Sharpe Gross: %.3f", test_sharpe)
        self.logger.info("Test Sharpe Net: %.3f", test_sharpe_net)

        # if best_valid_sharpe > self.best_valid_sharpe:
        #     self.best_valid_sharpe = best_valid_sharpe
        #     self.test_sharpe = test_sharpe
        #     self.test_sharpe_net = test_sharpe_net

        if SAVE_PREDICTIONS:
            predictions.to_csv(self.returns_save_path(wandb_run_name))

        return test_sharpe, best_valid_sharpe, test_sharpe_net

    def train_and_validate(
        self,
        model,
        lr,
        batch_size,
        max_gradient_norm,
        optimise_loss_function,
        iterations,
        early_stopping,
        log_wandb: bool,
        wandb_run_name: str,
        best_valid_sharpe=-np.inf,
        **kwargs,
    ):
        use_contexts = kwargs["use_contexts"]
        cross_section = kwargs["cross_section"]
        train_loss_metric = (
            "Sharpe" if optimise_loss_function == LossFunction.SHARPE.value else "Loss"
        )

        if "valid_force_sharpe_loss" in kwargs and kwargs["valid_force_sharpe_loss"]:
            valid_force_sharpe_loss = True
        else:
            valid_force_sharpe_loss = False

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        early_stopping_counter = 0

        for iteration in range(iterations):
            self.logger.info(f"Iteration {iteration + 1}")

            train_sharpes = []
            # count = 0
            # early = len(self.train_data) / BATCH_SIZE // seq_len

            if use_contexts and not cross_section:
                self.train_data.shuffle_context()
                self.valid_data.shuffle_context()

            model.train()
            for _, samples in enumerate(
                torch.utils.data.DataLoader(
                    (
                        self.train_data
                        if not self.train_extra_data
                        else torch.utils.data.ConcatDataset(
                            [self.train_data, *self.train_extra_data]
                        )
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                ),
            ):
                train_loss = model(
                    **unpack_torch_dataset(
                        samples,
                        self.train_data,
                        device,
                        use_dates_mask=False,
                        live_mode=False,
                    ),
                    mode=DmnMode.TRAINING,
                )

                optimizer.zero_grad()
                train_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                optimizer.step()

                train_sharpes.append(train_loss.negative().detach().item())
                # count += 1
                # if TRAIN_SHARPE_ALL_TIMESTEPS and count > early:
                #     break

            train_sharpe = np.mean(train_sharpes)

            # TODO loggging
            self.logger.info("Train %s: %.3f", train_loss_metric, train_sharpe)

            valid_sharpes = []
            # valid_indexes = []
            # valid_dates = []

            # if VAL_DIVERSIFIED_SHARPE:
            model.eval()
            valid_loader = torch.utils.data.DataLoader(
                self.valid_data, batch_size=batch_size, drop_last=False
            )
            with torch.no_grad():
                for _, samples in enumerate(valid_loader):
                    loss_valid = model(
                        **unpack_torch_dataset(
                            samples,
                            self.valid_data,
                            device,
                            use_dates_mask=False,
                            live_mode=False,
                        ),
                        mode=DmnMode.TRAINING,
                        force_sharpe_loss=valid_force_sharpe_loss,
                    )

                    # valid_positions.append(captured_return.detach().cpu())
                    valid_sharpes.append(loss_valid.negative().detach().item())

            # valid_positions = torch.cat(valid_positions).flatten().numpy()
            # iteration_valid_sharpe = sharpe_ratio(valid_positions)
            iteration_valid_sharpe = np.mean(valid_sharpes)

            if log_wandb:
                wandb.log(
                    {
                        "train_loss": train_sharpe,
                        "valid_loss": iteration_valid_sharpe,
                        # "epoch": iteration,
                    }
                )

            self.logger.info(
                "Valid %s: %.3f", train_loss_metric, iteration_valid_sharpe
            )

            if iteration_valid_sharpe >= best_valid_sharpe:
                torch.save(model.state_dict(), self.model_save_path(wandb_run_name))
                best_valid_sharpe = iteration_valid_sharpe
                early_stopping_counter = 0
                # break #todo REMOVE
            else:
                early_stopping_counter += 1
                if early_stopping_counter == early_stopping:
                    break
        return best_valid_sharpe

    def predict(
        self,
        model: DeepMomentumNetwork,
        model_save_path: str,
        batch_size: int,
        # use_contexts: bool,
        optimise_loss_function: int,
        mode: DmnMode,
    ):
        """
        Predictions.

        :param model: Model
        :type model: DeepMomentumNetwork
        :param model_save_path: Model save path
        :type model_save_path: str
        :param batch_size: Batch size
        :type batch_size: int
        :param optimise_loss_function: Loss function to optimise
        :type optimise_loss_function: int
        :param mode: Mode
        :type mode: DmnMode
        :return: Predictions, test sharpe, test sharpe net
        :rtype: pd.DataFrame, float, float
        """
        assert mode in [DmnMode.INFERENCE, DmnMode.LIVE]
        assert optimise_loss_function in [
            LossFunction.SHARPE.value,
            LossFunction.JOINT_GAUSS.value,
            LossFunction.JOINT_QRE.value,
        ]

        # if mode is DmnMode.LIVE:
        #     raise NotImplementedError("TODO Live mode not implemented")

        test_captured_returns = []
        test_positions = []
        test_mask = []
        test_dates = []
        test_tickers = []
        # test_se = []
        test_pred_mean = []
        test_pred_std = []
        test_quantile_vals = []
        test_all_targets = []
        test_vol_scaling_amount = []
        test_vol_scaling_amount_prev = []

        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        output_signal_weight_mode = (
            # model.is_signal_combine and model.output_signal_weights
            model.output_signal_weights
        )
        with torch.no_grad():
            for _, samples in enumerate(
                torch.utils.data.DataLoader(
                    self.test_data, batch_size=batch_size, drop_last=False
                )
            ):
                torch_dataset = unpack_torch_dataset(
                    samples,
                    self.test_data,
                    device,
                    use_dates_mask=True,
                    live_mode=False,
                )
                results = model(
                    **torch_dataset,
                    mode=mode,
                )

                if mode is DmnMode.LIVE:
                    if (
                        optimise_loss_function == LossFunction.SHARPE.value
                        or output_signal_weight_mode
                    ):
                        positions = results
                        test_positions.append(positions.detach().cpu())

                    elif optimise_loss_function == LossFunction.JOINT_QRE.value:
                        positions, quantile_results = results
                        test_positions.append(positions.detach().cpu())
                        test_quantile_vals.append(
                            torch.flatten(quantile_results, start_dim=0, end_dim=1)
                            .detach()
                            .cpu()
                        )
                    elif optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                        positions, pred_mean, pred_std = results
                        test_positions.append(positions.detach().cpu())
                        test_pred_mean.append(pred_mean.detach().cpu())
                        test_pred_std.append(pred_std.detach().cpu())
                else:
                    if optimise_loss_function == LossFunction.SHARPE.value:
                        captured_return, positions = results

                        test_captured_returns.append(captured_return.detach().cpu())
                        test_positions.append(positions.detach().cpu())
                    elif optimise_loss_function == LossFunction.JOINT_QRE.value:
                        captured_return, positions, quantile_results = results
                        test_captured_returns.append(captured_return.detach().cpu())
                        test_positions.append(positions.detach().cpu())
                        test_quantile_vals.append(
                            torch.flatten(quantile_results, start_dim=0, end_dim=1)
                            .detach()
                            .cpu()
                        )
                    elif optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                        captured_return, positions, pred_mean, pred_std = results
                        test_captured_returns.append(captured_return.detach().cpu())
                        test_positions.append(positions.detach().cpu())
                        test_pred_mean.append(pred_mean.detach().cpu())
                        test_pred_std.append(pred_std.detach().cpu())

                date_mask = torch_dataset["date_mask"]
                target_tickers = torch_dataset["target_tickers"]
                dates = torch_dataset["dates"]
                if (
                    model.is_cross_section
                    or model.is_multitask
                    or model.is_signal_combine
                ):
                    date_mask = model.combine_batch_and_asset_dim(date_mask, 3)
                    target_tickers = model.combine_batch_and_asset_dim(
                        target_tickers, 2
                    )
                    dates = model.combine_batch_and_asset_dim(dates, 3)
                # elif model.is_signal_combine:
                #     # ddates need to broadcast
                #     date_mask = model.combine_batch_and_asset_dim(date_mask, 2)
                #     target_tickers = model.combine_batch_and_asset_dim(
                #         target_tickers, 2
                #     )
                #     dates = model.combine_batch_and_asset_dim(dates, 2)

                # will not be in live mode...
                if mode is not DmnMode.LIVE:
                    target_y = torch_dataset["target_y"]
                    vol_scaling_amount = torch_dataset["vol_scaling_amount"]
                    vol_scaling_amount_prev = torch_dataset["vol_scaling_amount_prev"]

                    if (
                        model.is_cross_section
                        or model.is_multitask
                        or model.is_signal_combine
                    ):
                        if model.is_signal_combine:
                            target_y = target_y.swapaxes(1, 2).unsqueeze(-1)
                        target_y = model.combine_batch_and_asset_dim(target_y, 4)
                        vol_scaling_amount = model.combine_batch_and_asset_dim(
                            vol_scaling_amount, 3
                        )
                        vol_scaling_amount_prev = model.combine_batch_and_asset_dim(
                            vol_scaling_amount, 3
                        )
                        # trans_cost_bp = self.combine_batch_and_asset_dim(trans_cost_bp, 3)

                    test_all_targets.append(
                        target_y[:, -positions.shape[1] :, -1].detach().cpu()
                    )

                    test_vol_scaling_amount.append(
                        vol_scaling_amount[:, -positions.shape[1] :].detach().cpu()
                    )
                    test_vol_scaling_amount_prev.append(
                        vol_scaling_amount_prev[:, -positions.shape[1] :].detach().cpu()
                    )

                test_mask.append(date_mask[:, -positions.shape[1] :].detach().cpu())
                test_tickers.append(target_tickers.detach().cpu())

                if (
                    model.is_cross_section
                    or model.is_multitask
                    or model.is_signal_combine
                ):
                    test_dates.append(
                        self.test_data.dates_string[
                            dates[:, -positions.shape[1] :]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(int)
                        ]
                    )
                else:
                    # TODO get to the bottom of this - seems unnecessary
                    test_dates.append(np.array([*dates]).T[:, -positions.shape[1] :])

            test_mask = torch.cat(test_mask).flatten().numpy()
            test_dates = np.concatenate(test_dates).flatten()
            test_tickers = torch.cat(test_tickers).flatten().numpy()

            tick_mapping = self.test_data.tickers_from_idx_dict

            test_tickers = sum(
                [[tick_mapping[t]] * positions.shape[1] for t in test_tickers],
                [],
            )

            if output_signal_weight_mode:
                test_positions = torch.cat(test_positions).numpy()
                shape = test_positions.shape
                test_positions = test_positions.reshape(shape[0] * shape[1], shape[2])

                if shape[2] == 1:
                    return pd.DataFrame(
                        {
                            "ticker": test_tickers,
                            "dmn_signal": test_positions.flatten(),
                        },
                        index=test_dates,
                    )[test_mask]

                # todo map tuple to feature...
                return pd.DataFrame(
                    {
                        "ticker": test_tickers,
                        **{
                            f: test_positions[:, i].tolist()
                            for i, f in enumerate(self.test_data.weight_fields)
                        },
                    },
                    index=test_dates,
                )[test_mask]

            test_positions = torch.cat(test_positions).flatten().numpy()

            if mode is DmnMode.LIVE:
                return pd.DataFrame(
                    {
                        "ticker": test_tickers,
                        "position": test_positions,
                    },
                    index=test_dates,
                )[test_mask]
            else:
                test_vol_scaling_amount = (
                    torch.cat(test_vol_scaling_amount).flatten().numpy()
                )
                test_vol_scaling_amount_prev = (
                    torch.cat(test_vol_scaling_amount_prev).flatten().numpy()
                )
                test_all_targets = torch.cat(test_all_targets).flatten().numpy()
                test_captured_returns = (
                    torch.cat(test_captured_returns).flatten().numpy()
                )
                predictions_by_asset = pd.DataFrame(
                    {
                        "ticker": test_tickers,
                        "captured_return": test_captured_returns,
                        "position": test_positions,
                        "target_return": test_all_targets,
                        "cost_scaling": test_vol_scaling_amount,
                        "cost_scaling_prev": test_vol_scaling_amount_prev,
                    },
                    index=test_dates,
                )[test_mask]

            predictions_by_asset = (
                predictions_by_asset.reset_index()
                .merge(
                    self.test_data.ticker_ref[["ticker", "transaction_cost"]],
                    on="ticker",
                )
                .rename(columns={"index": ""})
                .set_index("")
            )

            predictions_by_asset["cost_scaling"] *= (
                predictions_by_asset["transaction_cost"].values * 1e-4
            )
            predictions_by_asset["cost_scaling_prev"] *= (
                predictions_by_asset["transaction_cost"].values * 1e-4
            )

            predictions_by_asset["pos_prev"] = (
                predictions_by_asset.groupby("ticker")["position"].shift(1).fillna(0.0)
            )
            predictions_by_asset["cost"] = (
                predictions_by_asset["cost_scaling"] * predictions_by_asset["position"]
                - predictions_by_asset["cost_scaling_prev"]
                * predictions_by_asset["pos_prev"]
            ).abs()

            predictions_by_asset["captured_return_net"] = (
                predictions_by_asset["captured_return"] - predictions_by_asset["cost"]
            )

            if optimise_loss_function == LossFunction.JOINT_QRE.value:
                extra_results = pd.DataFrame(
                    torch.cat(test_quantile_vals).numpy(), columns=QUANTILES
                )
                extra_results.index = test_dates
                extra_results = extra_results[test_mask].copy()
                extra_results.index = predictions_by_asset.index
                predictions_by_asset = pd.concat(
                    [predictions_by_asset, extra_results], axis=1
                )

            elif optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                # raise NotImplementedError("TODO")

                # elif OPTIMISE == LossFunction.JOINT_GAUSS.value:
                test_pred_mean = torch.cat(test_pred_mean).flatten().numpy()
                test_pred_std = torch.cat(test_pred_std).flatten().numpy()

                extra_results = pd.DataFrame(
                    {
                        "pred_mean": test_pred_mean,
                        "pred_std": test_pred_std,
                    },
                    index=test_dates,
                )[test_mask]
                extra_results.index = predictions_by_asset.index
                predictions_by_asset = pd.concat(
                    [predictions_by_asset, extra_results], axis=1
                )

            # if mode is DmnMode.LIVE:
            #     # TODO maybe deal with this up higher
            #     return predictions_by_asset.drop(columns="target_return")

            predictions_portfolio = predictions_by_asset.copy()
            predictions_portfolio.index.name = "idx"
            # # use mean because not every day has same number of tickers..
            predictions_portfolio = predictions_portfolio.groupby("idx")[
                ["captured_return", "captured_return_net"]
            ].sum()
            # iteration_test_sharpe = sharpe_ratio(diversified)

            # undiversified
            test_sharpe_net = sharpe_ratio(predictions_portfolio["captured_return_net"])
            predictions_portfolio = predictions_portfolio["captured_return"]
            test_sharpe = sharpe_ratio(predictions_portfolio)

        return predictions_by_asset, test_sharpe, test_sharpe_net

    def variable_importance(
        self,
        model: DeepMomentumNetwork,
        model_save_path: str,
        batch_size: int,
    ):
        """
        Predictions.

        :param model: Model
        :type model: DeepMomentumNetwork
        :param model_save_path: Model save path
        :type model_save_path: str
        :param batch_size: Batch size
        :type batch_size: int
        :param optimise_loss_function: Loss function to optimise
        :type optimise_loss_function: int
        :param mode: Mode
        :type mode: DmnMode

        """
        importances = []
        test_tickers = []
        test_dates = []
        test_mask = []

        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        with torch.no_grad():
            for _, samples in enumerate(
                torch.utils.data.DataLoader(
                    self.test_data, batch_size=batch_size, drop_last=False
                )
            ):
                torch_dataset = unpack_torch_dataset(
                    samples,
                    self.test_data,
                    device,
                    use_dates_mask=True,
                    live_mode=False,
                )
                results = model.variable_importance(
                    **torch_dataset,
                )
                results = results[:, -self.test_data.pred_steps :]

                importances.append(results.detach().cpu())

                date_mask = torch_dataset["date_mask"]
                target_tickers = torch_dataset["target_tickers"]
                dates = torch_dataset["dates"]

                # will not be in live mode...

                test_tickers.append(target_tickers.detach().cpu())
                # TODO get to the bottom of this - seems unnecessary
                test_dates.append(np.array([*dates]).T[:, -results.shape[1] :])
                test_mask.append(date_mask[:, -results.shape[1] :].detach().cpu())

            test_mask = torch.cat(test_mask).flatten().numpy()
            test_dates = np.concatenate(test_dates).flatten()
            test_tickers = torch.cat(test_tickers).flatten().numpy()

            tick_mapping = self.test_data.tickers_from_idx_dict
            test_tickers = sum(
                [[tick_mapping[t]] * results.shape[1] for t in test_tickers],
                [],
            )

            importances = torch.cat(importances).flatten(start_dim=0, end_dim=1).numpy()

        return pd.DataFrame(
            importances,
            index=pd.MultiIndex.from_arrays(
                [test_dates, test_tickers], names=["date", "ticker"]
            ),
            # columns=self.test_data.feature_names,
        )[test_mask]

