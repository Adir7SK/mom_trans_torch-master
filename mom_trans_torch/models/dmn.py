from abc import ABC, ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.distributions as D
import torch.nn.functional as F

from torch import nn

from mom_trans_torch.models.common import (
    VariableSelectionNetwork,
    DropoutNoScaling,
    LossFunction,
    GateAddNorm,
    GatedResidualNetwork,
    positional_encoding,
    rolling_average_conv,
)

from enum import Enum

QUANTILES = [
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.99,
]


JOINT_SCALING_FACTOR_GAUSSIAN = 1.0
JOINT_SCALING_FACTOR_QR = 2.0
JOINT_ADDITIONAL_SCALING_FACTOR_XDMN = 4.0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def quantile_loss(y, y_pred, quantile):
    error = y - y_pred
    q_loss = torch.max(quantile * error, (1.0 - quantile) * -error)

    return q_loss.mean()


class DmnMode(Enum):
    TRAINING = 1
    INFERENCE = 2
    LIVE = 3


class DoNothing(nn.Module):
    """Do nothing module"""

    def __init__(self):
        # Call the parent class's constructor
        super(DoNothing, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return itself

        :param x: input tensor
        :type x: torch.Tensor
        :return: torch.Tensor
        :rtype: input tensor
        """
        return x


class RNNSingleOutput(nn.Module):
    """Custom module to extract just the RNN output"""

    def __init__(self, input_size, hidden_size, batch_first=False):
        super(RNNSingleOutput, self).__init__()
        # self.rnn = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=batch_first)

    def forward(self, x):
        # Forward pass through the RNN
        rnn_output, _ = self.rnn(x)
        return rnn_output


class SequenceRepresentationSimple(nn.Module):
    """Converts sequence aand ticker to a representation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_tickers: int,
        fuse_encoder_input: bool = False,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,  # TODO maybe diable this for now...
        use_prescaler=False,
        **kwargs,
    ):
        """_summary_

        :param input_dim: _description_
        :type input_dim: int
        :param hidden_dim: _description_
        :type hidden_dim: int
        :param dropout: _description_
        :type dropout: float
        :param num_tickers: _description_
        :type num_tickers: int
        :param fuse_encoder_input: _description_, defaults to False
        :type fuse_encoder_input: bool, optional
        :param use_static_ticker: _description_, defaults to True
        :type use_static_ticker: bool, optional
        :param auto_feature_num_input_linear: _description_, defaults to 0
        :type auto_feature_num_input_linear: int, optional
        """

        super(SequenceRepresentationSimple, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_tickers = num_tickers
        self.use_static_ticker = use_static_ticker
        self.use_prescaler = use_prescaler

        if auto_feature_num_input_linear:
            raise NotImplementedError("TODO")

        if fuse_encoder_input:
            raise NotImplementedError("TODO")

        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.hidden_dim)

            self.static_context_variable_selection = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )

            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
        if self.use_prescaler:
            self.prescaler = nn.Linear(input_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim, hidden_dim)
        else:
            self.lstm = nn.LSTM(self.input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: torch.Tensor,
        static_tickers: torch.Tensor,
        encoder_representation: Union[torch.Tensor, None] = None,
        automatic_features: Union[torch.Tensor, None] = None,
        variable_importance: bool = False,
        pos_encoding_batch: Union[torch.Tensor, None] = None,
    ):
        sequence = sequence.swapaxes(0, 1)
        if self.use_prescaler:
            sequence = self.prescaler(sequence)
    
        if self.use_static_ticker:
            ticker_embedding = self.ticker_embedding(static_tickers)

        if variable_importance:
            raise NotImplementedError("TODO")
        #     _, importance = self.vsn(
        #         inputs,
        #         (
        #             self.static_context_variable_selection(ticker_embedding)
        #             if self.use_static_ticker
        #             else None
        #         ),
        #     )
        #     return importance

        # vsn_out, _ = self.vsn(
        #     inputs,
        #     (
        #         self.static_context_variable_selection(ticker_embedding)
        #         if self.use_static_ticker
        #         else None
        #     ),
        # )

        if pos_encoding_batch is not None:
            sequence = sequence + pos_encoding_batch.swapaxes(0, 1)
            # vsn_out = vsn_out + pos_encoding_batch.swapaxes(0, 1)

        if self.use_static_ticker:
            hidden_state, _ = self.lstm(
                sequence,
                (
                    self.static_context_state_h(ticker_embedding).unsqueeze(
                        0
                    ),  # .unsqueeze(1).repeat(1, seq_len, 1),
                    self.static_context_state_c(ticker_embedding).unsqueeze(
                        0
                    ),  # .unsqueeze(1).repeat(1, seq_len, 1),
                ),
            )
        else:
            hidden_state, _ = self.lstm(sequence)

        hidden_state = self.dropout(hidden_state)

        return hidden_state.swapaxes(0, 1)


class SequenceRepresentation(nn.Module):
    """Converts sequence aand ticker to a representation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float,
        num_tickers: int,
        fuse_encoder_input: bool = False,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,  # TODO maybe diable this for now...
        **kwargs,
    ):
        """_summary_

        :param input_dim: _description_
        :type input_dim: int
        :param hidden_dim: _description_
        :type hidden_dim: int
        :param dropout: _description_
        :type dropout: float
        :param num_tickers: _description_
        :type num_tickers: int
        :param fuse_encoder_input: _description_, defaults to False
        :type fuse_encoder_input: bool, optional
        :param use_static_ticker: _description_, defaults to True
        :type use_static_ticker: bool, optional
        :param auto_feature_num_input_linear: _description_, defaults to 0
        :type auto_feature_num_input_linear: int, optional
        """

        super(SequenceRepresentation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_tickers = num_tickers
        self.encoder_input = fuse_encoder_input
        self.use_static_ticker = use_static_ticker
        if fuse_encoder_input:
            self.combine_layer = GatedResidualNetwork(
                2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout
            )

        if auto_feature_num_input_linear:
            prescalers = nn.ModuleDict(
                dict(
                    [
                        (
                            str(i),
                            nn.Linear(1, hidden_dim),
                        )
                        for i in range(auto_feature_num_input_linear)
                    ]
                    + [
                        (
                            str(i),
                            DoNothing(),
                        )  # swap batch and time
                        for i in range(auto_feature_num_input_linear, input_dim)
                    ]
                )
            )
            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers=prescalers,
            )

        else:
            prescalers = {}

            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers=prescalers,
            )

        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.hidden_dim)

            self.static_context_variable_selection = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )

            self.static_context_enrichment = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )

        self.lstm = nn.LSTM(hidden_dim, hidden_dim)  # , batch_first=True)

        self.gate_add_norm_ltsm = GateAddNorm(hidden_dim, dropout=dropout)
        self.grn = GatedResidualNetwork(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            context_size=hidden_dim if self.use_static_ticker else None,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        static_tickers: torch.Tensor,
        encoder_representation: Union[torch.Tensor, None] = None,
        automatic_features: Union[torch.Tensor, None] = None,
        variable_importance: bool = False,
        pos_encoding_batch: Union[torch.Tensor, None] = None,
    ):
        if self.use_static_ticker:
            ticker_embedding = self.ticker_embedding(static_tickers)

        if automatic_features:
            # sequence = sequence[:, -automatic_features[0].shape[1] :]

            # inputs = dict(
            #     [
            #         (
            #             str(i),
            #             sequence[:, :, i : (i + 1)].swapaxes(0, 1),
            #         )  # swap batch and time
            #         for i in range(sequence.shape[-1])
            #     ]
            #     + list(
            #         zip(
            #             [str(i) for i in range(sequence.shape[-1], self.input_dim)],
            #             [af.swapaxes(0, 1) for af in automatic_features],
            #         )
            #     )
            # )

            # if variable_importance:
            #     _, importance = self.vsn(
            #         inputs,
            #         self.static_context_variable_selection(ticker_embedding)
            #         if self.use_static_ticker
            #         else None,
            #     )
            #     return importance

            # vsn_out, _ = self.vsn(
            #     inputs,
            #     self.static_context_variable_selection(ticker_embedding)
            #     if self.use_static_ticker
            #     else None,
            # )
            raise NotImplementedError("TODO")

        else:
            inputs = dict(
                [
                    (
                        str(i),
                        sequence[:, :, i : (i + 1)].swapaxes(0, 1),
                    )  # swap batch and time
                    for i in range(self.input_dim)
                ]
            )

            if variable_importance:
                _, importance = self.vsn(
                    inputs,
                    (
                        self.static_context_variable_selection(ticker_embedding)
                        if self.use_static_ticker
                        else None
                    ),
                )
                return importance

            vsn_out, _ = self.vsn(
                inputs,
                (
                    self.static_context_variable_selection(ticker_embedding)
                    if self.use_static_ticker
                    else None
                ),
            )

        if pos_encoding_batch is not None:
            vsn_out = vsn_out + pos_encoding_batch.swapaxes(0, 1)

        if self.encoder_input:
            vsn_out = self.combine_layer(
                torch.cat([vsn_out, encoder_representation.swapaxes(0, 1)], dim=-1)
            )

        if self.use_static_ticker:
            hidden_state, _ = self.lstm(
                vsn_out,
                (
                    self.static_context_state_h(ticker_embedding).unsqueeze(
                        0
                    ),  # .unsqueeze(1).repeat(1, seq_len, 1),
                    self.static_context_state_c(ticker_embedding).unsqueeze(
                        0
                    ),  # .unsqueeze(1).repeat(1, seq_len, 1),
                ),
            )
        else:
            hidden_state, _ = self.lstm(vsn_out)

        hidden_state = self.gate_add_norm_ltsm(hidden_state, vsn_out)

        ffn_output = self.grn(
            hidden_state,
            (
                self.static_context_enrichment(ticker_embedding)
                if self.use_static_ticker
                else None
            ),
        )

        ffn_output = ffn_output.swapaxes(0, 1)
        hidden_state = hidden_state.swapaxes(0, 1)

        return ffn_output, hidden_state


class SequenceRepresentationXLSTM(nn.Module):
    """Converts sequence and ticker to a representation using xLSTM"""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float,
            num_tickers: int,
            fuse_encoder_input: bool = False,
            use_static_ticker: bool = True,
            auto_feature_num_input_linear=0,
            xlstm_expansion_factor: int = 2,
            **kwargs,
    ):
        """
        :param input_dim: Number of input features
        :param hidden_dim: Size of hidden layers
        :param dropout: Dropout rate
        :param num_tickers: Number of unique ticker symbols
        :param fuse_encoder_input: Whether to fuse encoder representation
        :param use_static_ticker: Whether to use static ticker embeddings
        :param auto_feature_num_input_linear: Number of automatic features to process
        :param xlstm_expansion_factor: Expansion factor for xLSTM intermediate dimensions
        """

        super(SequenceRepresentationXLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.num_tickers = num_tickers
        self.encoder_input = fuse_encoder_input
        self.use_static_ticker = use_static_ticker
        self.xlstm_expansion_factor = xlstm_expansion_factor

        if fuse_encoder_input:
            self.combine_layer = GatedResidualNetwork(
                2 * hidden_dim, hidden_dim, hidden_dim, dropout=dropout
            )

        # Variable Selection Network setup
        if auto_feature_num_input_linear:
            prescalers = nn.ModuleDict(
                dict(
                    [
                        (
                            str(i),
                            nn.Linear(1, hidden_dim),
                        )
                        for i in range(auto_feature_num_input_linear)
                    ]
                    + [
                        (
                            str(i),
                            DoNothing(),
                        )
                        for i in range(auto_feature_num_input_linear, input_dim)
                    ]
                )
            )
            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers=prescalers,
            )
        else:
            prescalers = {}
            self.vsn = VariableSelectionNetwork(
                dict([(str(i), self.hidden_dim) for i in range(input_dim)]),
                self.hidden_dim,
                dropout=self.dropout,
                context_size=self.hidden_dim,
                prescalers=prescalers,
            )

        # Ticker embedding setup
        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(self.num_tickers, self.hidden_dim)
            self.static_context_variable_selection = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_enrichment = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_h = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )
            self.static_context_state_c = GatedResidualNetwork(
                self.hidden_dim, self.hidden_dim, self.hidden_dim, dropout=self.dropout
            )

        # Replace LSTM with xLSTM
        self.xlstm = self._create_xlstm(hidden_dim, hidden_dim)

        self.gate_add_norm_lstm = GateAddNorm(hidden_dim, dropout=dropout)
        self.grn = GatedResidualNetwork(
            hidden_dim,
            hidden_dim,
            hidden_dim,
            dropout=dropout,
            context_size=hidden_dim if self.use_static_ticker else None,
        )

    def _create_xlstm(self, input_dim, hidden_dim):
        """Create an xLSTM layer"""
        intermediate_dim = hidden_dim * self.xlstm_expansion_factor

        # xLSTM components
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.forget_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.output_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)

        # xLSTM specific components with expanded dimensionality
        self.cell_intermediate = nn.Linear(input_dim + hidden_dim, intermediate_dim)
        self.cell_projection = nn.Linear(intermediate_dim, hidden_dim)

        # Layer normalization for better training stability
        self.layer_norm_ih = nn.LayerNorm(hidden_dim)
        self.layer_norm_hh = nn.LayerNorm(hidden_dim)
        self.layer_norm_c = nn.LayerNorm(hidden_dim)

        return self

    def _xlstm_cell_forward(self, x, h_prev, c_prev):
        """Forward pass for a single xLSTM cell"""
        combined = torch.cat([x, h_prev], dim=-1)

        # Gates with normalization
        i = torch.sigmoid(self.input_gate(combined))
        f = torch.sigmoid(self.forget_gate(combined))
        o = torch.sigmoid(self.output_gate(combined))

        # Enhanced cell state computation with expansion
        intermediate = F.gelu(self.cell_intermediate(combined))
        c_tilde = self.cell_projection(intermediate)

        # Update cell state with normalization
        c = f * c_prev + i * c_tilde
        c = self.layer_norm_c(c)

        # Output with normalization
        h = o * torch.tanh(c)
        h = self.layer_norm_hh(h)

        return h, c

    def _xlstm_forward(self, input_seq, init_states=None):
        """Forward pass for the entire xLSTM sequence"""
        seq_len, batch_size, input_dim = input_seq.size()
        hidden_dim = self.hidden_dim

        # Initialize hidden state and cell state
        if init_states is None:
            h_0 = torch.zeros(batch_size, hidden_dim, device=input_seq.device)
            c_0 = torch.zeros(batch_size, hidden_dim, device=input_seq.device)
        else:
            h_0, c_0 = init_states

        # Process the sequence
        output_sequence = []
        h_t, c_t = h_0, c_0

        for t in range(seq_len):
            x_t = input_seq[t]
            h_t, c_t = self._xlstm_cell_forward(x_t, h_t, c_t)
            output_sequence.append(h_t)

        # Stack outputs into a sequence
        outputs = torch.stack(output_sequence)

        return outputs, (h_t, c_t)

    def forward(
            self,
            sequence: torch.Tensor,
            static_tickers: torch.Tensor,
            encoder_representation: Union[torch.Tensor, None] = None,
            automatic_features: Union[torch.Tensor, None] = None,
            variable_importance: bool = False,
            pos_encoding_batch: Union[torch.Tensor, None] = None,
    ):
        if self.use_static_ticker:
            ticker_embedding = self.ticker_embedding(static_tickers)

        if automatic_features:
            raise NotImplementedError("TODO")
        else:
            inputs = dict(
                [
                    (
                        str(i),
                        sequence[:, :, i: (i + 1)].swapaxes(0, 1),
                    )  # swap batch and time
                    for i in range(self.input_dim)
                ]
            )

            if variable_importance:
                _, importance = self.vsn(
                    inputs,
                    (
                        self.static_context_variable_selection(ticker_embedding)
                        if self.use_static_ticker
                        else None
                    ),
                )
                return importance

            vsn_out, _ = self.vsn(
                inputs,
                (
                    self.static_context_variable_selection(ticker_embedding)
                    if self.use_static_ticker
                    else None
                ),
            )

        if pos_encoding_batch is not None:
            vsn_out = vsn_out + pos_encoding_batch.swapaxes(0, 1)

        if self.encoder_input:
            vsn_out = self.combine_layer(
                torch.cat([vsn_out, encoder_representation.swapaxes(0, 1)], dim=-1)
            )

        # Use xLSTM instead of LSTM
        if self.use_static_ticker:
            hidden_state, _ = self._xlstm_forward(
                vsn_out,
                (
                    self.static_context_state_h(ticker_embedding).unsqueeze(0).squeeze(0),
                    self.static_context_state_c(ticker_embedding).unsqueeze(0).squeeze(0),
                ),
            )
        else:
            hidden_state, _ = self._xlstm_forward(vsn_out)

        hidden_state = self.gate_add_norm_lstm(hidden_state, vsn_out)

        ffn_output = self.grn(
            hidden_state,
            (
                self.static_context_enrichment(ticker_embedding)
                if self.use_static_ticker
                else None
            ),
        )

        ffn_output = ffn_output.swapaxes(0, 1)
        hidden_state = hidden_state.swapaxes(0, 1)

        return ffn_output, hidden_state


class DeepMomentumNetworkMeta(ABCMeta, type(nn.Module)):
    pass


class DeepMomentumNetwork(ABC, nn.Module, metaclass=DeepMomentumNetworkMeta):
    def __init__(
        self,
        input_dim,
        num_tickers,
        is_fuse_static_ptp=False,
        is_position_mapped_in_arch=False,
        **kwargs,
    ) -> None:
        super().__init__()
        try:
            self.hidden_dim = kwargs["hidden_dim"]
            self.use_static_ticker = kwargs["use_static_ticker"]
            self.dropout = kwargs["dropout"]
            self.pre_loss_steps = kwargs["pre_loss_steps"] # if kwargs["run_name"] not in ["PatchTST", "PxLSTM"] else max(kwargs["pre_loss_steps"], kwargs["patch_len"]+1)
            self.seq_len = kwargs["seq_len"] # if not (kwargs["run_name"] in ["PatchTST", "PxLSTM"] and kwargs["patch_len"] > kwargs["pre_loss_steps"]) else kwargs["seq_len"]*2
            # self.all_hidden_attention = kwargs["all_hidden_attention"]
            # self.context_segmented = kwargs["context_segmented"]
            self.optimise_loss_function = kwargs["optimise_loss_function"]
            self.date_time_embedding = kwargs["date_time_embedding"]
            self.use_transaction_costs = kwargs["use_transaction_costs"]
            self.tcost_inputs = kwargs["tcost_inputs"]
            if kwargs["specify_weight_features"]:
                self.weight_features = kwargs["specify_weight_features"]
            else:
                self.weight_features = None
            if self.date_time_embedding:
                self.datetime_embedding_global_max_length = kwargs[
                    "datetime_embedding_global_max_length"
                ]
                self.local_time_embedding = kwargs["local_time_embedding"]
            if self.use_transaction_costs:
                self.vs_factor_scaler = kwargs["vs_factor_scaler"]  # .data_max_[0]
                self.trans_cost_scaler = kwargs["trans_cost_scaler"]  # .data_max_[0]
                self.fixed_trans_cost_bp_loss = kwargs["fixed_trans_cost_bp_loss"]
                self.volscale_tc_loss = kwargs["volscale_tc_loss"]
                # self.remove_over_scaling_factor = kwargs["remove_over_scaling_factor"]
                # self.remove_over_vol_scale_factor = kwargs["remove_over_vol_scale_factor"]
                self.turnover_regulariser_scaler = kwargs["turnover_regulariser_scaler"]
                self.assume_same_leverage_for_prev = kwargs[
                    "assume_same_leverage_for_prev"
                ]

            self.trans_cost_separate_loss = (
                kwargs["trans_cost_separate_loss"]
                if "trans_cost_separate_loss" in kwargs
                else False
            )

            self.replace_sharpe_loss = kwargs["replace_sharpe_loss"]

            if "output_signal_weights" in kwargs:
                self.output_signal_weights = kwargs["output_signal_weights"]
            else:
                self.output_signal_weights = False

            # self.dilated_convolution_features = kwargs["dilated_convolution_features"]
        except KeyError as exception:
            raise KeyError(
                f"Missing key in architecture settings: {exception}"
            ) from exception

        self.is_fuse_static_ptp = is_fuse_static_ptp
        self.is_l1_penalty_weight = False  # dealt with where l1 penalty declared
        self.candidate_return_loss = False

        if self.date_time_embedding:
            self.positional_encoding = positional_encoding(
                self.datetime_embedding_global_max_length, self.hidden_dim
            )

        assert isinstance(self.hidden_dim, int)
        if self.use_transaction_costs and not self.volscale_tc_loss:
            assert not self.assume_same_leverage_for_prev

        if "asset_dropout" in kwargs and kwargs["asset_dropout"] > 0.0:
            self.is_asset_dropout = True
            self.asset_dropout = DropoutNoScaling(kwargs["asset_dropout"])
        else:
            self.is_asset_dropout = False

        self.avg_returns_over = kwargs.get("avg_returns_over", None)
        self.avg_exponentially_weighted = kwargs.get("avg_exponentially_weighted", False)

        if self.use_transaction_costs and self.tcost_inputs:
            extra_dim = 3
            if self.fixed_trans_cost_bp_loss:
                extra_dim -= 1
            if not self.volscale_tc_loss:
                extra_dim -= 2
            else:
                if self.assume_same_leverage_for_prev:
                    extra_dim -= 1

            if self.optimise_loss_function == LossFunction.SHARPE.value:
                input_dim += extra_dim
        else:
            extra_dim = 0

        self.num_tickers = num_tickers

        if self.optimise_loss_function == LossFunction.SHARPE.value:
            # self.ptp_input_dim = hidden_dim
            if is_position_mapped_in_arch:
                self.prediction_to_position = DoNothing()
            elif self.weight_features:
                self.prediction_to_position = nn.Linear(
                    self.hidden_dim, len(self.weight_features)
                )
            else:
                self.prediction_to_position = nn.Sequential(
                    nn.Linear(self.hidden_dim if kwargs['run_name'] != 'PatchCopy' else 9, 1), nn.Tanh()
                )
        else:
            if is_position_mapped_in_arch:
                raise NotImplementedError("TODO")
            if self.weight_features:
                raise NotImplementedError("TODO")
            self.ptp_input_dim = (
                self.hidden_dim
            )  # * (2 if self.use_static_ticker else 1)
            if self.optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                self.to_prediction = nn.Linear(self.hidden_dim, 2)
                if self.use_transaction_costs:
                    self.ptp_prescaler = nn.Linear(2 + extra_dim, self.hidden_dim)
                else:
                    self.ptp_prescaler = nn.Linear(2, self.hidden_dim)
                    # self.ptp_input_dim = 2
            elif self.optimise_loss_function == LossFunction.JOINT_QRE.value:
                self.to_prediction = self.to_prediction = nn.Linear(
                    self.hidden_dim, len(QUANTILES)
                )
                if self.use_transaction_costs:
                    self.ptp_prescaler = nn.Linear(
                        len(QUANTILES) + extra_dim, self.hidden_dim
                    )
                else:
                    # self.ptp_input_dim = len(QUANTILES)
                    self.ptp_prescaler = nn.Linear(len(QUANTILES), self.hidden_dim)

            else:
                raise ValueError("This is not a valid loss function.")
            # trans cost ptp extra
            if self.is_fuse_static_ptp:
                self.ticker_embedding_ptp = nn.Embedding(
                    self.num_tickers, self.hidden_dim
                )

                self.static_ptp = GatedResidualNetwork(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    dropout=self.dropout,
                )

                self.ptp_fuse_static = GatedResidualNetwork(
                    self.hidden_dim,
                    self.hidden_dim,
                    self.hidden_dim,
                    # dropout=self.dropout,
                    context_size=self.hidden_dim,
                )

            if self.use_transaction_costs:
                self.prediction_to_position = nn.Sequential(
                    RNNSingleOutput(
                        self.ptp_input_dim, self.hidden_dim, batch_first=True
                    ),  # default activation is tanh
                    nn.Linear(
                        self.hidden_dim,
                        self.hidden_dim,
                    ),
                    nn.ELU(),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Tanh(),
                )
                # nn.Linear(
                #     self.ptp_input_dim,
                #     self.hidden_dim,
                # ),
                # nn.ELU(),
                # # nn.Dropout(self.dropout),
                # nn.LayerNorm(self.hidden_dim),
                # nn.Linear(self.hidden_dim, 1),
                # nn.Tanh(),
                # )
            else:
                self.prediction_to_position = nn.Sequential(
                    nn.Linear(
                        self.ptp_input_dim,
                        self.hidden_dim,
                    ),
                    nn.ELU(),
                    # nn.Dropout(self.dropout),
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, 1),
                    nn.Tanh(),
                )

        if self.optimise_loss_function == LossFunction.JOINT_QRE.value:
            self._quantiles = torch.Tensor(QUANTILES).to(device)

        self.input_dim = input_dim

        # TODO make abstract functions instead of this
        self.is_cross_section = False
        self.__multitask = False  # set in STOM arch TODO make this more modular
        self.is_signal_combine = False

    # def variable_importance(self):
    #     # maybe even condense this into the forward pass...
    #     raise NotImplementedError("TODO")

    def set_multitask(self, num_tasks, override_hidden_dim=None, l1_penalty_weight=0.0):
        self.__multitask = True
        self.num_tasks = num_tasks
        self.is_l1_penalty_weight = l1_penalty_weight > 0.0
        self.l1_penalty_weight = l1_penalty_weight

        if self.optimise_loss_function == LossFunction.SHARPE.value:
            if self.is_l1_penalty_weight:
                self.final_layer_l1_weights = nn.Linear(
                    (
                        self.num_tasks
                        * (
                            self.hidden_dim
                            if override_hidden_dim is None
                            else override_hidden_dim
                        )
                    ),
                    self.num_tasks,
                )
                self.prediction_to_position = nn.Sequential(
                    self.final_layer_l1_weights,
                    nn.Tanh(),
                )
            else:  # because of the issue with loading a model
                self.prediction_to_position = nn.Sequential(
                    nn.Linear(
                        self.num_tasks
                        * (
                            self.hidden_dim
                            if override_hidden_dim is None
                            else override_hidden_dim
                        ),
                        self.num_tasks,
                    ),
                    nn.Tanh(),
                )

        else:
            raise NotImplementedError("TODO")

    @property
    def is_multitask(self):
        return self.__multitask

    @abstractmethod
    def forward_candidate_arch(
        self,
        target_x,
        target_tickers,
        # vol_scaling_amount,
        # vol_scaling_amount_prev,
        # trans_cost_bp,
        # target_y=None,
        **kwargs,
    ):
        pass

    @abstractmethod
    def variable_importance(
        self,
        target_x,
        target_tickers,
        # vol_scaling_amount,
        # vol_scaling_amount_prev,
        # trans_cost_bp,
        # target_y=None,
        **kwargs,
    ):
        pass

    @staticmethod
    def conditional_value_at_risk(losses, alpha):
        """
        Calculates the Conditional Value at Risk (CVaR) for a given set of losses.

        Args:
            losses (torch.Tensor): A 1D tensor of losses.
            alpha (float): The confidence level (0 <= alpha <= 1).

        Returns:
            torch.Tensor: The CVaR value.
        """
        sorted_losses, _ = torch.sort(losses)
        num_samples = len(losses)
        num_to_keep = int(alpha * num_samples)
        sorted_losses_tail = sorted_losses[-num_to_keep:]
        cvar = sorted_losses_tail.mean()

        return cvar

    def concat_transaction_cost_inputs(
        self,
        input_tensor: torch.Tensor,
        vol_scaling_amount: torch.TensorType,
        vol_scaling_amount_prev: torch.TensorType,
        trans_cost_bp: torch.Tensor,
    ) -> torch.Tensor:
        if self.volscale_tc_loss:
            if self.assume_same_leverage_for_prev:
                input_tensor = torch.concat(
                    [
                        input_tensor,
                        vol_scaling_amount.unsqueeze(-1) * self.vs_factor_scaler,
                    ],
                    axis=-1,
                )
            else:
                input_tensor = torch.concat(
                    [
                        input_tensor,
                        vol_scaling_amount.unsqueeze(-1) * self.vs_factor_scaler,
                        vol_scaling_amount_prev.unsqueeze(-1) * self.vs_factor_scaler,
                    ],
                    axis=-1,
                )

        if not self.fixed_trans_cost_bp_loss:
            input_tensor = torch.concat(
                [
                    input_tensor,
                    trans_cost_bp.unsqueeze(-1) * self.trans_cost_scaler,
                ],
                axis=-1,
            )
        return input_tensor

    def combine_batch_and_asset_dim(
        self, x: torch.Tensor, num_dimensions=4
    ) -> torch.Tensor:
        """Combine batch and asset dimensions"""
        if num_dimensions == 2:
            return x.reshape(-1)
        elif num_dimensions == 3:
            return x.reshape(-1, x.shape[-1])
        elif num_dimensions == 4:
            return x.reshape(-1, x.shape[-2], x.shape[-1])
        else:
            raise NotImplementedError("TODO")

    def revert_to_original_dimensions(
        self, x: torch.Tensor, num_dimensions_original=4
    ) -> torch.Tensor:
        """Revert to original dimensions"""
        if num_dimensions_original == 3:
            return x.reshape(-1, self.num_tickers, x.shape[-1])
        elif num_dimensions_original == 4:
            return x.reshape(-1, self.num_tickers, x.shape[-2], x.shape[-1])
        else:
            raise NotImplementedError("TODO")

    def forward(
        self,
        target_x: torch.Tensor,
        target_tickers: torch.Tensor,
        vol_scaling_amount: torch.Tensor,
        vol_scaling_amount_prev: torch.Tensor,
        trans_cost_bp: torch.Tensor,
        mode: DmnMode,
        target_y: Optional[torch.Tensor] = None,
        force_sharpe_loss: bool = False,
        date_time_embedding_index=None,
        **kwargs,
    ):
        kwargs = kwargs.copy()
        last_t_steps = self.seq_len - self.pre_loss_steps

        regulariser_loss = 0.0
        regularise = False

        is_mask_entire_sequence = (
            "mask_entire_sequence" in kwargs
            and kwargs["mask_entire_sequence"] is not None
        )

        if self.date_time_embedding:
            if self.local_time_embedding:
                date_time_embedding_index = (
                    date_time_embedding_index - date_time_embedding_index[:, :1]
                )
            kwargs["pos_encoding_batch"] = self.positional_encoding[
                date_time_embedding_index
            ].to(device)

        if (
            self.optimise_loss_function == LossFunction.SHARPE.value
            and self.use_transaction_costs
            and self.tcost_inputs
        ):
            target_x = self.concat_transaction_cost_inputs(
                target_x, vol_scaling_amount, vol_scaling_amount_prev, trans_cost_bp
            )

        if self.is_asset_dropout:
            apply_dropout = True
            counter = 0

            # in case mask and missing assets means no assets in port
            if "mask_entire_sequence" not in kwargs:
                kwargs["mask_entire_sequence"] = (
                    torch.zeros(kwargs["batch_size"], self.num_tickers)
                    .to(device)
                    .bool()
                )
            while apply_dropout and counter < 100:
                dropout_mask = ~(
                    self.asset_dropout(
                        torch.ones(kwargs["batch_size"], self.num_tickers).to(device)
                    ).bool()
                )

                new_mask = torch.logical_or(
                    kwargs["mask_entire_sequence"], dropout_mask
                )

                counter += 1
                if (~new_mask).sum(axis=1).min() > 0:
                    apply_dropout = False
                else:
                    # if causes issues just remove asset dropout but retry a few times
                    new_mask = kwargs["mask_entire_sequence"]

            kwargs["mask_entire_sequence"] = new_mask

        if self.candidate_return_loss:
            representation, candidate_extra_loss = self.forward_candidate_arch(
                target_x, target_tickers, **kwargs
            )
        else:
            representation = self.forward_candidate_arch(
                target_x, target_tickers, **kwargs
            )

        if self.is_cross_section and not self.is_multitask:
            representation = self.combine_batch_and_asset_dim(representation, 4)
            target_y = self.combine_batch_and_asset_dim(target_y, 4)
            vol_scaling_amount = self.combine_batch_and_asset_dim(vol_scaling_amount, 3)
            vol_scaling_amount_prev = self.combine_batch_and_asset_dim(
                vol_scaling_amount, 3
            )
            trans_cost_bp = self.combine_batch_and_asset_dim(trans_cost_bp, 3)
            target_tickers = self.combine_batch_and_asset_dim(target_tickers, 2)

        if (
            mode is DmnMode.LIVE
            and self.is_signal_combine
            and self.output_signal_weights
        ):
            if self.avg_returns_over:
                raise NotImplementedError("TODO")
            return self.combine_batch_and_asset_dim(
                representation[:, :, -last_t_steps:, :], 4
            )

        elif self.optimise_loss_function == LossFunction.SHARPE.value:
            positions = self.prediction_to_position(representation)
            if self.avg_returns_over:
                positions = rolling_average_conv(positions, self.avg_returns_over, device, exponentially_weighted=self.avg_exponentially_weighted)
            if (
                mode is DmnMode.LIVE
                and self.output_signal_weights
                and not self.weight_features
            ):
                return positions[:, -last_t_steps:, :]
            if self.weight_features:
                signal_weights = torch.softmax(positions, dim=-1)
                if mode is DmnMode.LIVE and self.output_signal_weights:
                    return signal_weights[:, -last_t_steps:, :]
                positions = torch.sum(
                    signal_weights * kwargs["to_weight_x"], dim=-1
                ).unsqueeze(-1)
            if self.is_multitask:
                # TODO should actually be task instead of asset
                positions = self.combine_batch_and_asset_dim(
                    positions.swapaxes(1, 2).unsqueeze(-1), 4
                )
                target_y = self.combine_batch_and_asset_dim(target_y, 4)
                vol_scaling_amount = self.combine_batch_and_asset_dim(
                    vol_scaling_amount, 3
                )
                vol_scaling_amount_prev = self.combine_batch_and_asset_dim(
                    vol_scaling_amount, 3
                )
                trans_cost_bp = self.combine_batch_and_asset_dim(trans_cost_bp, 3)
                target_tickers = self.combine_batch_and_asset_dim(target_tickers, 2)
            if mode is DmnMode.LIVE:  # and not self.is_multitask:
                return positions[:, -last_t_steps:, -1]
        else:
            predictions = self.to_prediction(representation)
            if self.use_transaction_costs and self.tcost_inputs:
                ptp_input = self.ptp_prescaler(
                    self.concat_transaction_cost_inputs(
                        predictions,
                        vol_scaling_amount,
                        vol_scaling_amount_prev,
                        trans_cost_bp,
                    )
                )
            else:
                # ptp_input = predictions #self.ptp_prescaler(predictions)
                ptp_input = self.ptp_prescaler(predictions)

            if self.is_fuse_static_ptp:
                ptp_input = self.ptp_fuse_static(
                    ptp_input.swapaxes(0, 1),
                    self.static_ptp(self.ticker_embedding_ptp(target_tickers)),
                ).swapaxes(0, 1)

            positions = self.prediction_to_position(ptp_input)
            if self.avg_returns_over:
                positions = rolling_average_conv(positions, self.avg_returns_over, device, exponentially_weighted=self.avg_exponentially_weighted)

            if self.optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                if self.avg_returns_over:
                    raise NotImplementedError("TODO")
                
                mu, sigma = torch.split(predictions, 1, dim=-1)
                sigma = torch.nn.functional.softplus(sigma)

                if mode is DmnMode.LIVE:
                    if is_mask_entire_sequence:
                        raise NotImplementedError(
                            "TODO - incorporate mask into loss function for gaussian loss function."
                        )
                    return (
                        positions[:, -last_t_steps:, -1],
                        mu[:, -last_t_steps:, -1],
                        sigma[:, -last_t_steps:, -1],
                    )

                # negative because maximising log-likelihood
                dist = D.Normal(loc=mu, scale=sigma)
                log_prob = dist.log_prob(target_y)[:, -last_t_steps:, -1]

                scaling_factor = JOINT_SCALING_FACTOR_GAUSSIAN
                if self.is_cross_section:
                    scaling_factor *= JOINT_ADDITIONAL_SCALING_FACTOR_XDMN

                if is_mask_entire_sequence:
                    raise NotImplementedError(
                        "TODO - incorporate mask into loss function for gaussian loss function."
                    )

                    if self.is_cross_section:
                        keep_entires = (
                            (~kwargs["mask_entire_sequence"])
                            .flatten()
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                        )
                    else:
                        raise NotImplementedError("TODO")
                    predictions = predictions * keep_entires

                loss_predictions = -scaling_factor * torch.mean(log_prob)

            elif self.optimise_loss_function == LossFunction.JOINT_QRE.value:
                if self.avg_returns_over:
                    raise NotImplementedError("TODO")
                if mode is DmnMode.LIVE:
                    if is_mask_entire_sequence:
                        if self.is_cross_section:
                            keep_entires = (
                                (~kwargs["mask_entire_sequence"])
                                .flatten()
                                .unsqueeze(-1)
                                .unsqueeze(-1)
                            )
                        else:
                            raise NotImplementedError("TODO")
                    
                        positions = positions * keep_entires
                        predictions = predictions * keep_entires

                    return (
                        positions[:, -last_t_steps:, -1],
                        predictions[:, -last_t_steps:],
                    )
                scaling_factor = JOINT_SCALING_FACTOR_QR
                if self.is_cross_section:
                    scaling_factor *= JOINT_ADDITIONAL_SCALING_FACTOR_XDMN

                if is_mask_entire_sequence:
                    if self.is_cross_section:
                        keep_entires = (
                            (~kwargs["mask_entire_sequence"])
                            .flatten()
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                        )
                    else:
                        raise NotImplementedError("TODO")
                    predictions = predictions * keep_entires

                # positive because regression
                loss_predictions = scaling_factor * quantile_loss(
                    target_y[:, -last_t_steps:],
                    predictions[:, -last_t_steps:],
                    self._quantiles,
                )

            else:
                raise ValueError("This is not a valid loss function.")

        if is_mask_entire_sequence:
            if self.is_cross_section or self.is_multitask:
                keep_entires = (
                    (~kwargs["mask_entire_sequence"])
                    .flatten()
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            else:
                # keep_entires = (~kwargs["mask_entire_sequence" ]).unsqueeze(-1).unsqueeze(-1)
                raise NotImplementedError("TODO")
            if mode is not DmnMode.LIVE:
                target_y = target_y * keep_entires
            positions = (
                positions * keep_entires
            )  # this should already be done but just to be sure

        # captured_returns = positions[:, -last_t_steps:] * target_y[:, -last_t_steps:]
        captured_returns = positions * target_y

        if self.is_signal_combine:
            if mode == DmnMode.TRAINING:
                captured_returns = captured_returns.mean(dim=-1).unsqueeze(-1)
            else:
                positions = self.combine_batch_and_asset_dim(
                    positions.swapaxes(1, 2), 3
                ).unsqueeze(-1)
                captured_returns = self.combine_batch_and_asset_dim(
                    captured_returns.swapaxes(1, 2), 3
                ).unsqueeze(-1)

        if mode is DmnMode.INFERENCE:
            if self.optimise_loss_function == LossFunction.SHARPE.value:
                return (
                    captured_returns[:, -last_t_steps:, -1],
                    positions[:, -last_t_steps:, -1],
                )
            elif self.optimise_loss_function == LossFunction.JOINT_QRE.value:
                return (
                    captured_returns[:, -last_t_steps:, -1],
                    positions[:, -last_t_steps:, -1],
                    predictions[:, -last_t_steps:],
                )
            elif self.optimise_loss_function == LossFunction.JOINT_GAUSS.value:
                return (
                    captured_returns[:, -last_t_steps:, -1],
                    positions[:, -last_t_steps:, -1],
                    mu[:, -last_t_steps:],
                    sigma[:, -last_t_steps:],
                )

        if self.use_transaction_costs:
            if self.volscale_tc_loss:
                if self.fixed_trans_cost_bp_loss:
                    positions_scaled = (
                        positions
                        * vol_scaling_amount.unsqueeze(-1)
                        * self.fixed_trans_cost_bp_loss
                        * 1e-4
                    )
                    prev_positions_scaled = F.pad(positions_scaled, (0, 0, 1, 0))[
                        :, :-1
                    ]
                else:
                    positions_scaled = (
                        positions
                        * vol_scaling_amount.unsqueeze(-1)
                        * trans_cost_bp.unsqueeze(-1)
                        * 1e-4
                    )
                    if self.assume_same_leverage_for_prev:
                        prev_positions_scaled = (
                            F.pad(positions, (0, 0, 1, 0))[:, :-1]
                            * vol_scaling_amount.unsqueeze(-1)
                            * trans_cost_bp.unsqueeze(-1)
                            * 1e-4
                        )
                    else:
                        prev_positions_scaled = F.pad(positions_scaled, (0, 0, 1, 0))[
                            :, :-1
                        ]
            else:
                if self.fixed_trans_cost_bp_loss:
                    positions_scaled = positions * self.fixed_trans_cost_bp_loss * 1e-4
                else:
                    positions_scaled = positions * trans_cost_bp.unsqueeze(-1) * 1e-4

                prev_positions_scaled = F.pad(positions_scaled, (0, 0, 1, 0))[:, :-1]

            if not self.trans_cost_separate_loss:
                captured_returns = (
                    captured_returns
                    - self.turnover_regulariser_scaler
                    * torch.abs(positions_scaled - prev_positions_scaled)
                )
            else:
                regularise = True
                regulariser_loss += (
                    self.turnover_regulariser_scaler
                    * torch.abs(positions_scaled - prev_positions_scaled)
                ).mean()

        if self.is_cross_section or self.is_multitask:
            captured_returns = self.revert_to_original_dimensions(
                captured_returns, 4
            ).mean(dim=1)
            # if self.optimise_loss_function != LossFunction.SHARPE.value:
            #     raise NotImplementedError("TODO")

        if self.replace_sharpe_loss is None:
            loss_sharpe = -(
                torch.mean(captured_returns[:, -last_t_steps:, -1])
                / (torch.std(captured_returns[:, -last_t_steps:, -1]) + 1e-20)
            ) * np.sqrt(252.0)
        else:
            args = self.replace_sharpe_loss.split("_")
            if args[0] == "CVaR":
                alpha = 1.0 - float(args[1]) / 100.0
                losses = captured_returns[:, -last_t_steps:, -1].flatten()
                cvar = self.conditional_value_at_risk(losses, alpha)
                loss_sharpe = -(torch.mean(losses) / cvar) * np.sqrt(252.0)
            else:
                raise NotImplementedError("TODO")

        if self.is_l1_penalty_weight:
            regularise = True
            if self.candidate_return_loss:
                regulariser_loss += candidate_extra_loss
            else:
                regulariser_loss += (
                    self.l1_penalty_weight
                    * self.final_layer_l1_weights.weight.abs().sum()
                )

        if regularise:
            loss_sharpe += regulariser_loss

        if (
            self.optimise_loss_function == LossFunction.SHARPE.value
            or force_sharpe_loss
        ):
            return loss_sharpe

        return 0.5 * loss_sharpe + 0.5 * loss_predictions
