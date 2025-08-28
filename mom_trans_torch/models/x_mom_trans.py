
import torch
import torch.nn.functional as F
# from pytorch_forecasting.models.temporal_fusion_transformer import (
#     GateAddNorm,
#     GatedResidualNetwork,
# )
from torch import nn

from mom_trans_torch.models.common import (
    # LossFunction,
    for_masked_fill,
    # for_masked_fill_forecasting,
    # look_ahead_mask,
    # max_norm,
    GateAddNorm,
    GatedResidualNetwork,
)
from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentationXLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class XMomentumTransformer(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,  # TODO maybe diable this for now..
        # output_signal_weights=False,
        **kwargs
    ):
        fuse_encoder_input = False
        # self.output_signal_weights = output_signal_weights
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        assert isinstance(num_heads, int)

        self.seq_rep = SequenceRepresentationXLSTM(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input,
            use_static_ticker,
            auto_feature_num_input_linear,
        )
        self.num_heads = num_heads

        self.gate_add_norm_mha = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )
        # grn3 = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        # self.interp_mha = InterpretableMultiHeadAttention(
        #     num_heads, hidden_dim, dropout
        # )

        # TODO remove dropout
        self.self_att = nn.MultiheadAttention(
            hidden_dim, self.num_heads, batch_first=True,
        )

        self.gate_add_norm_block = GateAddNorm(hidden_dim, dropout=dropout)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, lstm_hidden_state = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)

        # TODO cache
        mask = for_masked_fill(self.seq_len).to(device)
        mha, _ = self.self_att.forward(
            representation, representation, representation, attn_mask=mask
        )
        # w_imha[0, 0,0,:].sum()
        add = self.gate_add_norm_mha.forward(mha, representation)
        ffn_representation = self.ffn(add)
        transformer_representation = self.gate_add_norm_block(
            ffn_representation, lstm_hidden_state
        )
        return transformer_representation

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)
