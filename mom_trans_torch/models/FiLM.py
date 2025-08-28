import torch
import torch.nn as nn
from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation
from mom_trans_torch.models.common import (
    GateAddNorm,
    GatedResidualNetwork,
    PositionalEncoding,
)
from neuralforecast.models.film import FiLM


class FiLMBaseline(DeepMomentumNetwork):
    def __init__(self, input_dim, num_tickers, hidden_dim, dropout,
                 use_static_ticker=True, auto_feature_num_input_linear=0, **kwargs):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )
        self.seq_rep = SequenceRepresentation(
            input_dim, hidden_dim, dropout, num_tickers,
            fuse_encoder_input=False, use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear
        )
        self.pos_enc = PositionalEncoding(hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.masking = True
        self.film = FiLM(d_model=hidden_dim, n_layers=2, dropout=dropout)
        self.gate_norm = GateAddNorm(hidden_dim, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward_candidate_arch(self, x, tickers, pos_encoding_batch=None, **kwargs):
        x, _ = self.seq_rep(x, tickers, pos_encoding_batch=pos_encoding_batch)
        x = self.pos_enc(x)
        mask = None
        if self.masking:
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        x = self.grn(x)
        out = self.film(x, attn_mask=mask)
        out = self.gate_norm(out, x)
        return self.decoder(out)

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)
