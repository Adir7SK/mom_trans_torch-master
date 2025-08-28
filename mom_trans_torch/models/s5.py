import torch
import torch.nn as nn
from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation
from mom_trans_torch.models.common import (
    GateAddNorm,
    GatedResidualNetwork,
    PositionalEncoding,
)
from x_transformers import S5


class S5Baseline(DeepMomentumNetwork):
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
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        self.pos_enc = PositionalEncoding(hidden_dim)
        self.grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)

        self.s5 = S5(
            num_tokens=None,    # No fixed token limit
            dim=hidden_dim,
            depth=2,
            heads=4,
            causal=True,        # <--- Ensures causality
            dropout=dropout,
        )

        self.gate_norm = GateAddNorm(hidden_dim, dropout)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        # [B, L, D]
        x, _ = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)
        x = self.pos_enc(x)
        x = self.grn(x)

        out = self.s5(x)  # Causal=True ensures autoregressive flow
        out = self.gate_norm(out, x)

        return self.decoder(out)  # [B, L, D]

    def variable_importance(self, target_x, target_tickers, **kwargs):
        # Importance based on output sensitivity to each input feature
        output = self.forward_candidate_arch(target_x, target_tickers)  # [B, L, D]
        importance = torch.norm(output, dim=1).mean(dim=0)              # [D]
        return importance / (importance.sum() + 1e-6)                   # Normalize
