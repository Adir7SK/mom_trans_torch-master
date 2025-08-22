import torch
import torch.nn as nn
import torch.nn.functional as F
from mom_trans_torch.models.dmn import DeepMomentumNetwork


class CausalDLinearSequenceRepresentation(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int):
        super().__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.decomposition = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            padding=0,
            groups=input_dim,
            bias=False
        )
        weight = torch.ones(input_dim, 1, kernel_size) / kernel_size
        self.decomposition.weight = nn.Parameter(weight)

        self.linear = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=1,
            groups=1,
            bias=True,
        )

    def forward(self, x, target_tickers=None, pos_encoding_batch=None, variable_importance=False, **kwargs):
        """
        x: [B, L, D]
        """
        x = x.permute(0, 2, 1)  # [B, D, L]

        # if x.shape[2] < self.kernel_size:
        #     pad = self.kernel_size - x.shape[2]
        #     x_padded = F.pad(x, (pad, 0), mode="replicate")
        # else:
        #     x_padded = x

        x_padded = F.pad(x, (self.kernel_size - 1, 0))      # -> remember that with padding F.pad(x, (left, right)) where you will get left additional zeros on the left of the series (before the beginning of the time series) and right zeros at the right of the series.

        trend = self.decomposition(x_padded)
        trend = trend[..., -x.shape[2]:]    # = trend[:, :, -x.shape[2]:]
        seasonal = x - trend

        out = self.linear(seasonal) + trend  # [B, D, L]

        if variable_importance:
            # Same approach as in NLinear: L2 norm over time, average over batch
            importance = torch.norm(out, p=2, dim=2).mean(dim=0)  # shape: [D]
            importance = importance / (importance.sum() + 1e-6)
        else:
            importance = None

        return out.permute(0, 2, 1), importance  # [B, L, D], [D]


class CausalDLinearBaseline(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        kernel_size: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        self.seq_rep = CausalDLinearSequenceRepresentation(
            input_dim=input_dim,
            kernel_size=kernel_size,
        )
        self.feature_projection = nn.Linear(input_dim, hidden_dim)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, _ = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)
        return self.feature_projection(representation)  # [B, L, hidden_dim]

    def variable_importance(self, target_x, target_tickers, **kwargs):
        _, importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance
