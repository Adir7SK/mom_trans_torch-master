import torch
import torch.nn as nn
import torch.nn.functional as F
from mom_trans_torch.models.dmn import DeepMomentumNetwork


def causal_running_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Cumulative mean that does not see into the future
    """
    # x: [B, D, T]
    cumsum = x.cumsum(dim=2)
    time_steps = torch.arange(1, x.size(2) + 1, device=x.device).float()
    return cumsum / time_steps.view(1, 1, -1)


class NLinearBaseline(DeepMomentumNetwork):
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
        self.seq_rep = NLinearSequenceRepresentation(
            input_dim=input_dim,
            seq_len=self.seq_len,
        )
        # self.linear_layer = nn.Linear(self.seq_len, self.seq_len)
        self.kernel_size = kernel_size
        self.linear_layer = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=self.kernel_size,
            padding=0,      # No Auto-padding -> manual padding stays consistent over larger time-series
            groups=input_dim,  # makes it per-feature (depthwise)
            bias=True,
        )

        self.feature_projection = nn.Linear(input_dim, self.hidden_dim)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        B, L, D = target_x.shape

        if self.seq_len != L or self.linear_layer is None:
            self._build_layers(seq_len=L)

        # Transpose to [B, D, L]
        x = target_x.permute(0, 2, 1)

        # Remove mean
        # mean = x.mean(dim=2, keepdim=True)
        mean = causal_running_mean(x)
        x_centered = x - mean

        # Causal left-padding
        x_centered = F.pad(x_centered, (self.kernel_size - 1, 0))  # only pad on the left -> hence we enforce that the kernek will only look at past steps

        out = self.linear_layer(x_centered)

        # Trim future-looking padding (keep only causal output)
        out = out[:, :, :x.shape[2]]  # Keep first L time steps only

        # Add mean back
        out = out + mean

        # Return to shape [B, L, D]
        out = out.permute(0, 2, 1)
        return self.feature_projection(out)

    def variable_importance(self, target_x, target_tickers, **kwargs):
        _, importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)


class NLinearSequenceRepresentation(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
    ):
        super().__init__()
        self.channels = input_dim
        self.seq_len = None  # will set dynamically
        self.linear = nn.Linear(seq_len, seq_len)

    def _build_layers(self, seq_len):
        self.linear = nn.Linear(seq_len, seq_len)
        self.seq_len = seq_len

    def forward(self, x, target_tickers=None, pos_encoding_batch=None, variable_importance=False, **kwargs):
        """
        x: [B, L, D]
        Returns:
            - output: [B, L, D]
            - importance (optional): [D] or [B, D]
        """
        if self.seq_len != x.shape[1] or self.linear is None:
            self._build_layers(x.shape[1])

        x = x.permute(0, 2, 1)  # [B, D, L]
        mean = x.mean(dim=2, keepdim=True)
        x_centered = x - mean

        out = self.linear(x_centered)  # [B, D, L]
        out = out + mean

        # Compute variable importance if requested
        if variable_importance:
            # Sum of squared output per variable over time, then average over batch
            importance = torch.norm(out, p=2, dim=2).mean(dim=0)  # shape: [D]
            importance = importance / (importance.sum() + 1e-6)  # normalize to sum to 1
        else:
            importance = None

        out = out.permute(0, 2, 1)  # [B, L, D]
        return out, importance
