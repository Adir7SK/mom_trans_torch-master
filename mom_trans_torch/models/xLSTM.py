import torch
import torch.nn as nn
from torch import Tensor
from mom_trans_torch.models.dmn import DeepMomentumNetwork


class xLSTM(DeepMomentumNetwork):
    """Pure xLSTM model with the same interface as P-sLSTM but without patching"""

    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            num_layers: int,
            num_heads: int,
            dropout: float,
            use_static_ticker: bool = True,
            **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            **kwargs
        )
        self.input_dim = input_dim

        # Input projection (no patching)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout_func = nn.Dropout(self.dropout)

        # xLSTM blocks
        self.xlstm_blocks = nn.ModuleList([
            xLSTMBlock(hidden_dim, num_heads, self.dropout)
            for _ in range(num_layers)
        ])

        # Ticker embedding
        self.use_static_ticker = use_static_ticker
        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
            self.ticker_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward_candidate_arch(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            **kwargs
    ) -> Tensor:
        # Input projection
        x = self.input_proj(target_x)  # [B, L, H]
        x = self.dropout_func(x)

        # xLSTM processing
        for block in self.xlstm_blocks:
            x = block(x)  # Each block maintains (B, L, H)

        # Add ticker embedding if enabled
        if self.use_static_ticker:
            ticker_emb = self.ticker_proj(self.ticker_embedding(target_tickers))
            x = x + ticker_emb.unsqueeze(1)  # Broadcast over sequence

        return self.norm(self.output_proj(x))

    def variable_importance(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            n_samples: int = 10,
            **kwargs
    ) -> Tensor:
        """Gradient-based variable importance with Monte Carlo sampling"""
        target_x = target_x.detach().requires_grad_(True)
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            self.train()  # Enable dropout
            output = self.forward(target_x, target_tickers)
            output.mean().backward()

            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None  # Reset gradients

        return importance / n_samples  # Average over samples


class xLSTMBlock(nn.Module):
    def __init__(self, hidden_dim, dropout, num_heads=None):
        super().__init__()

        # num_heads parameter is received but not used in LSTM implementation
        # kept for consistency with other model implementations
        self.stateful = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.memoryless = nn.Linear(hidden_dim, hidden_dim)

        # Gate mechanism
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Process input through both paths
        s_out, _ = self.stateful(x)  # [B, L, H]
        m_out = self.memoryless(x)  # [B, L, H]

        # Compute gate values - concatenate both outputs
        concat = torch.cat([s_out, m_out], dim=-1)  # [B, L, H*2]
        gate = self.gate(concat)  # [B, L, H]

        # Apply gating mechanism
        combined = gate * s_out + (1 - gate) * m_out  # [B, L, H]
        return self.norm(x + self.dropout(self.proj(combined)))

# class xLSTMBlock(nn.Module):
#     """Enhanced xLSTM block combining sLSTM and mLSTM pathways"""
#
#     def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
#         super().__init__()
#         # sLSTM path (global memory)
#         self.slstm = nn.LSTM(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim // 2,
#             num_layers=1,
#             batch_first=True,
#             dropout=dropout
#         )
#
#         # mLSTM path (local convolution-like)
#         self.mlstm = nn.LSTM(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim // 2,
#             num_layers=1,
#             batch_first=True,
#             dropout=dropout
#         )
#
#         # Gating mechanism
#         self.gate = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.Sigmoid()
#         )
#
#         # Projection
#         self.proj = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x: Tensor) -> Tensor:
#         # sLSTM path (captures long-range dependencies)
#         s_out, _ = self.slstm(x)
#
#         # mLSTM path (captures local patterns)
#         m_out, _ = self.mlstm(x)
#
#         # Dynamic gating
#         combined = torch.cat([s_out, m_out], dim=-1)
#         gate = self.gate(combined)
#
#         # Weighted combination
#         return self.proj(gate * s_out + (1 - gate) * m_out)
