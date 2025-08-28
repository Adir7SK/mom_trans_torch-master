import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mom_trans_torch.models.dmn import DeepMomentumNetwork


class Patch_xLSTM(DeepMomentumNetwork):
    """Causal Patch-based xLSTM with:
    - Same hierarchical patching as PatchTST
    - xLSTM blocks for sequence processing
    - Strict causality guarantees
    """

    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            patch_len: int,
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
        self.patch_len = patch_len
        self.stride = kwargs.get('stride', 1)  # Default overlap

        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)

        # Patching
        self.patch_embed = nn.Linear(patch_len * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # xLSTM blocks (mixture of sLSTM and mLSTM)
        self.xlstm_blocks = nn.ModuleList([
            xLSTMBlock(hidden_dim, num_heads, dropout)
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

    def _create_patches(self, x: Tensor):
        """Convert (B, L, C) input to (B*C, L, H) causal patches"""
        B, L, C = x.shape

        # Channel-independent embedding
        x = x.permute(0, 2, 1).reshape(B * C, L, 1)  # [B*C, L, 1]
        x = self.input_proj(x)  # [B*C, L, H]

        # Create lookback-only patches (strictly causal)
        patches = []
        for i in range(L):
            # For each position, create a patch using only current and past values
            start_idx = max(0, i - self.patch_len + 1)
            patch_data = x[:, start_idx:i + 1, :]

            # Pad if needed (for early positions with insufficient history)
            if patch_data.size(1) < self.patch_len:
                padding = torch.zeros(
                    B * C,
                    self.patch_len - patch_data.size(1),
                    self.hidden_dim,
                    device=x.device
                )
                patch_data = torch.cat([padding, patch_data], dim=1)

            patches.append(patch_data.reshape(B * C, 1, -1))

        # Concatenate all patches
        patches = torch.cat(patches, dim=1)  # [B*C, L, patch_len*H]

        # Project patches to hidden dimension
        patches = self.patch_embed(patches)  # [B*C, L, H]

        return patches, B, C

    def forward_candidate_arch(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            pos_encoding_batch=None,
            **kwargs
    ) -> Tensor:
        patches, B, C = self._create_patches(target_x)

        # xLSTM processing (causal by design)
        for block in self.xlstm_blocks:
            patches = block(patches)  # [B*C, N, H]

        # Reshape and project
        patches = patches.view(B, C, -1, self.hidden_dim).mean(dim=1)
        output = self._temporal_upsample(patches, target_x.size(1))

        if self.use_static_ticker:
            output += self.ticker_proj(self.ticker_embedding(target_tickers)).unsqueeze(1)

        return self.norm(self.output_proj(output))

    def _temporal_upsample(self, x: Tensor, target_len: int) -> Tensor:
        """Causal upsampling without interpolation"""
        x = x.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
        return x.transpose(1, 2)

    def variable_importance(
            self,
            target_x: torch.Tensor,
            target_tickers: torch.Tensor,
            n_samples: int = 10,
            normalize: bool = True,
            **kwargs
    ) -> torch.Tensor:
        """Computes gradient-based feature importance.

        Args:
            target_x: Input tensor (B, L, C)
            target_tickers: Ticker indices (B,)
            n_samples: Number of Monte Carlo samples for stability
            normalize: Whether to normalize scores to [0,1]

        Returns:
            importance: Tensor of shape (C,) with feature importance scores
        """
        # Enable gradient tracking
        target_x = target_x.detach().requires_grad_(True)

        # Monte Carlo sampling for robust importance
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            # Forward pass with dropout active
            self.train()
            output = self.forward(target_x, target_tickers)

            # L1 norm aggregation
            loss = output.abs().mean()
            loss.backward()

            # Accumulate absolute gradients
            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None  # Reset gradients

        # Average over samples
        importance /= n_samples

        # Normalize if requested
        if normalize:
            importance = importance / (importance.sum() + 1e-8)

        return importance


class xLSTMBlock(nn.Module):
    """xLSTM block combining sLSTM and mLSTM"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        # sLSTM component
        self.slstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Split capacity
            num_layers=1,
            batch_first=True
        )

        # mLSTM component
        self.mlstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )

        # Projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        # sLSTM path
        s_out, _ = self.slstm(x)

        # mLSTM path
        m_out, _ = self.mlstm(x)

        # Combine
        return self.proj(torch.cat([s_out, m_out], dim=-1))
