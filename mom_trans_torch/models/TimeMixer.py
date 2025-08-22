import torch
import torch.nn as nn
from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class CausalDWT(nn.Module):
    """Causal Discrete Wavelet Transform using Haar wavelet"""

    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        self.dec_lo = nn.Parameter(torch.tensor([1.0, 1.0]), requires_grad=False)
        self.dec_hi = nn.Parameter(torch.tensor([1.0, -1.0]), requires_grad=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # x: [B, L, C]
        B, L, C = x.shape
        x = x.transpose(1, 2)  # [B, C, L]

        # Use zero padding instead of reflect for strict causality
        # Only pad on the left to maintain causality
        x = F.pad(x, (1, 0), mode='constant', value=0)

        # Process each channel group separately
        lo = torch.zeros(B, C, L // 2, device=x.device)
        hi = torch.zeros(B, C, L // 2, device=x.device)

        # Group convolution approach
        for c in range(C):
            # For each channel, apply the filters
            channel_data = x[:, c:c + 1, :]

            lo[:, c:c + 1, :] = F.conv1d(
                channel_data,
                self.dec_lo.view(1, 1, 2),
                stride=2
            )

            hi[:, c:c + 1, :] = F.conv1d(
                channel_data,
                self.dec_hi.view(1, 1, 2),
                stride=2
            )

        # Return in the original dimension ordering
        return lo.transpose(1, 2), hi.transpose(1, 2)  # [B, L//2, C]


class TimeMixer(DeepMomentumNetwork):
    """Strictly causal TimeMixer for financial forecasting"""

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

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Causal frequency decomposition
        self.dwt = CausalDWT()

        # Create proper causal transformer encoders
        self.lo_mixers = nn.ModuleList()
        self.hi_mixers = nn.ModuleList()

        for _ in range(num_layers):
            # Create encoder layers with causal self-attention
            lo_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            # Wrap in TransformerEncoder for mask support
            self.lo_mixers.append(nn.TransformerEncoder(lo_layer, num_layers=1))

            hi_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            # Wrap in TransformerEncoder for mask support
            self.hi_mixers.append(nn.TransformerEncoder(hi_layer, num_layers=1))

        # Inter-scale mixing
        self.inter_mix = nn.Linear(2 * hidden_dim, hidden_dim)

        # Ticker embeddings
        self.use_static_ticker = use_static_ticker
        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
            self.ticker_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def _causal_mask(self, size: int) -> Tensor:
        """Generate upper triangular causal mask"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool().to(self.input_proj.weight.device)

    def forward_candidate_arch(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            pos_encoding_batch: Optional[Tensor] = None,
            **kwargs
    ) -> Tensor:
        # Input projection
        x = self.input_proj(target_x)  # [B, L, H]

        # Causal frequency decomposition
        lo, hi = self.dwt(x)  # [B, L//2, H]

        # Intra-scale mixing with causal masking
        lo_mask = self._causal_mask(lo.size(1))
        hi_mask = self._causal_mask(hi.size(1))

        for lo_layer, hi_layer in zip(self.lo_mixers, self.hi_mixers):
            # Apply transformer with causal mask
            lo = lo_layer(lo, mask=lo_mask)
            hi = hi_layer(hi, mask=hi_mask)

        # Inter-scale mixing with causal upsampling
        B, L_half, H = lo.shape
        L_full = x.size(1)

        # Causal upsampling by repeating each element
        lo_upsampled = torch.zeros(B, L_full, H, device=x.device)
        hi_upsampled = torch.zeros(B, L_full, H, device=x.device)

        # Each element in lo/hi corresponds to 2 elements in the original sequence
        for i in range(L_half):
            lo_upsampled[:, 2 * i:2 * i + 2, :] = lo[:, i:i + 1, :]
            hi_upsampled[:, 2 * i:2 * i + 2, :] = hi[:, i:i + 1, :]

        mixed = self.inter_mix(torch.cat([lo_upsampled, hi_upsampled], dim=-1))

        # Add ticker embedding if enabled
        if self.use_static_ticker and target_tickers is not None:
            ticker_emb = self.ticker_proj(self.ticker_embedding(target_tickers))
            mixed = mixed + ticker_emb.unsqueeze(1)

        return self.norm(self.output_proj(mixed))

    def variable_importance(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            n_samples: int = 5,
            **kwargs
    ) -> Tensor:
        """Gradient-based feature importance with MC sampling"""
        target_x = target_x.detach().requires_grad_(True)
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            self.train()
            output = self.forward(target_x, target_tickers)
            output.mean().backward()

            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None

        return importance / n_samples

    def verify_causality(self, test_seq: Tensor) -> bool:
        """Empirical causality verification"""
        with torch.no_grad():
            midpoint = test_seq.size(1) // 2
            test_seq[:, midpoint:, :] = 0
            output = self.forward(test_seq, torch.zeros(test_seq.size(0), dtype=torch.long))
            return torch.allclose(
                output[:, midpoint:, :],
                torch.zeros_like(output[:, midpoint:, :]),
                atol=1e-6
            )
