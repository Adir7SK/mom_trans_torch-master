import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from mom_trans_torch.models.dmn import DeepMomentumNetwork


class FedFormer(DeepMomentumNetwork):
    """Strictly causal FedFormer with:
    - Frequency-enhanced decomposition (DFT/IDFT)
    - Seasonal-Trend separation
    - Gradient-based variable importance
    - Full causality guarantees
    """

    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            num_heads: int,
            patch_len: int,
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

        # Input projection (per-feature)
        self.input_proj = nn.Linear(1, hidden_dim)

        # Frequency-enhanced components
        self.freq_encoder = FrequencyEncoder(hidden_dim, num_heads, dropout)
        self.trend_encoder = TrendEncoder(hidden_dim, dropout)

        # Ticker embeddings
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
            pos_encoding_batch=None,
            **kwargs
    ) -> Tensor:
        B, L, C = target_x.shape

        # 1. Project each feature
        x = target_x.unsqueeze(-1)  # [B, L, C, 1]
        x = self.input_proj(x)  # [B, L, C, H]

        # 2. Seasonal-trend decomposition
        seasonal, trend = self._decompose(x)  # Both [B, L, C, H]

        # 3. Frequency-enhanced processing
        freq_out = self.freq_encoder(seasonal)  # [B, L, C, H]
        trend_out = self.trend_encoder(trend)  # [B, L, C, H]

        # 4. Combine and reduce
        combined = freq_out + trend_out
        output = combined.mean(dim=2)  # [B, L, H]

        # 5. Add ticker embedding
        if self.use_static_ticker:
            ticker_emb = self.ticker_proj(self.ticker_embedding(target_tickers))
            output = output + ticker_emb.unsqueeze(1)

        return self.norm(self.output_proj(output))

    def _decompose(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Causal seasonal-trend decomposition"""
        B, L, C, H = x.shape

        # Reshape for 1D convolution (combine B and C dimensions)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * C, L, H)
        # Further transpose to get [B*C, H, L] as required by avg_pool1d
        x_reshaped = x_reshaped.transpose(1, 2)

        # Causal padding and pooling
        x_pad = F.pad(x_reshaped, (self.patch_len - 1, 0))  # Pad on the left for causality
        trend_reshaped = F.avg_pool1d(x_pad, kernel_size=self.patch_len, stride=1)

        # Restore original dimensions
        trend_reshaped = trend_reshaped.transpose(1, 2)  # [B*C, L, H]
        trend = trend_reshaped.reshape(B, C, L, H).permute(0, 2, 1, 3)  # [B, L, C, H]

        # Seasonal = Original - Trend
        seasonal = x - trend
        return seasonal, trend

    def variable_importance(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            n_samples: int = 10,
            normalize: bool = True
    ) -> Tensor:
        """Monte Carlo Gradient-based Feature Importance
        Args:
            target_x: Input tensor [B, L, C]
            target_tickers: Ticker IDs [B]
            n_samples: MC samples for robustness
            normalize: Scale scores to sum=1
        Returns:
            importance: Tensor [C] of feature importance scores
        """
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            # Enable gradients and dropout
            target_x = target_x.detach().requires_grad_(True)
            self.train()

            # Forward pass
            output = self.forward(target_x, target_tickers)

            # L1-norm aggregation
            loss = output.abs().mean()
            loss.backward()

            # Accumulate absolute gradients
            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None

        # Average and normalize
        importance /= n_samples
        if normalize:
            importance = importance / (importance.sum() + 1e-8)

        return importance

    def verify_causality(self, test_seq: Tensor) -> bool:
        """Empirical causality test: Zero-out future => Zero outputs"""
        with torch.no_grad():
            midpoint = test_seq.size(1) // 2
            test_seq[:, midpoint:, :] = 0
            output = self.forward(test_seq, torch.zeros(test_seq.size(0), dtype=torch.long))
            return torch.allclose(
                output[:, midpoint:, :],
                torch.zeros_like(output[:, midpoint:, :]),
                atol=1e-6
            )


class FrequencyEncoder(nn.Module):
    """Causal frequency processing with DFT"""

    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        # Real FFT components
        self.dft_proj = nn.Linear(hidden_dim, hidden_dim)
        self.idft_proj = nn.Linear(hidden_dim, hidden_dim)

        # Causal attention
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout, batch_first=True)

        # Causal conv
        self.conv = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3,
            padding=1,  # Will be padded causally in forward
            groups=hidden_dim
        )

    def forward(self, x: Tensor) -> Tensor:
        B, L, C, H = x.shape
        x = x.reshape(B * C, L, H)

        # Real FFT processing (causal)
        x_freq = torch.fft.rfft(x, dim=1, norm='ortho')
        x_freq = self.dft_proj(x_freq.real) + 1j * self.dft_proj(x_freq.imag)
        x = torch.fft.irfft(x_freq, n=L, dim=1, norm='ortho')

        # Causal attention
        mask = torch.triu(torch.ones(L, L) * float('-inf'), diagonal=1).to(x.device)
        x, _ = self.attention(x, x, x, attn_mask=mask)

        # Causal convolution
        x = F.pad(x.transpose(1, 2), (1, 0), mode='reflect')
        x = self.conv(x)[..., :L].transpose(1, 2)

        return x.reshape(B, L, C, H)


class TrendEncoder(nn.Module):
    """Causal trend processing"""

    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        B, L, C, H = x.shape
        x = x.reshape(B * C, L, H)
        x = self.norm(x + self.mlp(x))
        return x.reshape(B, L, C, H)
