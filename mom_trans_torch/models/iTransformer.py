from mom_trans_torch.models.dmn import DeepMomentumNetwork
# from iTransformer import iTransformer  # official repo
import torch
import torch.nn as nn
from torch import Tensor


class iTransformerOld(DeepMomentumNetwork):

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
        # Project each timestep independently
        self.proj = nn.Linear(1, hidden_dim)

        # Causal transformer (time-wise)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=4 * hidden_dim,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # Feature attention (with causal masking)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

    def forward_candidate_arch(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            pos_encoding_batch=None,
            **kwargs
    ) -> Tensor:
        """Forward pass with channel-wise attention
        Args:
            target_x: [B, L, C]
            target_tickers: [B]
        Returns:
            [B, L, H]
        """
        # x: [B, L, C]
        B, L, C = target_x.shape

        # 1. Independent channel processing
        x = target_x.unsqueeze(-1)  # [B, L, C, 1]
        x = self.proj(x)  # [B, L, C, H]

        # 2. Causal temporal processing
        x = x.permute(0, 2, 1, 3)  # [B, C, L, H]
        x = x.reshape(B * C, L, self.hidden_dim)
        temporal_out = self.temporal_transformer(x, mask=self._causal_mask(L))  # [B*C, L, H]

        # 3. Causal feature attention
        temporal_out = temporal_out.view(B, C, L, self.hidden_dim)
        feature_out = torch.zeros_like(temporal_out)
        for t in range(L):
            # Only use current/past timesteps
            feat_in = temporal_out[:, :, :t + 1, :].reshape(B, -1, self.hidden_dim)
            attn_out, _ = self.feature_attention(
                query=temporal_out[:, :, t, :],
                key=feat_in,
                value=feat_in
            )
            feature_out[:, :, t, :] = attn_out

        return feature_out.mean(dim=1)  # [B, L, H]

    def _causal_mask(self, size):
        return torch.triu(torch.ones(size, size) * float('-inf'), diagonal=1)

    def variable_importance(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            n_samples: int = 5,
            **kwargs
    ) -> Tensor:
        """Channel importance via gradient attribution"""
        target_x = target_x.detach().requires_grad_(True)
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            self.train()  # Enable dropout
            output = self.forward(target_x, target_tickers)
            output.mean().backward()

            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None

        return importance / n_samples


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, scales):
        super().__init__()
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=4 * hidden_dim,
                    batch_first=True
                ),
                num_layers=num_layers
            ) for _ in scales
        ])
        self.scales = scales

    def forward(self, x, mask_fn):
        # x: [B*C, L, H]
        outs = []
        for i, scale in enumerate(self.scales):
            mask = mask_fn(x.size(1), scale)
            outs.append(self.transformers[i](x, mask=mask))
        return torch.cat(outs, dim=-1)  # Concatenate on feature dim


class DynamicFeatureSelector(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, C, L, H]
        gate_scores = torch.sigmoid(self.gate(x))  # [B, C, L, 1]
        return x * gate_scores


class iTransformer(DeepMomentumNetwork):
    def __init__(self, input_dim, num_tickers, hidden_dim, num_layers, num_heads, dropout, use_static_ticker=True, **kwargs):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            **kwargs
        )
        self.proj = nn.Linear(1, hidden_dim)
        self.multi_scale_temporal = MultiScaleTemporalBlock(hidden_dim, num_heads, num_layers, scales=[1, 2, 4])
        self.feature_selector = DynamicFeatureSelector(hidden_dim * 3)
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 3,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        # Add this projection layer
        self.final_proj = nn.Linear(hidden_dim * 3, hidden_dim)

    def _causal_mask(self, size, scale=1):
        mask = torch.triu(torch.ones(size, size) * float('-inf'), diagonal=scale)
        return mask

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        B, L, C = target_x.shape
        x = target_x.unsqueeze(-1)
        x = self.proj(x)
        x = x.permute(0, 2, 1, 3).reshape(B * C, L, self.hidden_dim)
        temporal_out = self.multi_scale_temporal(x, self._causal_mask)
        temporal_out = temporal_out.view(B, C, L, -1)
        selected = self.feature_selector(temporal_out)
        x = selected.permute(0, 2, 1, 3).reshape(B * L, C, -1)
        mask = torch.triu(torch.ones(C, C, device=x.device) * float('-inf'), diagonal=1)
        attn_out, _ = self.feature_attention(x, x, x, attn_mask=mask)
        attn_out = attn_out.reshape(B, L, C, -1).permute(0, 2, 1, 3)
        out = attn_out.mean(dim=1)  # [B, L, hidden_dim * 3]
        # Project to expected hidden_dim
        out = self.final_proj(out)  # [B, L, hidden_dim]
        return out

    def variable_importance(
            self,
            target_x: Tensor,
            target_tickers: Tensor,
            n_samples: int = 5,
            **kwargs
    ) -> Tensor:
        """Channel importance via gradient attribution"""
        target_x = target_x.detach().requires_grad_(True)
        importance = torch.zeros(target_x.shape[-1], device=target_x.device)

        for _ in range(n_samples):
            self.train()  # Enable dropout
            output = self.forward(target_x, target_tickers)
            output.mean().backward()

            importance += target_x.grad.abs().mean(dim=(0, 1))
            target_x.grad = None

        return importance / n_samples
