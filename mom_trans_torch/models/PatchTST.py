from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentationSimple, GatedResidualNetwork
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import math
from transformers import PatchTSTConfig, PatchTSTModel


class CustomStridePatchTSTModel(PatchTSTModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        self.patch_len = config.patch_length
        self.stride = config.patch_stride
        self.hidden_dim = config.d_model
        self.input_dim = config.num_input_channels
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.num_layers = config.num_hidden_layers
        assert self.stride <= self.patch_len, "stride must be <= patch_len for causality."

        # First project features to hidden_dim, then create patches
        # self.feature_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Flatten patch_len * input_dim → hidden_dim
        self.proj = nn.Linear(self.patch_len * self.input_dim, self.hidden_dim)

        # Positional embeddings
        max_patches = ((config.context_length - self.patch_len) // max(1, self.stride)) + 1
        if max_patches <= 0:
            max_patches = 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, self.hidden_dim))
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.patch_transformer = TransformerEncoder(encoder_layer,
                                                    num_layers=self.num_layers)

        # Optional norm/head
        if not hasattr(self, "ln"):
            self.ln = nn.LayerNorm(self.hidden_dim)
        if not hasattr(self, "head"):
            self.head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, **kwargs):
        """
        Create patches at indices t, t-stride, t-2*stride, ...
        """
        B, L, C = x.shape
        device = x.device

        # Add padding at the beginning for causal constraints
        pad_len = self.patch_len - 1
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, device=device, dtype=x.dtype)
            x = torch.cat([pad, x], dim=1)  # [B, L + pad_len, C]

        # Process each timestep independently - this returns one embedding per timestep
        timestep_embeddings = []

        for t in range(L):
            # Calculate position in padded sequence
            t_padded = t + pad_len

            # Collect all valid patches for this timestep
            t_patches = []
            for k in range((t_padded // self.stride) + 1):
                start = t_padded - k * self.stride - self.patch_len + 1
                if start < 0:
                    break

                patch = x[:, start:start + self.patch_len, :]  # [B, patch_len, C]
                flat_patch = patch.reshape(B, self.patch_len * C)
                token = self.proj(flat_patch)  # [B, hidden_dim]
                t_patches.append(token)

            if not t_patches:
                # Fallback (shouldn't happen with proper padding)
                token = self.proj(torch.zeros(B, self.patch_len * C, device=device))
                t_patches = [token]

            # Stack patches for this timestep and process through a small transformer
            # We need to handle the variable number of patches for each timestep
            patches_t = torch.stack(t_patches, dim=1)  # [B, num_patches_t, hidden_dim]

            # Process patches with transformer and take the last token
            # This handles the variable number of patches per timestep
            timestep_embedding = self.patch_transformer(patches_t)[:, -1]  # [B, hidden_dim]
            timestep_embeddings.append(timestep_embedding.unsqueeze(1))  # [B, 1, hidden_dim]

        # Concatenate all timestep embeddings
        return torch.cat(timestep_embeddings, dim=1)  # [B, L, hidden_dim]


class PatchTST(DeepMomentumNetwork):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_tickers: int,
            dropout: float,
            use_static_ticker: bool,
            num_heads: int,
            **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            **kwargs
        )
        self.patch_len = kwargs['patch_len']
        self.num_tickers = num_tickers
        self.stride = kwargs['stride']

        config = PatchTSTConfig(
            num_input_channels=input_dim,
            context_length=self.seq_len,
            patch_length=self.patch_len,
            patch_stride=self.stride,
            d_model=hidden_dim,
            num_hidden_layers=kwargs['num_layers'],
            num_attention_heads=num_heads,
            dropout=dropout,
            prediction_length=1,
            pooling_type="mean"
        )
        self.tst_model = CustomStridePatchTSTModel(config)# PatchTSTModel(config)

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        # target_x: (B, seq_len, input_dim)

        patch_reps = self.tst_model(target_x)

        return patch_reps

    def variable_importance(
            self,
            target_x: torch.Tensor,
            target_tickers: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Computes feature-level importance scores using gradient attribution.

        Args:
            target_x: torch.Tensor (B, seq_len, input_dim)
            target_tickers: optional; not used in this implementation
        Returns:
            importance: torch.Tensor (input_dim,) — normalized feature importances.
        """
        # Make sure gradients flow to the input
        if not target_x.requires_grad:
            target_x = target_x.clone().detach().requires_grad_(True)

        # Forward pass through PatchTST with your causal mapping
        rep = self.forward_candidate_arch(target_x, target_tickers)  # (B, seq_len, hidden_dim)

        # Collapse time & hidden dims into a single prediction vector
        # (You can change this depending on how you want importance scored)
        rep_mean = rep.mean(dim=1)  # (B, hidden_dim)
        pred = self.output_head(rep_mean)  # (B, num_tickers) or whatever your head outputs

        # Use a simple attribution target: magnitude of predictions
        loss = pred.abs().mean()
        loss.backward()

        # Gradients wrt original inputs
        grads = target_x.grad.abs()  # (B, seq_len, input_dim)

        # Average over batch & time to get per-feature score
        importance = grads.mean(dim=(0, 1))  # (input_dim,)

        # Normalize so sum = 1
        importance = importance / (importance.sum() + 1e-8)

        return importance
