import torch
import torch.nn as nn
from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation
from mom_trans_torch.models.common import (
    GateAddNorm,
    GatedResidualNetwork,
)


class PatchTST2(DeepMomentumNetwork):
    def __init__(self, input_dim, num_tickers, hidden_dim, dropout, num_heads, patch_len=16, stride=8,
                 use_static_ticker=True, **kwargs):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            **kwargs,
        )

        self.patch_len = patch_len
        self.stride = stride

        # Financial sequence representation
        self.seq_rep = SequenceRepresentation(
            input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker
        )

        # Patch projection with ticker-aware adaptation
        self.proj = nn.Linear(patch_len * input_dim, hidden_dim)
        self.ticker_modulation = nn.Linear(hidden_dim, hidden_dim)

        # Memory-efficient encoder from PatchCopy2
        from mom_trans_torch.models.PatchCopy2 import Encoder, EncoderLayer, AttentionLayer, FullAttention, Transpose

        self.patch_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 4, attention_dropout=dropout, output_attention=False),
                        hidden_dim, num_heads),
                    hidden_dim,
                    hidden_dim * 2,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(kwargs.get('num_layers', 2))
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(hidden_dim), Transpose(1, 2))
        )

        # Gating mechanisms for financial context integration
        self.gate_add_norm = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        # Get financial context
        fin_rep, _ = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)

        # Extract ticker embeddings for modulation
        ticker_embed = self.seq_rep.ticker_embedding(target_tickers) if self.use_static_ticker else None

        B, L, C = target_x.shape
        device = target_x.device

        # Add padding for causal constraints
        pad_len = self.patch_len - 1
        if pad_len > 0:
            pad = torch.zeros(B, pad_len, C, device=device, dtype=target_x.dtype)
            x = torch.cat([pad, target_x], dim=1)
        else:
            x = target_x

        # Process each timestep with financial context
        timestep_embeddings = []

        for t in range(L):
            t_padded = t + max(self.patch_len - t, 0)

            # Collect patches for this timestep
            t_patches = []
            residual = (t_padded - self.patch_len) % self.stride

            for k in range(((t_padded - self.patch_len) // self.stride) + 1):
                start = residual + (k * self.stride)
                if start < 0:
                    break

                patch = x[:, start:start + self.patch_len, :]
                flat_patch = patch.reshape(B, self.patch_len * C)

                # Project patch
                token = self.proj(flat_patch)

                # Modulate with ticker embedding if available
                if ticker_embed is not None:
                    token = token * torch.sigmoid(self.ticker_modulation(ticker_embed))

                t_patches.append(token)

            if not t_patches:
                token = self.proj(torch.zeros(B, self.patch_len * C, device=device))
                t_patches = [token]

            # Stack patches and process
            patches_t = torch.stack(t_patches, dim=1)

            # Process with memory-efficient encoder
            patches_t, _ = self.patch_transformer(patches_t)

            # Extract last token and integrate with financial context
            t_context = fin_rep[:, t:t + 1]
            timestep_embedding = self.gate_add_norm(patches_t[:, -1:], t_context)
            timestep_embedding = self.ffn(timestep_embedding)
            timestep_embeddings.append(timestep_embedding)

        return torch.cat(timestep_embeddings, dim=1)

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
