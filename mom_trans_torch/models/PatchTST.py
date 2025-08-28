from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentationSimple, GatedResidualNetwork
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import math
from transformers import PatchTSTConfig, PatchTSTModel
from torch.nn import functional as F


class ChunkedSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1, chunk_size=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.shape

        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Process in chunks to save memory
        output = torch.zeros_like(q)

        # Process attention in chunks
        for i in range(0, seq_len, self.chunk_size):
            # Get current chunk
            end_idx = min(i + self.chunk_size, seq_len)
            q_chunk = q[:, :, i:end_idx]

            # Compute attention scores for this chunk only
            # Shape: [batch_size, num_heads, chunk_size, seq_len]
            scores = torch.matmul(q_chunk, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

            # Apply softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention weights to values
            # Shape: [batch_size, num_heads, chunk_size, head_dim]
            output[:, :, i:end_idx] = torch.matmul(attn_weights, v)

        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(output)

        return output


class ChunkedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, chunk_size=128):
        super().__init__()
        self.self_attn = ChunkedSelfAttention(d_model, nhead, dropout, chunk_size)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.GELU()

    def forward(self, src):
        # Self-attention block with residual connection
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout(src2)

        # Feed forward block with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout(src2)

        return src


class ChunkedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, chunk_size=128):
        super().__init__()
        self.self_attn = ChunkedSelfAttention(d_model, nhead, dropout, chunk_size)

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Activation
        self.activation = nn.GELU()

    def forward(self, src):
        # Self-attention block with residual connection
        src2 = self.norm1(src)
        src2 = self.self_attn(src2)
        src = src + self.dropout(src2)

        # Feed forward block with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout(src2)

        return src


class CustomStridePatchTSTModel(nn.Module):
    def __init__(self, config: PatchTSTConfig):
        super().__init__()
        self.config = config

        # Handle multi-scale configuration
        self.patch_lens = config.patch_length if isinstance(config.patch_length, list) else [config.patch_length]
        self.strides = config.patch_stride if isinstance(config.patch_stride, list) else [config.patch_stride]
        assert len(self.patch_lens) == len(self.strides), "patch_lens and strides must be the same length"

        self.hidden_dim = config.d_model
        self.input_dim = config.num_input_channels
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.num_layers = config.num_hidden_layers

        # Create a separate projection layer for each patch scale
        self.proj_layers = nn.ModuleList()
        for patch_len in self.patch_lens:
            self.proj_layers.append(nn.Linear(patch_len * self.input_dim, self.hidden_dim))

        # Learnable positional embeddings for each scale
        self.pos_embeddings = nn.ParameterList()
        self.max_patches_per_scale = []

        for patch_len, stride in zip(self.patch_lens, self.strides):
            max_patches_for_scale = ((config.context_length - patch_len) // max(1, stride)) + 1
            if max_patches_for_scale <= 0:
                max_patches_for_scale = 1
            self.max_patches_per_scale.append(max_patches_for_scale)
            pos_embed = nn.Parameter(torch.randn(1, max_patches_for_scale, self.hidden_dim))
            self.pos_embeddings.append(pos_embed)

        # Transformer layers - shared across all scales
        self.patch_transformer = nn.ModuleList([
            ChunkedTransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_dim,
                dropout=self.dropout,
                chunk_size=64  # Adjust based on available memory
            ) for _ in range(self.num_layers)
        ])

        # Layer norm and output projection
        self.ln = nn.LayerNorm(self.hidden_dim)

        # Fusion layer to combine multi-scale embeddings
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * len(self.patch_lens), self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

    def forward(self, x, **kwargs):
        """
        Create multi-scale patches for causal time series forecasting.
        Returns one embedding per timestep by processing patches from multiple scales.
        """
        B, L, C = x.shape
        device = x.device

        # Determine max padding needed across all patch sizes
        max_pad_len = max(pl - 1 for pl in self.patch_lens)
        if max_pad_len > 0:
            pad = torch.zeros(B, max_pad_len, C, device=device, dtype=x.dtype)
            x_padded = torch.cat([pad, x], dim=1)  # [B, L + max_pad_len, C]
        else:
            x_padded = x
            max_pad_len = 0

        # Store embeddings for each timestep
        all_timestep_embeddings = []

        # Process each timestep in the original sequence
        for t in range(L):
            scale_embeddings = []  # Will store the embedding from each scale for this timestep

            # Position in the padded sequence for this timestep
            t_padded = t + max_pad_len

            # Process each scale (patch length) independently
            for scale_idx, (patch_len, stride) in enumerate(zip(self.patch_lens, self.strides)):
                proj_layer = self.proj_layers[scale_idx]
                patches_for_scale = []

                # Generate all patches that end at or before t_padded for this scale
                # We start from the patch that ends exactly at t_padded and go backwards
                current_end = t_padded

                while current_end >= patch_len - 1:  # Ensure we have a complete patch
                    start_idx = current_end - patch_len + 1
                    if start_idx < 0:
                        break

                    patch = x_padded[:, start_idx:start_idx + patch_len, :]  # [B, patch_len, C]
                    flat_patch = patch.reshape(B, -1)  # [B, patch_len * C]
                    token = proj_layer(flat_patch)  # [B, hidden_dim]
                    patches_for_scale.append(token)

                    # Move to the previous patch according to stride
                    current_end -= stride

                # If no patches were found (shouldn't happen with proper padding), use zeros
                if not patches_for_scale:
                    zero_patch = torch.zeros(B, patch_len * C, device=device)
                    token = proj_layer(zero_patch)
                    patches_for_scale = [token]

                # Reverse to get chronological order (oldest first, newest last)
                patches_for_scale.reverse()

                # Stack patches for this scale: [B, num_patches, hidden_dim]
                patches_tensor = torch.stack(patches_for_scale, dim=1)
                num_patches = patches_tensor.size(1)

                # Add scale-specific positional embeddings
                if num_patches <= self.max_patches_per_scale[scale_idx]:
                    pos_embed = self.pos_embeddings[scale_idx][:, :num_patches, :]
                    patches_tensor = patches_tensor + pos_embed

                # Apply transformer layers
                for layer in self.patch_transformer:
                    patches_tensor = layer(patches_tensor)

                # Take the representation from the most recent patch (last token)
                scale_embedding = patches_tensor[:, -1, :]  # [B, hidden_dim]
                scale_embeddings.append(scale_embedding)

            # Combine embeddings from all scales for this timestep
            if len(scale_embeddings) > 1:
                # Concatenate embeddings from all scales
                combined_embedding = torch.cat(scale_embeddings, dim=-1)  # [B, hidden_dim * num_scales]
                # Fuse multi-scale information
                final_embedding = self.scale_fusion(combined_embedding)  # [B, hidden_dim]
            else:
                # Single scale case
                final_embedding = scale_embeddings[0]

            all_timestep_embeddings.append(final_embedding.unsqueeze(1))  # [B, 1, hidden_dim]

        # Combine all timestep embeddings
        output = torch.cat(all_timestep_embeddings, dim=1)  # [B, L, hidden_dim]
        output = self.ln(output)

        return output


class CustomStridePatchTSTModelOld(PatchTSTModel):
    def __init__(self, config: PatchTSTConfig):
        super().__init__(config)

        # self.patch_len = config.patch_length
        # self.stride = config.patch_stride
        self.patch_lens = config.patch_length if isinstance(config.patch_length, list) else [config.patch_length]
        self.strides = config.patch_stride if isinstance(config.patch_stride, list) else [config.patch_stride]
        assert len(self.patch_lens) == len(self.strides), "patch_lens and strides must be the same length"
        self.hidden_dim = config.d_model
        self.input_dim = config.num_input_channels
        self.num_heads = config.num_attention_heads
        self.dropout = config.dropout
        self.num_layers = config.num_hidden_layers
        # assert self.stride <= self.patch_len, "stride must be <= patch_len for causality."

        # First project features to hidden_dim, then create patches
        # self.feature_proj = nn.Linear(self.input_dim, self.hidden_dim)

        # Flatten patch_len * input_dim → hidden_dim
        self.proj = nn.Linear(self.patch_len * self.input_dim, self.hidden_dim)

        # Positional embeddings
        max_patches = ((config.context_length - self.patch_len) // max(1, self.stride)) + 1
        if max_patches <= 0:
            max_patches = 1
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, self.hidden_dim))
        # encoder_layer = TransformerEncoderLayer(
        #     d_model=self.hidden_dim,
        #     nhead=self.num_heads,
        #     dim_feedforward=4 * self.hidden_dim,
        #     dropout=self.dropout,
        #     batch_first=True
        # )
        # self.patch_transformer = TransformerEncoder(encoder_layer,
        #                                             num_layers=self.num_layers)
        self.patch_transformer = nn.ModuleList([
            ChunkedTransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=4 * self.hidden_dim,
                dropout=self.dropout,
                chunk_size=64  # Adjust based on available memory
            ) for _ in range(self.num_layers)
        ])

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
            t_padded = t + max(self.patch_len - t, 0)

            # Collect all valid patches for this timestep
            t_patches = []
            residual = (t_padded - self.patch_len) % self.stride
            for k in range(((t_padded - self.patch_len) // self.stride) + 1):
                start = residual + (k * self.stride)
                # start = t_padded - k * self.stride - self.patch_len + 1
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

            # Apply each transformer layer in sequence
            for layer in self.patch_transformer:
                patches_t = layer(patches_t)

            # Take the last token after processing through all transformer layers
            timestep_embedding = patches_t[:, -1]  # [B, hidden_dim]
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
        self.stride = min(kwargs['stride'], self.patch_len)

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
