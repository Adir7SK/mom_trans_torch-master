from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation
# from iTransformer import iTransformer  # official repo
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, Union
from mom_trans_torch.models.common import (
    for_masked_fill,
    GateAddNorm,
    # GatedResidualNetwork,
)
from torch import Tensor


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) from the TFT paper."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1,
                 context_size: Optional[int] = None):
        super().__init__()
        self.output_size = output_size
        self.context_size = context_size

        # First linear layer + ELU
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Context projection if provided (for conditional computation)
        if context_size is not None:
            self.context_proj = nn.Linear(context_size, hidden_size, bias=False)

        # Gating mechanism and second linear
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Initial projection
        output = self.linear1(x)
        if context is not None:
            # Add context information
            output = output + self.context_proj(context)
        output = F.elu(output)
        output = self.dropout(output)

        # Gating mechanism
        gate = torch.sigmoid(self.gate(output))
        output = self.linear2(output)
        output = self.dropout(output)

        # Residual connection and layer norm
        if x.shape[-1] == self.output_size:
            output = gate * output + (1 - gate) * x
        else:
            output = gate * output
        output = self.layer_norm(output)
        return output


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) from the TFT paper, made causal for time series."""

    def __init__(self, variable_sizes: Dict[str, int], hidden_size: int, dropout: float = 0.1,
                 context_size: Optional[int] = None, prescalers: Optional[nn.ModuleDict] = None,
                 kernel_size: Union[int, list] = 3):
        super().__init__()
        self.variable_sizes = variable_sizes
        self.num_variables = len(variable_sizes)
        self.hidden_size = hidden_size
        self.prescalers = prescalers
        self.dropout_rate = dropout

        # Flattened GRN for each variable
        self.grn_list = nn.ModuleList()
        for var, size in variable_sizes.items():
            self.grn_list.append(
                GatedResidualNetwork(size, hidden_size, hidden_size, dropout=dropout, context_size=context_size)
            )

        # GRN to calculate variable selection weights
        self.weights_grn = GatedResidualNetwork(
            self.num_variables * hidden_size, hidden_size, self.num_variables, dropout=dropout,
            context_size=context_size
        )

        self.num_vars_times_hidden = self.num_variables * hidden_size

        # --- Causal Convolution Setup ---
        # Handle kernel_size configuration
        if isinstance(kernel_size, int):
            kernel_sizes = [kernel_size]
        else:
            kernel_sizes = kernel_size  # Assume it's already a list

        self.causal_convs = nn.ModuleList()
        self.causal_pads = nn.ModuleList()

        for k in kernel_sizes:
            # Pad the beginning of the sequence to make convolution causal
            self.causal_pads.append(nn.ConstantPad1d((k - 1, 0), 0))
            # Depthwise convolution: each channel processed independently
            self.causal_convs.append(
                nn.Conv1d(
                    in_channels=self.num_vars_times_hidden,
                    out_channels=self.num_vars_times_hidden,
                    kernel_size=k,
                    padding=0,
                    groups=self.num_vars_times_hidden,
                    bias=False  # Often beneficial in depthwise convolutions
                )
            )

        # If using multiple kernels, add a fusion layer to combine them
        if len(kernel_sizes) > 1:
            self.multi_scale_fusion = nn.Linear(len(kernel_sizes) * self.num_vars_times_hidden,
                                                self.num_vars_times_hidden)
        else:
            self.multi_scale_fusion = None

        # Dropout for the convolution outputs
        self.conv_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Dict[str, torch.Tensor], context: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
        # Pre-scale each variable if prescalers are provided
        processed_vars = []
        for i, (var_name, var_tensor) in enumerate(inputs.items()):
            if self.prescalers is not None and str(i) in self.prescalers:
                var_tensor = self.prescalers[str(i)](var_tensor)
            processed_vars.append(var_tensor)

        # Process each variable through its own GRN
        grn_outputs = []
        for i, var_tensor in enumerate(processed_vars):
            grn_outputs.append(self.grn_list[i](var_tensor, context=context))

        # Concatenate all processed variables
        concat_output = torch.cat(grn_outputs, dim=-1)  # [B, T, Num_Vars * Hidden]

        # --- Causal Weight Calculation ---
        # Conv1d expects [Batch, Channels, Time], so we permute dimensions
        concat_output_permuted = concat_output.transpose(1, 2)  # [B, Num_Vars*H, T]

        multi_scale_outputs = []
        for causal_pad, causal_conv in zip(self.causal_pads, self.causal_convs):
            # Apply causal padding to the time dimension (left padding)
            concat_output_padded = causal_pad(concat_output_permuted)  # [B, Num_Vars*H, T + (k-1)]
            # Process with causal depthwise convolution
            conv_output = causal_conv(concat_output_padded)  # [B, Num_Vars*H, T]
            conv_output = self.conv_dropout(conv_output)  # Apply dropout
            multi_scale_outputs.append(conv_output)

        # Handle both single-scale and multi-scale cases
        if self.multi_scale_fusion is not None:
            # Multi-scale: Concatenate outputs along the channel dimension
            multi_scale_concat = torch.cat(multi_scale_outputs, dim=1)  # [B, (Num_Kernels * Num_Vars*H), T]
            # Fuse the multi-scale information: [B, C, T] -> [B, T, C] -> Linear -> [B, T, H'] -> [B, H', T]
            fused_conv_output = self.multi_scale_fusion(multi_scale_concat.transpose(1, 2)).transpose(1, 2)
            conv_output_final = F.gelu(fused_conv_output)
        else:
            # Single-scale: Just use the output from the first (and only) convolution
            conv_output_final = F.gelu(multi_scale_outputs[0])

        # Permute back to [B, T, Num_Vars*H] for the GRN
        conv_output = conv_output_final.transpose(1, 2)  # [B, T, Num_Vars*H]

        # Now apply the GRN. This is causal because its input for time t
        # only contains information up to time t.
        flat_weights = self.weights_grn(conv_output, context=context)  # [B, T, Num_Vars]
        weights = torch.softmax(flat_weights, dim=-1)

        # Apply weights and sum
        weighted_outputs = []
        for i, grn_out in enumerate(grn_outputs):
            weighted_outputs.append(grn_out * weights[..., i:i + 1])
        output = torch.sum(torch.stack(weighted_outputs, dim=-1), dim=-1)

        return output, weights


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for the time dimension."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


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

        # Financial sequence representation
        self.seq_rep = SequenceRepresentation(
            input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker
        )

        # Feature positional encoding
        self.feature_pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Enhanced feature projection with normalization
        self.feature_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Enhanced transformer with gating
        num_layers = kwargs.get('transformer_layers', 2)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dim, num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Cross-feature attention for modeling relationships between features
        self.cross_feature_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Memory mechanism for important financial patterns
        num_memories = kwargs.get('num_memories', 8)
        self.memory_keys = nn.Parameter(torch.randn(1, num_memories, hidden_dim))
        self.memory_values = nn.Parameter(torch.randn(1, num_memories, hidden_dim))

        # Gating mechanisms
        self.gate_add_norm_mha = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, dropout=dropout)
        self.gate_add_norm_block = GateAddNorm(hidden_dim, dropout=dropout)

        # Adaptive feature fusion
        self.feature_fusion = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # Time feature embedder for market regime awareness
        self.time_feature_proj = nn.Linear(kwargs.get('time_feat_dim', 4), hidden_dim)
        self.has_time_features = kwargs.get('has_time_features', False)

        # Volatility awareness layer (financial-specific)
        self.volatility_proj = nn.Linear(1, hidden_dim)
        self.vol_gate = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        # Get financial feature representation
        rep_financial, lstm_state = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)

        # iTransformer style processing (transpose for feature-as-token)
        B, T, F = target_x.shape
        x_transposed = target_x.transpose(1, 2)  # [B, F, T]
        device = target_x.device

        # Project each feature time series to embedding dimension
        x_tokens = []
        for i in range(F):
            feature_series = x_transposed[:, i:i + 1, :]  # [B, 1, T]
            token = self.feature_projection(feature_series.transpose(1, 2))  # [B, T, d_model]
            x_tokens.append(token)

        x_encoded = torch.stack(x_tokens, dim=1)  # [B, F, T, d_model]

        # Calculate feature volatility for adaptive weighting
        feature_volatility = torch.std(target_x, dim=1, keepdim=True)  # [B, 1, F]
        vol_embedding = self.volatility_proj(feature_volatility.transpose(1, 2))  # [B, F, hidden_dim]

        # Process each feature sequence separately
        processed_features = []
        for i in range(F):
            # Get this feature's sequence
            feat_seq = x_encoded[:, i]  # [B, T, hidden_dim]

            # Add positional encoding
            feat_seq = self.feature_pos_encoding(feat_seq)  # [B, T, hidden_dim]

            # Create causal mask for transformer
            # This ensures predictions only use past information
            causal_mask = torch.ones(T, T, device=device).triu_(1).bool()

            # Process with transformer encoder (with causal masking)
            feat_transformed = self.transformer_encoder(
                feat_seq,
                mask=causal_mask
            )  # [B, T, hidden_dim]

            processed_features.append(feat_transformed)

        # Stack processed features
        feature_stack = torch.stack(processed_features, dim=1)  # [B, F, T, hidden_dim]

        # For each timestep, integrate across features with cross-attention
        timestep_embeddings = []
        for t in range(T):
            # Extract features at this timestep
            features_t = feature_stack[:, :, t]  # [B, F, hidden_dim]

            # Use cross-feature attention to model relationships
            # Apply causal mask in feature dimension if needed
            features_t_context, _ = self.cross_feature_attention(
                features_t, features_t, features_t
            )  # [B, F, hidden_dim]

            # Weight by volatility
            import torch.nn.functional as F
            tt = vol_embedding @ features_t_context.transpose(-1, -2)
            vol_weights = F.softmax(tt / math.sqrt(self.hidden_dim),
                                    dim=-1)
            vol_weighted = torch.bmm(vol_weights, features_t_context)  # [B, F, hidden_dim]

            # Pool features (weighted mean)
            timestep_emb = vol_weighted.mean(dim=1)  # [B, hidden_dim]

            # Get financial context for this timestep
            t_financial = rep_financial[:, t:t + 1]  # [B, 1, hidden_dim]

            # Integrate with gating
            t_integrated = self.gate_add_norm_mha(
                timestep_emb.unsqueeze(1),
                t_financial
            )  # [B, 1, hidden_dim]

            timestep_embeddings.append(t_integrated)

        # Combine timestep embeddings
        sequence_rep = torch.cat(timestep_embeddings, dim=1)  # [B, T, hidden_dim]

        # Process with FFN
        sequence_rep = self.ffn(sequence_rep)

        # Apply memory mechanism for financial patterns
        memory_query = sequence_rep @ self.memory_keys.transpose(-1, -2)
        memory_attn = F.softmax(memory_query / math.sqrt(self.hidden_dim), dim=-1)
        memory_output = torch.bmm(memory_attn, self.memory_values.expand(B, -1, -1))

        # Final integration with LSTM state
        final_rep = self.gate_add_norm_block(sequence_rep + memory_output, lstm_state)

        # Add time features if available (market regime awareness)
        if self.has_time_features and 'time_features' in kwargs and kwargs['time_features'] is not None:
            time_embedding = self.time_feature_proj(kwargs['time_features'])
            final_rep = final_rep + time_embedding

        return final_rep

    def variable_importance(
            self,
            target_x: torch.Tensor,
            target_tickers: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        Computes feature-level importance scores using attention weights
        and gradient-based attribution.
        """
        # Store attention weights during forward pass
        B, T, F = target_x.shape
        device = target_x.device

        # Register hooks to extract attention weights
        attention_weights = []

        def hook_fn(module, input, output):
            # Extract attention weights from transformer layers
            attention_weights.append(output[1])

        hooks = []
        for layer in self.transformer_encoder.layers:
            hooks.append(layer.self_attn.register_forward_hook(hook_fn))

        # Forward pass
        with torch.enable_grad():
            if not target_x.requires_grad:
                target_x = target_x.clone().detach().requires_grad_(True)

            rep = self.forward_candidate_arch(target_x, target_tickers, **kwargs)

            # Use attention weights to calculate feature importance
            if attention_weights:
                # Average attention weights across layers and heads
                avg_attention = torch.cat([w.mean(dim=1) for w in attention_weights], dim=0).mean(dim=0)
                # Sum attention scores for each feature
                feature_scores = avg_attention.sum(dim=1)  # [F]
            else:
                feature_scores = torch.ones(F, device=device)

            # Also use gradient-based approach
            rep_mean = rep.mean(dim=1)  # [B, hidden_dim]
            pred = self.output_head(rep_mean)  # [B, output_dim]
            loss = pred.abs().mean()
            loss.backward()

            # Gradients with respect to inputs
            grads = target_x.grad.abs()  # [B, T, F]
            grad_scores = grads.mean(dim=(0, 1))  # [F]

            # Combine both approaches
            importance = feature_scores * grad_scores
            importance = importance / (importance.sum() + 1e-8)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return importance


class iTransformerOLD(DeepMomentumNetwork):
    """
    Enhanced iTransformer for sequence representation.
    Input: [batch_size, seq_len, num_features]
    Output: [batch_size, seq_len, hidden_dim]
    """

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
        self.num_heads = num_heads

        # 1. Variable Selection Network (Pre-processor)
        # Treat each original feature as an input variable to the VSN
        variable_sizes = {str(i): 1 for i in range(input_dim)}
        self.vsn = VariableSelectionNetwork(
            variable_sizes,
            hidden_dim,
            dropout=dropout,  # Use the provided dropout
            context_size=hidden_dim if use_static_ticker else None,
            kernel_size=kwargs['kernel_size']
        )

        # 2. Static Ticker Context (Optional)
        if use_static_ticker and num_tickers is not None:
            self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
            self.static_context_grn = GatedResidualNetwork(
                hidden_dim, hidden_dim, hidden_dim, dropout=dropout
            )
        else:
            self.use_static_ticker = False

        # 3. iTransformer Core
        # a) Inversion and Embedding: Now handled per time step by the VSN output
        # b) Positional Encoding for the time dimension
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)

        # c) Transformer Encoder (FULL attention over features)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.seq_len,  # This should be the sequence length
            nhead=self.num_heads,  # Note: nhead must divide seq_len evenly!
            dim_feedforward=self.seq_len * 4,  # Optional, but good practice to scale with d_model
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        # No mask needed -> full attention over features
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=kwargs['transformer_layers'])

        # 4. Optional Output Projection (to ensure output is hidden_dim)
        self.output_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, num_features]
            static_tickers: Optional tensor of ticker indices of shape [batch_size]
        Returns:
            output: Representation tensor of shape [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, num_features = target_x.shape

        # Prepare static context from ticker embedding if available
        if self.use_static_ticker and target_tickers is not None:
            ticker_emb = self.ticker_embedding(target_tickers)  # [B, H]
            context = self.static_context_grn(ticker_emb)  # [B, H]
            # Expand context for VSN: [B, H] -> [B, Seq_Len, H]
            context_seq = context.unsqueeze(1).expand(-1, seq_len, -1)
        else:
            context_seq = None

        # 1. Variable Selection Network
        # Prepare inputs for VSN: split features into a dictionary
        inputs = {str(i): target_x[..., i:i + 1] for i in range(num_features)}  # Each value is [B, T, 1]
        vsn_out, variable_weights = self.vsn(inputs, context=context_seq)  # vsn_out: [B, T, H]

        # 2. Add positional encoding to the time dimension
        vsn_out = self.pos_encoder(vsn_out)  # [B, T, H]

        # 3. iTransformer Core Processing
        # Invert: [B, T, H] -> [B, H, T]
        inverted_sequence = vsn_out.transpose(1,
                                              2)  # Now 'H' is the feature dim, 'T' is the sequence dim for the transformer

        # Pass through transformer encoder (applies self-attention over the 'H' dimension)
        # transformer expects [B, Seq_Len, Feat] -> our 'Seq_Len' is H (hidden_dim), our 'Feat' is T (seq_len)
        transformer_in = inverted_sequence
        transformer_out = self.transformer_encoder(transformer_in)  # [B, H, T]
        transformer_out = transformer_out + transformer_in # Residual Connection

        # Invert back: [B, H, T] -> [B, T, H]
        transformer_out = transformer_out.transpose(1, 2)  # [B, T, H]

        # 4. Final output projection
        projected = self.output_projection(transformer_out)
        gate_input = torch.cat([transformer_out, projected], dim=-1)  # [B, T, 2*H]
        gate = torch.sigmoid(self.output_gate(gate_input))
        output = gate * projected + (1 - gate) * transformer_out

        return output

    def variable_importance(
            self,
            target_x: torch.Tensor,
            target_tickers: torch.Tensor = None,
            **kwargs
    ) -> torch.Tensor:
        """
        THIS IS A BAD VARIABLE IMPORTANCE!!!!! BETTER AS CHAT-GPT
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
