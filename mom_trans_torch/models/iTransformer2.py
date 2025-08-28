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


class iTransformer2(DeepMomentumNetwork):
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
