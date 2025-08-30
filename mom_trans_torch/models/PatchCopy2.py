from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentationSimple, GatedResidualNetwork
import torch
import torch.nn as nn
from torch import Tensor
import math
import numpy as np
from torch.nn import functional as F
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False  # May help with memory issues


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class FullAttention(nn.Module):  # THIS IS THE RIGHT ONE!!!!!!
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        # self.padding_patch_layer = nn.ReplicationPad1d((0, padding)) # Replicating the last value padding times (for padding) and adds it to the right
        self.padding_patch_layer = nn.ConstantPad1d((padding, 0), 0)  # Adds zeros to the left side only

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Feature handling - use a learnable projection instead of simple averaging
        self.feature_projection = nn.Linear(1, 1)  # Maps each feature independently

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model) # Each time step in seq_len will have d_model dimensions. Each time step gets a unique positional encoding applied to all d_model dimensions equally

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x is expected to be [B, C, L]
        B, C, L = x.shape

        # Apply padding - [B, C, L+padding]
        x = self.padding_patch_layer(x)

        # Create patches using unfold
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [B, C, num_patches, patch_len]
        num_patches = x.size(2)

        # Reshape to [B*num_patches, C, patch_len] for feature projection
        x = x.permute(0, 2, 1, 3).reshape(B * num_patches, C, self.patch_len)

        # Project each feature separately - maintains feature importance
        # Reshape to handle each feature independently
        x = x.view(B * num_patches * C, 1, self.patch_len)
        x = self.feature_projection(x.transpose(1, 2)).transpose(1, 2)  # Learn feature weights
        x = x.view(B * num_patches, C, self.patch_len)

        # Combine features (weighted sum)
        x = x.sum(dim=1)  # [B*num_patches, patch_len]

        # Reshape back to [B, num_patches, patch_len]
        x = x.view(B, num_patches, self.patch_len)

        # Linear projection and positional encoding
        x = self.value_embedding(x)  # [B, num_patches, d_model]
        x = x + self.position_embedding(x)

        return self.dropout(x), None


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, nf, target_window, head_dropout=0.0, device="cpu"):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window).to(device)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTSTTS2(DeepMomentumNetwork):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf

    MUST HANDLE THAT THERE WILL BE A NEW USAGE OF PATCHING AT EVERY TIME STEP
    ENSURE THAT THE PADDING IS DONE CORRECTLY +: seems to be done correctly already within PatchEmbedding class
    ENSURE THAT THE OUTPUT DIMENSIONS ARE [B, L, H] INSTEAD OF  [B, 1, C] WHICH EFFECTIVELY BECOMES AFTER DOING ITERATIONS FOR EACH TIME STEP [B, L, C]
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
        self.patch_len = kwargs['patch_len']
        self.num_tickers = num_tickers
        self.stride = kwargs['stride']

        # Get device from kwargs or default to 'cuda' if available
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            hidden_dim, self.patch_len, self.stride, self.patch_len-1, dropout)   # use this iteratively (no need to create a new object instance every iteration) -> iteratively will mean that its first input has dimensions [B, 1, C] the second input will be [B, 2, C], third [B, 3, C],...., [B, L, C] and we can just reuse this same class instance

        # Encoder
        self.encoder = Encoder(   # Expect input and provide output of dimensions [B*C, P, H]
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, kwargs['attention_sparsity_factor'], attention_dropout=dropout,
                                      output_attention=False), hidden_dim, num_heads),
                    hidden_dim,
                    hidden_dim*2,
                    dropout=dropout,
                    activation='gelu'
                ) for l in range(kwargs['transformer_layers'])
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(hidden_dim), Transpose(1,2))
        )

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        # Make sure input is on the correct device
        target_x = target_x.to(self.device)

        # Initialize tensor to collect all outputs
        batch_size, seq_len, channels = target_x.shape
        all_outputs = torch.zeros(batch_size, seq_len, self.hidden_dim, device=target_x.device)

        for t in range(seq_len):
            x_step = target_x[:, :t + 1, :]  # [B, t+1, C]

            # Transpose for patching (patch embedding expects [B, C, L])
            x_step = x_step.transpose(1, 2)  # [B, C, t+1]

            # Patching and embedding
            enc_out, _ = self.patch_embedding(x_step)  # [B, P, H]

            # Encoder
            enc_out, _ = self.encoder(enc_out)  # [B, P, H]

            # # Reshape: no need to handle multiple channels now
            # enc_out = enc_out.unsqueeze(1)  # [B, 1, P, H]
            # enc_out = enc_out.permute(0, 3, 1, 2)  # [B, H, 1, P]
            # 
            # # Prediction Head
            # head_nf = enc_out.size(-1)
            # head = FlattenHead(head_nf, 1,
            #                    head_dropout=self.dropout, device=self.device)
            # 
            # # Decoder - shape will be [B, 1, H] now
            # dec_out = head(enc_out)  # [B, 1, H]

            # # Store the output
            # all_outputs[:, t, :] = dec_out.squeeze(-1)
            all_outputs[:, t, :] = enc_out[:, -1, :]  # Take the last patch's representation

        return all_outputs

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
