import torch
import torch.nn as nn
from transformers import AutoformerModel, AutoformerConfig


class CausalAutoformerBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 1. Project input to model dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. Autoformer config
        config = AutoformerConfig(
            d_model=hidden_dim,
            activation="gelu",
            dropout=dropout,
            enc_in=hidden_dim,
            dec_in=hidden_dim,
            c_out=hidden_dim,
            factor=5,
            n_heads=4,
            e_layers=2,
            d_ff=hidden_dim * 4,
        )

        # 3. Autoformer model (encoder-only usage here)
        self.autoformer = AutoformerModel(config)

        # 4. Output projection to input dimension
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def _causal_mask(self, size, device):
        # Creates an upper triangular mask (True = masked)
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward(self, x, target_tickers=None, pos_encoding_batch=None, variable_importance=False, **kwargs):
        """
        x: [B, L, D] (same as NLinear)
        Returns:
            - output: [B, L, D]
            - variable_importance: [D] if requested
        """
        B, L, D = x.shape

        # 1. Project input to hidden dimension
        x_proj = self.input_proj(x)  # [B, L, H]

        # 2. Create causal attention mask
        attn_mask = self._causal_mask(L, x.device)  # [L, L]

        # 3. Forward through Autoformer
        output = self.autoformer(inputs_embeds=x_proj, attention_mask=attn_mask).last_hidden_state  # [B, L, H]

        # 4. Project back to original input dimension
        out = self.decoder(output)  # [B, L, D]

        # 5. Variable importance if requested
        if variable_importance:
            importance = torch.norm(out, p=2, dim=1).mean(dim=0)  # [D]
            importance = importance / (importance.sum() + 1e-6)
        else:
            importance = None

        return out, importance

    def variable_importance(self, x, target_tickers=None, **kwargs):
        _, importance = self.forward(x, variable_importance=True)
        return importance  # shape: [D]
