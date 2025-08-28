import torch
import torch.nn as nn
from transformers import InformerConfig # InformerModel
from transformers.models.informer.modeling_informer import InformerModel  ###
from mom_trans_torch.models.dmn import DeepMomentumNetwork


# class CausalInformerBaseline(DeepMomentumNetwork):
#     def __init__(
#             self,
#             input_dim: int,
#             num_tickers: int,
#             hidden_dim: int,
#             dropout: float,
#             num_heads: int,
#             transformer_layers: int,
#             attention_sparsity_factor: int,
#             context_seq_len: int,
#             use_static_ticker: bool = True,
#             auto_feature_num_input_linear=0,
#             **kwargs,
#     ):
#         super().__init__(
#             input_dim=input_dim,
#             num_tickers=num_tickers,
#             hidden_dim=hidden_dim,
#             dropout=dropout,
#             use_static_ticker=use_static_ticker,
#             auto_feature_num_input_linear=auto_feature_num_input_linear,
#             **kwargs,
#         )
#
#         # 1. Project input to hidden dim
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
#
#         # 2. Informer Config (only encoder is used here)
#         cfg = InformerConfig(
#             d_model=hidden_dim,
#             activation="gelu",
#             dropout=dropout,
#             enc_in=hidden_dim,
#             dec_in=hidden_dim,
#             c_out=hidden_dim,
#             factor=attention_sparsity_factor,
#             n_heads=num_heads,
#             e_layers=transformer_layers,
#             d_ff=hidden_dim * 4,
#             context_length=self.seq_len,
#             prediction_length=0,        # We'll return full sequence
#         )
#
#         self.informer = InformerModel(cfg)
#         self.decoder = nn.Linear(hidden_dim, input_dim)
#
#     def _causal_mask(self, size, device):
#         # Create a mask of shape [size, size] with upper triangle masked
#         return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()
#
#     def forward_candidate_arch(self, target_x, target_tickers=None, pos_encoding_batch=None, variable_importance=False, **kwargs):
#         B, L, D = target_x.shape
#         x_proj = self.input_proj(target_x)  # [B, L, H]
#
#         # No future known inputs, but use causal decoder
#         outputs = self.informer(
#             past_values=x_proj,  # [B, L, H]
#             past_time_features=torch.zeros(B, L, self.hidden_dim, device=x_proj.device),
#             past_observed_mask=torch.ones(B, L, dtype=torch.bool, device=target_x.device),  # same shape for time steps    torch.ones_like(x_proj[..., :1], dtype=torch.bool)
#             future_values=None,
#             future_time_features=None,
#             # future_observed_mask=None,
#             static_categorical_features=None,
#             static_real_features=None,
#         )
#         hidden = outputs.last_hidden_state  # [B, L, H]
#         decoded = self.decoder(hidden)      # [B, L, D]
#
#         if variable_importance:
#             imp = torch.norm(decoded, p=2, dim=1).mean(dim=0)
#             imp = imp / (imp.sum() + 1e-6)
#         else:
#             return decoded
#         return imp
#
#     def variable_importance(self, x, target_tickers=None, **kwargs):
#         importance = self.forward(x, variable_importance=True)
#         return importance  # shape: [D]


class CausalInformerBaseline(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        transformer_layers: int,
        attention_sparsity_factor: int,
        context_seq_len: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        # Project input to hidden dimension
        # self.input_proj = nn.Linear(input_dim, hidden_dim)
        # self.to_context_len_projection = nn.Linear(self.seq_len, hidden_dim)

        # Informer configuration (encoder-only for causality)
        cfg = InformerConfig(
            input_size=input_dim, # hidden_dim
            d_model=hidden_dim,
            activation="gelu",
            dropout=dropout,
            enc_in=input_dim,  # hidden_dim
            dec_in=input_dim,  # hidden_dim
            c_out=hidden_dim,
            factor=attention_sparsity_factor,   # sparsity factor for attention = factor * log(seq_len)
            n_heads=num_heads,
            e_layers=transformer_layers,
            d_ff=hidden_dim * 4,
            context_length=self.seq_len, #hidden_dim, #context_seq_len,  # context window size
            prediction_length=0,  # No future predictions
            distil=False, # distillation - halves the sequence length between layers (reduce memory and highlights important features)
            num_static_real_features=0,
            num_static_categorical_features=0,
        )
        # inputs_embeds.shape[-1] = enc_in = dec_in = c_out = d_model = context_length = hidden_dim
        InformerModel.from_pretrained = None

        self.informer = InformerModel(config=cfg)

        # Monkey-patch the method to avoid static feature errors
        def patched_create_network_inputs(
                this,
                past_values,
                past_time_features,
                past_observed_mask,
                future_values,
                future_time_features,
                static_categorical_features=None,
                static_real_features=None
        ):
            context = past_values
            observed_context = past_observed_mask
            time_feat = past_time_features
            _, loc, scale = this.scaler(context, observed_context)

            transformer_inputs = (context - loc) / scale
            return transformer_inputs, loc, scale, None  # No static_feat

        # Apply patch
        self.informer.create_network_inputs = patched_create_network_inputs.__get__(self.informer,
                                                                                    self.informer.__class__)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def _causal_mask(self, size, device):
        """
        Create a causal mask to prevent attention to future time steps.
        """
        return ~torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    def forward_candidate_arch(
        self, target_x, target_tickers=None, pos_encoding_batch=None, variable_importance=False, **kwargs
    ):
        """
        Forward pass for the Informer model.
        Ensures causality by applying a causal mask.
        """
        B, L, D = target_x.shape
        # x_proj = self.input_proj(target_x)  # [B, L, H]

        # x_proj = self.to_context_len_projection(x_proj.transpose(1, 2)).transpose(1, 2)
        x_proj = target_x

        # Apply causal mask
        # causal_mask = self._causal_mask(L, x_proj.device).unsqueeze(0).expand(B, -1, -1)
        # causal_mask = self._causal_mask(self.hidden_dim, x_proj.device).unsqueeze(0).expand(B, -1, -1)      # Casual attention mask - generally good for attention layers but not used for Informer

        past_observed_mask = torch.ones(B, L, device=x_proj.device)  # [B, L]

        # ✅ Patch: empty tensor with [B, 0] shape (no features, but not None)
        static_real_features = torch.empty(B, 0, device=x_proj.device)

        # Pass through Informer encoder
        outputs = self.informer(
            past_values=x_proj,  # [B, L, H]  past_values
            past_time_features=torch.zeros(B, L, 1, device=x_proj.device),  # We don't have time features to encode, and therefore we are making primitive tensor that will only match the batch_size and seq_len
            # past_time_features=torch.zeros(B, self.hidden_dim, self.hidden_dim, device=x_proj.device),
            past_observed_mask=torch.ones(B, L, 1, device=x_proj.device),    # causal_mask         , x_proj.shape[-1]
            future_values=None,
            future_time_features=None,
            static_real_features=None, #static_real_features, # None, #torch.zeros(B, self.hidden_dim, device=x_proj.device), #None,
            static_categorical_features=None,
        )
        hidden = outputs.last_hidden_state  # [B, L, H]
        decoded = self.decoder(hidden)  # [B, L, D]

        if variable_importance:
            imp = torch.norm(decoded, p=2, dim=1).mean(dim=0)
            imp = imp / (imp.sum() + 1e-6)
        else:
            return decoded
        return imp

    def variable_importance(self, x, target_tickers=None, **kwargs):
        """
        Compute variable importance based on the decoder output.
        """
        importance = self.forward_candidate_arch(x, variable_importance=True)
        return importance  # shape: [D]
