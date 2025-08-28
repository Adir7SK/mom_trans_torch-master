from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation, SequenceRepresentationSimple
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from mom_trans_torch.models.common import (
    for_masked_fill,
    GateAddNorm,
    GatedResidualNetwork,
)
import torch
import torch.nn as nn


class TFTBaseline(DeepMomentumNetwork):
    """
    This class is different from MomentumTransformer class in the following aspects:
    Component               MomentumTranformer
    self-attention:
    """
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,
        **kwargs,
    ):
        # self.output_signal_weights = output_signal_weights
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        self.seq_rep = SequenceRepresentation(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_tickers=num_tickers,
            fuse_encoder_input=False,  # same as in LstmBaseline
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,  # can be made a hyperparameter
        )

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        # Use shared input encoder
        x, _ = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch
        )  # x: [B, T, H]

        x_encoded = self.transformer_encoder(x)  # [B, T, H]
        output = self.output_proj(x_encoded)     # [B, T, H] or [B, T, 1] if regression and output_proj is Linear(hidden_dim, 1)

        return output

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)


class TFTStyleTransformer(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        num_heads: int,
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

        self.seq_rep = SequenceRepresentation(
            input_dim, hidden_dim, dropout, num_tickers,
            fuse_encoder_input=False,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
        )

        self.self_att = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.gate_add_norm_attn = GateAddNorm(hidden_dim, dropout=dropout)
        self.ffn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.gate_add_norm_out = GateAddNorm(hidden_dim, dropout=dropout)

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        representation, lstm_hidden_state = self.seq_rep(target_x, target_tickers, pos_encoding_batch=pos_encoding_batch)

        # attention mask for causality
        B, T, H = representation.shape
        attn_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(representation.device)

        attn_out, _ = self.self_att(representation, representation, representation, attn_mask=attn_mask)
        gated_attn = self.gate_add_norm_attn(attn_out, representation)

        ffn_out = self.ffn(gated_attn)
        output = self.gate_add_norm_out(ffn_out, lstm_hidden_state)

        return output

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)


class TFT(DeepMomentumNetwork):
    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            dropout: float,
            num_heads: int,
            use_static_ticker: bool = True,
            auto_feature_num_input_linear=0,  # TODO maybe diable this for now..
            # output_signal_weights=False,
            **kwargs
    ):
        fuse_encoder_input = False
        # self.output_signal_weights = output_signal_weights
        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )
        self.seq_rep = SequenceRepresentation(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input,
            use_static_ticker,
            auto_feature_num_input_linear,
        )
        self.num_heads = num_heads

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Optional ticker embedding
        if use_static_ticker:
            self.ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
        else:
            self.ticker_embedding = None

        # Positional encoding or embedding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)  # e.g., regression output

    def forward_candidate_arch(self, target_x, target_tickers, pos_encoding_batch=None, **kwargs):
        """
        target_x: [B, T, input_dim] - input features
        target_tickers: [B] - ticker ID per sample
        pos_encoding_batch: optional custom positional encodings
        """
        x = self.input_projection(target_x)  # [B, T, H]

        if self.use_static_ticker:
            ticker_embed = self.ticker_embedding(target_tickers)  # [B, H]
            ticker_embed = ticker_embed.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, H]
            x = x + ticker_embed

        if pos_encoding_batch is not None:
            x = x + pos_encoding_batch
        else:
            x = self.pos_encoding(x)

        encoded = self.transformer(x)  # [B, T, H]
        output = self.output_layer(encoded)  # [B, T, 1]
        return output.squeeze(-1)  # [B, T]

    def variable_importance(self, target_x, target_tickers, **kwargs):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)

