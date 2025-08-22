from mom_trans_torch.models.dmn import DeepMomentumNetwork, SequenceRepresentation, SequenceRepresentationSimple


class LstmBaseline(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,  # TODO maybe diable this for now..
        # output_signal_weights=False,
        **kwargs,
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
            # input_dim, # this was incorrect in the original code
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input,
            use_static_ticker,
            auto_feature_num_input_linear,
        )

    def forward_candidate_arch(
        self, target_x, target_tickers, pos_encoding_batch=None, **kwargs
    ):
        representation, _ = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch
        )
        return representation

    def variable_importance(
        self,
        target_x,
        target_tickers,
        # vol_scaling_amount,
        # vol_scaling_amount_prev,
        # trans_cost_bp,
        # target_y=None,
        **kwargs,
    ):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)


class LstmSimple(DeepMomentumNetwork):
    def __init__(
        self,
        input_dim: int,
        num_tickers: int,
        hidden_dim: int,
        dropout: float,
        use_static_ticker: bool = True,
        auto_feature_num_input_linear=0,  # TODO maybe diable this for now..
        # output_signal_weights=False,
        final_mlp=False,
        **kwargs,
    ):
        # self.output_signal_weights = output_signal_weights
        fuse_encoder_input = False

        super().__init__(
            input_dim=input_dim,
            num_tickers=num_tickers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_static_ticker=use_static_ticker,
            auto_feature_num_input_linear=auto_feature_num_input_linear,
            **kwargs,
        )

        self.seq_rep = SequenceRepresentationSimple(
            self.input_dim,
            hidden_dim,
            dropout,
            num_tickers,
            fuse_encoder_input,
            use_static_ticker,
            auto_feature_num_input_linear,
            use_prescaler=self.date_time_embedding,
            final_mlp=final_mlp,
        )

    def forward_candidate_arch(
        self, target_x, target_tickers, pos_encoding_batch=None, **kwargs
    ):
        representation = self.seq_rep(
            target_x, target_tickers, pos_encoding_batch=pos_encoding_batch
        )
        return representation

    def variable_importance(
        self,
        target_x,
        target_tickers,
        # vol_scaling_amount,
        # vol_scaling_amount_prev,
        # trans_cost_bp,
        # target_y=None,
        **kwargs,
    ):
        importance = self.seq_rep(target_x, target_tickers, variable_importance=True)
        return importance.squeeze(-2).swapaxes(0, 1)