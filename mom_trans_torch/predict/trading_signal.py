from mom_trans_torch.training.train import TrainMomentumTransformer
from mom_trans_torch.models.dmn import DmnMode

FIELDS_REQUIRED_FOR_PREDICT = [
    "dropout",
    "lr",
    "max_gradient_norm",
    "hidden_dim",
    "num_heads",
    "temporal_att_placement",
]

OTHER_FIELDS_REQUIRED_FOR_PREDICT = [
    "num_tickers",
    "num_tickers_full_univ",
    "vs_factor_scaler",
    "trans_cost_scaler",
]

FIELDS_REQUIRED_FOR_PREDICT = (
    FIELDS_REQUIRED_FOR_PREDICT + OTHER_FIELDS_REQUIRED_FOR_PREDICT
)




class TradingSignalDmn:
    def __init__(
        self,
        architecture,
        test_data,
        num_features,
        num_tickers,
        model_save_path,
        parent_class=TrainMomentumTransformer,
        extra_settings={},
        **kwargs,
    ):
        self.parent = parent_class(
            None,  # no train data
            None,  # no valid data
            test_data,
            kwargs["seq_len"],
            num_features,
            model_save_path,
            **extra_settings,
        )
        self.optimise_loss_function = kwargs["optimise_loss_function"]

        self.model = self.parent._load_architecture(
            architecture, num_features, num_tickers, **kwargs
        )

    def predict(  # pylint: disable=arguments-differ
        self,
        run_name,
        batch_size,
    ):
        return self.parent.predict(
            model=self.model,
            model_save_path=self.parent.model_save_path(run_name),
            batch_size=batch_size,
            optimise_loss_function=self.optimise_loss_function,
            mode=DmnMode.LIVE,
        )
