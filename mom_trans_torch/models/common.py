import torch
from enum import Enum
from typing import Dict
from torch import nn
import numpy as np

import torch.nn.functional as F

MAX_NORM = 3.0


class LossFunction(Enum):
    SHARPE = 0
    JOINT_GAUSS = 1
    JOINT_QRE = 2


def remap_values(x, remapping):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)

def positional_encoding(max_len, d_model):
    """
    Positional encoding function based on sine and cosine functions.

    :param max_len: Maximum sequence length.
    :param d_model: Embedding dimension.
    :return: Positional encoding matrix (max_len, d_model).
    """
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# pytorch transformer positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


def rolling_average_conv(tensor, window_size, device, feature_dim=1, exponentially_weighted=False):
    """
    Calculates the rolling average of a 1D tensor using a 1D convolution.

    :param data: A PyTorch tensor of shape (batch_size, sequence_length).
    :param window_size: The size of the rolling window.

    :return: A PyTorch tensor of the same shape as the input data, containing
                    the rolling average for each element.
    """
    if exponentially_weighted:
        # Calculate alpha from span
        alpha = 2 / (window_size + 1)

        sequence_length = tensor.shape[1]

        # Initialize the EMA tensor with the same shape as the input tensor and move to GPU
        ema = torch.zeros_like(tensor).cuda()

        # Compute the EMA for each element in the sequence
        ema[:, 0, :] = tensor[:, 0, :]  # The EMA for the first element is the element itself

        for t in range(1, sequence_length):
            ema[:, t, :] = alpha * tensor[:, t, :] + (1 - alpha) * ema[:, t - 1, :]
        
        return ema
        

    else:
        kernel = torch.ones(1, 1, window_size).to(device) / window_size

        # Apply the convolution along the sequence length dimension (dim=1)
        # First, we need to permute the dimensions to match the expected input shape for conv1d
        # from (batch_size, sequence_length, feature_dim) to (batch_size, feature_dim, sequence_length)
        tensor = tensor.permute(0, 2, 1)

        # Perform the convolution
        tensor = F.pad(tensor, (window_size-1, 0))
        rolling_avg = F.conv1d(tensor, kernel, groups=feature_dim)

        # Permute back to the original shape (batch_size, sequence_length, feature_dim)
        return rolling_avg.permute(0, 2, 1)


class DropoutNoScaling(nn.Module):
    def __init__(self, p=0.5):
        super(DropoutNoScaling, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mask = x.new_empty(x.size()).bernoulli_(1 - self.p)
            return mask * x #/ (1 - self.p)
        else:
            return x    

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: int, hidden_size: int = None, dropout: float = None):
        super().__init__()

        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class AddNorm(nn.Module):
    def __init__(
        self, input_size: int, trainable_add: bool = True
    ):
        super().__init__()

        self.input_size = input_size
        self.trainable_add = trainable_add

        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0

        output = self.norm(x + skip)
        return output


class GateAddNorm(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = None,

        trainable_add: bool = False,
        dropout: float = None,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.dropout = dropout

        self.glu = GatedLinearUnit(
            self.input_size, hidden_size=self.hidden_size, dropout=self.dropout
        )
        self.add_norm = AddNorm(
            self.hidden_size, trainable_add=trainable_add
        )

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.1,
        context_size: int = None,
        residual: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual

        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size

        if self.output_size != residual_size:
            self.resample_norm = nn.Linear(residual_size, self.output_size)

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()

        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size, bias=False)

        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.gate_norm = GateAddNorm(
            input_size=self.hidden_size,
            hidden_size=self.output_size,
            dropout=self.dropout,
            trainable_add=False,
        )

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x

        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)

        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(
        self,
        input_sizes: Dict[str, int],
        hidden_size: int,
        # input_embedding_flags: Dict[str, bool] = {},
        dropout: float = 0.1,
        context_size: int = None,
        single_variable_grns: Dict[str, GatedResidualNetwork] = {},
        prescalers: Dict[str, nn.Linear] = {},
    ):
        """
        Calcualte weights for ``num_inputs`` variables  which are each of size ``input_size``
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        # self.input_embedding_flags = input_embedding_flags
        self.dropout = dropout
        self.context_size = context_size

        if self.num_inputs > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    self.context_size,
                    residual=False,
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    self.input_size_total,
                    min(self.hidden_size, self.num_inputs),
                    self.num_inputs,
                    self.dropout,
                    residual=False,
                )

        self.single_variable_grns = nn.ModuleDict()
        self.prescalers = nn.ModuleDict()
        for name, input_size in self.input_sizes.items():
            if name in single_variable_grns:
                self.single_variable_grns[name] = single_variable_grns[name]
            # elif self.input_embedding_flags.get(name, False):
            #     self.single_variable_grns[name] = ResampleNorm(input_size, self.hidden_size)
            else:
                self.single_variable_grns[name] = GatedResidualNetwork(
                    input_size,
                    min(input_size, self.hidden_size),
                    output_size=self.hidden_size,
                    dropout=self.dropout,
                )
            if name in prescalers:  # reals need to be first scaled up
                self.prescalers[name] = prescalers[name]
            # elif not self.input_embedding_flags.get(name, False):
            else:
                self.prescalers[name] = nn.Linear(1, input_size)

        self.softmax = nn.Softmax(dim=-1)

    @property
    def input_size_total(self):
        return sum(
            size
            for _, size in self.input_sizes.items()
        )

    @property
    def num_inputs(self):
        return len(self.input_sizes)

    def forward(self, x: Dict[str, torch.Tensor], context: torch.Tensor = None):
        if self.num_inputs > 1:
            # transform single variables
            var_outputs = []
            weight_inputs = []
            for name in self.input_sizes.keys():
                # select embedding belonging to a single input
                variable_embedding = x[name]
                if name in self.prescalers:
                    variable_embedding = self.prescalers[name](variable_embedding)
                weight_inputs.append(variable_embedding)
                var_outputs.append(self.single_variable_grns[name](variable_embedding))
            var_outputs = torch.stack(var_outputs, dim=-1)

            # calculate variable weights
            flat_embedding = torch.cat(weight_inputs, dim=-1)
            sparse_weights = self.flattened_grn(flat_embedding, context)
            sparse_weights = self.softmax(sparse_weights).unsqueeze(-2)

            outputs = var_outputs * sparse_weights
            outputs = outputs.sum(dim=-1)
        else:  # for one input, do not perform variable selection but just encoding
            name = next(iter(self.single_variable_grns.keys()))
            variable_embedding = x[name]
            if name in self.prescalers:
                variable_embedding = self.prescalers[name](variable_embedding)
            outputs = self.single_variable_grns[name](
                variable_embedding
            )  # fast forward if only one variable
            if outputs.ndim == 3:  # -> batch size, time, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), outputs.size(1), 1, 1, device=outputs.device
                )  #
            else:  # ndim == 2 -> batch size, hidden size, n_variables
                sparse_weights = torch.ones(
                    outputs.size(0), 1, 1, device=outputs.device
                )
        return outputs, sparse_weights


def max_norm(model, max_val=MAX_NORM, eps=1e-8):
    for name, param in model.named_parameters():
        if "bias" not in name:
            # TOD revisit this
            norm = param.norm(2, dim=1, keepdim=True)
            # norm = param.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, 0, max_val)
            param = param * (desired / (eps + norm))


def look_ahead_mask(tgt_len: int, src_len: int) -> torch.BoolTensor:
    # return torch.tril(torch.full((tgt_len, src_len), 1)).bool()
    return torch.triu(torch.full((tgt_len, src_len), -float("inf")), diagonal=1)
    # mask = torch.triu(torch.ones(tgt_len, src_len), diagonal=1)
    # mask[mask!=0.0] = -float("inf")
    # return mask


def for_masked_fill(tgt_len: int, encoder_length=0):
    if encoder_length:
        left = torch.zeros((tgt_len, encoder_length))  # .bool()
        top_right = torch.ones((encoder_length, tgt_len - encoder_length))  # .bool()
        bottom_right = torch.triu(
            torch.full((tgt_len - encoder_length, tgt_len - encoder_length), 1),
            diagonal=1,
        )
        right = torch.cat([top_right, bottom_right])
        return torch.cat([left, right], axis=1).bool()
    else:
        # return torch.triu(torch.full((tgt_len, tgt_len), 1), diagonal=1).bool()
        return torch.BoolTensor(np.triu(np.full((tgt_len, tgt_len), 1), k=1) == 1)

    # From docs: [src/tgt/memory]_mask ensures that position i is allowed to attend the unmasked
    # positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
    # while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
    # are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
    # is provided, it will be added to the attention weight.


def for_masked_fill_forecasting(tgt_len, device):
    # TODO is this right???
    attend_step = torch.arange(tgt_len, device=device)
    # indices for which is predicted
    predict_step = torch.arange(0, tgt_len, device=device)[:, None]
    # do not attend to steps to self or after prediction
    decoder_mask = (attend_step >= predict_step).unsqueeze(
        0
    )  # .expand(encoder_lengths.size(0), -1, -1)


def numpy_normalised_quantile_loss(y, y_pred, quantiles):
    """Computes normalised quantile loss for numpy arrays.
    Args:
      y: Targets
      y_pred: Predictions
      quantile: Quantile to use for loss calculations (between 0 & 1)
    Returns:
      Float for normalised quantile loss.
    """
    quantiles = np.array(quantiles)
    error = np.expand_dims(y, 1) - y_pred
    weighted_errors = np.maximum(quantiles * error, (1.0 - quantiles) * -error)

    quantile_loss = weighted_errors.mean()
    normaliser = np.abs(y).mean()

    return 2 * quantile_loss / normaliser


def time_to_vec(x, T, custom_offset=None):
    if custom_offset:
        offset = custom_offset
    else:
        if T % 2 == 1:
            offset = 0
        else:
            offset = np.pi / (2 * T)
    return np.sin(2 * np.pi / T * x + offset)


class SeriesDecomposition(nn.Module):
    """
    Decompose time series into trend and seasonal components using moving average.
    """
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAverage(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - moving_mean
        return residual, moving_mean


class MovingAverage(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAverage, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [B, L, C] -> [B, C, L]
        front = x[:, 0:1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x_padded = torch.cat([front, x, end], dim=1)
        x_padded = x_padded.permute(0, 2, 1)  # [B, C, L]
        ma = self.avg_pool(x_padded)
        ma = ma.permute(0, 2, 1)  # [B, L, C]
        return ma


def series_decomp(kernel_size):
    return SeriesDecomposition(kernel_size)
