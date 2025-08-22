import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from mom_trans_torch.models.dmn import DeepMomentumNetwork


class MambaBaseline(DeepMomentumNetwork):
    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            dropout: float,
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
        self.ssm_rank = kwargs['ssm_rank']  # Rank of SSM parameterization
        self.kernel_size = kwargs['kernel_size']  # Kernel size for causal convolution

        # Causal convolution for local mixing
        self.conv1d = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim * 2,  # Double for gate/value
            kernel_size=self.kernel_size,
            padding=0,  # Will manually pad left
        )

        # Selective SSM parameters (Mamba-style)
        self.A_log = nn.Parameter(torch.randn(hidden_dim, self.ssm_rank))
        self.B_proj = nn.Linear(hidden_dim, self.ssm_rank, bias=False)
        self.C_proj = nn.Linear(hidden_dim, self.ssm_rank, bias=False)
        self.D_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dt_proj = nn.Linear(hidden_dim, self.ssm_rank)

        # Optional components
        if use_static_ticker:
            self.static_ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
        if auto_feature_num_input_linear > 0:
            self.auto_feature_linear = nn.Linear(auto_feature_num_input_linear, hidden_dim)

        # State buffer
        self.register_buffer('state', None)
        self.dropout = nn.Dropout(dropout)

    def reset_state(self, batch_size=1):
        """Initialize or reset the SSM state"""
        self.state = torch.zeros(
            batch_size, self.hidden_dim, self.ssm_rank,
            device=self.A_log.device
        )

    def _ssm_step(self, x, dt):
        """Discretized state-space model step"""
        # Discretize A (input-dependent)
        A = -torch.exp(self.A_log.clamp(max=0))  # Ensure stability
        deltaA = torch.exp(dt.unsqueeze(1) * A.unsqueeze(0))  # (B, H, R)

        # Project inputs
        B = self.B_proj(x)  # (B, R)
        C = self.C_proj(x)  # (B, R)

        # State update
        self.state = (deltaA * self.state + B.unsqueeze(1) * x.unsqueeze(2)).detach()

        # Output
        return torch.einsum('bhr,br->bh', self.state, C) + self.D_proj(x)

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        # Input projection with causal padding
        x = target_x.transpose(1, 2)  # (B, D, S)
        x = F.pad(x, (self.kernel_size - 1, 0))  # Left padding only
        x = self.conv1d(x)  # (B, 2*H, S)

        # Split into gate and value
        x = rearrange(x, 'b (g h) s -> b s g h', g=2)
        x_val, x_gate = x.unbind(dim=2)

        # Initialize state if needed
        if self.state is None or self.state.size(0) != x_val.size(0):
            self.reset_state(x_val.size(0))

        # Process sequence
        outputs = []
        for t in range(x_val.size(1)):
            x_t = x_val[:, t]  # (B, H)
            dt = F.softplus(self.dt_proj(x_t))  # (B, R)
            outputs.append(self._ssm_step(x_t, dt))

        # Combine outputs
        x_out = torch.stack(outputs, dim=1)  # (B, S, H)
        x_out = x_out * torch.sigmoid(x_gate)  # Apply gate
        x_out = self.dropout(x_out)

        # Add optional components
        if self.use_static_ticker:
            x_out += self.static_ticker_embedding(target_tickers).unsqueeze(1)
        if hasattr(self, 'auto_feature_linear') and 'auto_features' in kwargs:
            x_out += self.auto_feature_linear(kwargs['auto_features']).unsqueeze(1)

        return x_out

    def variable_importance(self, target_x, target_tickers, **kwargs):
        """Compute feature importance scores"""
        # Get conv output weights
        conv_weights = self.conv1d.weight.abs().sum(dim=(0, 2))  # (2*H, D)

        # Split into gate and value importance
        gate_imp, val_imp = conv_weights.chunk(2, dim=0)
        importance = (gate_imp + val_imp).t()  # (D, H)

        # Normalize
        importance = F.softmax(importance, dim=0)
        return importance.unsqueeze(0)  # (1, D, H)


class Mamba2Baseline(DeepMomentumNetwork):
    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            dropout: float,
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
        self.ssm_rank = kwargs['ssm_rank']  # Rank of SSM parameterization
        self.kernel_size = kwargs['kernel_size']  # Kernel size for causal convolution
        self.use_static_ticker = use_static_ticker
        self.max_delta = kwargs['max_delta']  # Store clamping threshold

        # 1. Input projection + gating
        self.in_proj = nn.Linear(input_dim, hidden_dim * 2)

        # 2. Causal depthwise conv
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=self.kernel_size,
            padding=0,
            groups=hidden_dim,
        )

        # 3. SSM parameters (Mamba2-style)
        self.A_log = nn.Parameter(torch.randn(hidden_dim, self.ssm_rank))
        self.B_proj = nn.Linear(hidden_dim, self.ssm_rank, bias=False)
        self.C_proj = nn.Linear(hidden_dim, self.ssm_rank, bias=False)
        self.D = nn.Parameter(torch.randn(hidden_dim))
        self.delta_proj = nn.Linear(hidden_dim, self.ssm_rank)

        # 4. Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # State management
        self.register_buffer('state', None)
        self.dropout = nn.Dropout(dropout)

        # Optional components
        if use_static_ticker:
            self.static_ticker_embedding = nn.Embedding(num_tickers, hidden_dim)
        if auto_feature_num_input_linear > 0:
            self.auto_feature_linear = nn.Linear(auto_feature_num_input_linear, hidden_dim)

    def reset_state(self, batch_size=1):
        """Initialize or reset the SSM state"""
        with torch.no_grad():
            self.state = torch.zeros(
                batch_size, self.hidden_dim, self.ssm_rank,
                device=self.A_log.device
            )

    def _causal_ssm_scan(self, u, delta):
        """Strictly causal SSM scan with corrected dimensions"""
        B, S, H = u.shape
        R = self.ssm_rank

        # Discretize A (H, R) -> (1, H, R)
        A = -torch.exp(self.A_log.clamp(max=0))  # (H, R)

        # delta comes in as (B, S, R)
        # We need to expand it to (B, H, R) for element-wise multiplication
        delta_expanded = delta.unsqueeze(2)  # (B, S, 1, R)
        A_expanded = A.unsqueeze(0).unsqueeze(0)  # (1, 1, H, R)
        deltaA = torch.exp(delta_expanded * A_expanded)  # (B, S, H, R)

        if self.state is None or self.state.size(0) != B:
            self.reset_state(B)

        outputs = []
        for t in range(S):
            # Get current timestep parameters
            deltaA_t = deltaA[:, t]  # (B, H, R)
            u_t = u[:, t]  # (B, H)

            # State update
            self.state = (deltaA_t * self.state + self.B_proj(u_t).unsqueeze(1)).detach()  # (B, H, R)

            # Output
            y = torch.einsum('bhr,br->bh', self.state, self.C_proj(u_t))  # (B, H)
            outputs.append(y)

        return torch.stack(outputs, dim=1) + self.D * u  # (B, S, H)

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        # Input projection
        x = self.in_proj(target_x)  # (B, S, 2H)
        x, gate = x.chunk(2, dim=-1)  # (B, S, H)

        # Causal depthwise conv
        x = rearrange(x, 'b s h -> b h s')
        x = F.pad(x, (self.kernel_size - 1, 0))  # Left padding only
        x = self.conv1d(x)  # (B, H, S)
        x = rearrange(x, 'b h s -> b s h')  # (B, S, H)

        # Input-dependent discretization with stability control
        delta = F.softplus(self.delta_proj(x))  # (B, S, R)
        if self.max_delta is not None:
            delta = delta.clamp(max=self.max_delta)

        # Process sequence
        x_ssm = self._causal_ssm_scan(x, delta)  # (B, S, H)

        # Output projection
        x_out = self.out_proj(F.silu(gate) * x_ssm) + x
        x_out = self.dropout(x_out)

        # Optional components
        if self.use_static_ticker:
            x_out += self.static_ticker_embedding(target_tickers).unsqueeze(1)
        if hasattr(self, 'auto_feature_linear') and 'auto_features' in kwargs:
            x_out += self.auto_feature_linear(kwargs['auto_features']).unsqueeze(1)

        return x_out  # (B, S, H)

    def variable_importance(self, target_x, target_tickers, **kwargs):
        """Compute feature importance scores with correct dimensions"""
        # Conv importance (H,)
        conv_weights = self.conv1d.weight.abs().sum(dim=(0, 2))  # Sum over input_dim and kernel

        # SSM importance (H,)
        ssm_importance = torch.exp(self.A_log).mean(dim=1)  # Mean over ssm_rank

        # Combine and normalize (1, H, 1)
        importance = (conv_weights * ssm_importance).softmax(dim=0)
        return importance.unsqueeze(0).unsqueeze(-1)


class Mamba2Finance(DeepMomentumNetwork):
    def __init__(
            self,
            input_dim: int,
            num_tickers: int,
            hidden_dim: int,
            dropout: float,
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
        self.ssm_rank = kwargs['ssm_rank']  # Rank of SSM parameterization
        self.kernel_size = kwargs['kernel_size']  # Kernel size for causal convolution
        self.use_static_ticker = use_static_ticker
        self.max_delta = kwargs['max_delta']  # Store clamping threshold

        # 1. Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim * 2)

        # 2. Causal depthwise conv
        self.conv1d = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=self.kernel_size,
            groups=hidden_dim,
            padding=self.kernel_size - 1
        )

        # 3. Mamba2 SSM parameters (dynamic)
        self.A_log = nn.Linear(hidden_dim, hidden_dim * self.ssm_rank)
        self.B = nn.Linear(hidden_dim, hidden_dim * self.ssm_rank)
        self.C = nn.Linear(hidden_dim, hidden_dim * self.ssm_rank)
        self.D = nn.Parameter(torch.ones(hidden_dim))
        self.delta_proj = nn.Linear(hidden_dim, self.ssm_rank)

        # 4. Gated MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # 5. Output
        self.out_proj = nn.Linear(hidden_dim, 1)  # Predict returns
        self.dropout = nn.Dropout(dropout)

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        # Input projection
        x = self.in_proj(target_x)  # (B, S, 2H)
        x, gate = x.chunk(2, dim=-1)

        # Causal conv
        x = rearrange(x, 'b s h -> b h s')
        x = self.conv1d(x)[..., :-self.conv1d.padding[0]]  # Remove excess padding
        x = rearrange(x, 'b h s -> b s h')

        # Discretization
        delta = F.softplus(self.delta_proj(x))  # (B, S, R)

        # Mamba2 SSM
        x_ssm = self._mamba2_scan(x, delta)
        x = F.silu(gate) * x_ssm

        # MLP + output
        x = x + self.mlp(x)  # Residual
        return self.out_proj(self.dropout(x))

    def _mamba2_scan(self, u, delta):
        B, S, H = u.shape
        R = self.ssm_rank

        # Dynamic A (B, S, H, R)
        A = -torch.exp(self.A_log(u).view(B, S, H, R).clamp(max=0))
        deltaA = torch.exp(delta.unsqueeze(2)) * A

        # Dynamic B, C (B, S, H, R)
        B_t = self.B(u).view(B, S, H, R)
        C_t = self.C(u).view(B, S, H, R)

        # Parallel scan (simplified)
        state = torch.zeros(B, H, R, device=u.device)
        outputs = []
        for t in range(S):
            state = deltaA[:, t] * state + B_t[:, t]
            outputs.append(torch.einsum('bhr,bhr->bh', state, C_t[:, t]))
        return torch.stack(outputs, dim=1) + self.D * u

    def variable_importance(self, target_x, target_tickers, **kwargs):
        """Compute feature importance scores with correct dimensions"""
        # Conv importance (H,)
        conv_weights = self.conv1d.weight.abs().sum(dim=(0, 2))  # Sum over input_dim and kernel

        # SSM importance (H,)
        ssm_importance = torch.exp(self.A_log).mean(dim=1)  # Mean over ssm_rank

        # Combine and normalize (1, H, 1)
        importance = (conv_weights * ssm_importance).softmax(dim=0)
        return importance.unsqueeze(0).unsqueeze(-1)
