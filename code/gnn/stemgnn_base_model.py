"""StemGNN base model for multivariate time-series modelling.

This is a cleaned and hardware-aware version of the baseline StemGNN model
adapted from Cao et al. (2020). It keeps the original public interface used by
`stemgnn_contagion.py` while fixing syntax issues, numerical edge cases, and
obvious performance problems in the uploaded file.

Input to Model.forward:
    x: torch.Tensor with shape [batch, time_step, node_count]

Output from Model.forward:
    forecast: torch.Tensor
        If horizon == 1: [batch, 1, node_count]
        Else:            [batch, horizon, node_count]
    attention: torch.Tensor with shape [node_count, node_count]
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GLU(nn.Module):
    """Gated Linear Unit used inside the spectral sequence cell."""

    def __init__(self, input_channel: int, output_channel: int) -> None:
        super().__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_left(x) * torch.sigmoid(self.linear_right(x))


class StockBlockLayer(nn.Module):
    """One StemGNN spectral-temporal block.

    The block applies graph spectral propagation followed by frequency-domain
    temporal processing and returns both a forecast and an optional backcast.
    """

    def __init__(self, time_step: int, unit: int, multi_layer: int, stack_cnt: int = 0) -> None:
        super().__init__()
        self.time_step = int(time_step)
        self.unit = int(unit)
        self.stack_cnt = int(stack_cnt)
        self.multi = int(multi_layer)

        self.weight = nn.Parameter(
            torch.empty(1, 4, 1, self.time_step * self.multi, self.multi * self.time_step)
        )
        nn.init.xavier_normal_(self.weight)

        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)

        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)

        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            in_features = self.time_step * 4 if i == 0 else self.time_step * self.output_channel
            out_features = self.time_step * self.output_channel
            self.GLUs.append(GLU(in_features, out_features))
            self.GLUs.append(GLU(in_features, out_features))

    def spe_seq_cell(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Frequency-domain temporal cell.

        FFT is forced to float32 for stability and compatibility with AMP. Some
        CUDA FFT paths do not support half precision for arbitrary lengths.
        """
        batch_size, k, input_channel, node_cnt, time_step = input_tensor.size()
        original_dtype = input_tensor.dtype

        x = input_tensor.reshape(batch_size, -1, node_cnt, time_step).float()
        ffted = torch.view_as_real(torch.fft.fft(x, dim=1))

        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        imag = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        for i in range(3):
            real = self.GLUs[i * 2](real)
            imag = self.GLUs[i * 2 + 1](imag)

        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        imag = imag.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()

        complex_tensor = torch.view_as_complex(torch.stack((real, imag), dim=-1).contiguous())
        iffted = torch.fft.irfft(complex_tensor, n=complex_tensor.shape[1], dim=1)
        return iffted.to(dtype=original_dtype)

    def forward(self, x: torch.Tensor, mul_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor | None]:
        # x: [B, 1, N, T]
        # mul_l: [4, N, N]
        mul_l = mul_l.unsqueeze(1)       # [4, 1, N, N]
        x_expanded = x.unsqueeze(1)      # [B, 1, 1, N, T]

        gfted = torch.matmul(mul_l, x_expanded)       # [B, 4, 1, N, T]
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)

        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)

        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x_expanded).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None

        return forecast, backcast_source


class Model(nn.Module):
    """StemGNN base model.

    Args:
        units: number of nodes / time series.
        stack_cnt: number of stacked StemGNN blocks.
        time_step: input lookback length.
        multi_layer: spectral expansion multiplier.
        horizon: prediction horizon.
        dropout_rate: dropout inside learned graph attention.
        leaky_rate: negative slope for LeakyReLU.
        device: optional initial placement device.
    """

    def __init__(
        self,
        units: int,
        stack_cnt: int,
        time_step: int,
        multi_layer: int,
        horizon: int = 1,
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.unit = int(units)
        self.stack_cnt = int(stack_cnt)
        self.alpha = float(leaky_rate)
        self.time_step = int(time_step)
        self.horizon = int(horizon)
        self.multi_layer = int(multi_layer)

        self.weight_key = nn.Parameter(torch.empty(self.unit, 1))
        self.weight_query = nn.Parameter(torch.empty(self.unit, 1))
        nn.init.xavier_uniform_(self.weight_key, gain=1.414)
        nn.init.xavier_uniform_(self.weight_query, gain=1.414)

        self.GRU = nn.GRU(self.time_step, self.unit)
        self.stock_block = nn.ModuleList(
            StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i)
            for i in range(self.stack_cnt)
        )

        self.fc = nn.Sequential(
            nn.Linear(self.time_step, self.time_step),
            nn.LeakyReLU(self.alpha),
            nn.Linear(self.time_step, self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=float(dropout_rate))

        if device is not None:
            self.to(device)

    @staticmethod
    def get_laplacian(graph: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        if normalize:
            degree = torch.sum(graph, dim=-1).clamp_min(1e-7)
            d_inv_sqrt = torch.diag(torch.rsqrt(degree))
            eye = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype)
            return eye - d_inv_sqrt @ graph @ d_inv_sqrt
        degree = torch.sum(graph, dim=-1)
        return torch.diag(degree) - graph

    @staticmethod
    def cheb_polynomial(laplacian: torch.Tensor) -> torch.Tensor:
        """Return Chebyshev polynomials T0..T3 with T0 = I."""
        n = laplacian.size(0)
        t0 = torch.eye(n, device=laplacian.device, dtype=laplacian.dtype).unsqueeze(0)
        t1 = laplacian.unsqueeze(0)
        t2 = 2 * torch.matmul(laplacian.unsqueeze(0), t1) - t0
        t3 = 2 * torch.matmul(laplacian.unsqueeze(0), t2) - t1
        return torch.cat([t0, t1, t2, t3], dim=0)

    def latent_correlation_layer(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, N]
        gru_input = x.permute(2, 0, 1).contiguous()  # [N, B, T]
        gru_out, _ = self.GRU(gru_input)             # [N, B, N]
        node_features = gru_out.permute(1, 0, 2).contiguous()  # [B, N, N]

        attention = self.self_graph_attention(node_features)
        attention = attention.mean(dim=0)            # [N, N]
        attention = 0.5 * (attention + attention.transpose(0, 1))

        degree = attention.sum(dim=1).clamp_min(1e-7)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(torch.rsqrt(degree))
        laplacian = diagonal_degree_hat @ (degree_l - attention) @ diagonal_degree_hat
        mul_l = self.cheb_polynomial(laplacian)
        return mul_l, attention

    def self_graph_attention(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # input_tensor: [B, N, feature_dim]
        key = torch.matmul(input_tensor, self.weight_key)       # [B, N, 1]
        query = torch.matmul(input_tensor, self.weight_query)   # [B, N, 1]
        data = key + query.transpose(1, 2)                      # [B, N, N]
        attention = F.softmax(self.leakyrelu(data), dim=2)
        return self.dropout(attention)

    @staticmethod
    def graph_fft(input_tensor: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        return torch.matmul(eigenvectors, input_tensor)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"StemGNN expected input [batch, time_step, nodes], got {tuple(x.shape)}")
        if x.size(1) != self.time_step:
            raise ValueError(f"Expected time_step={self.time_step}, got {x.size(1)}")
        if x.size(2) != self.unit:
            raise ValueError(f"Expected nodes={self.unit}, got {x.size(2)}")

        mul_l, attention = self.latent_correlation_layer(x)
        block_input = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()  # [B, 1, N, T]

        forecasts = []
        for block in self.stock_block:
            forecast, backcast = block(block_input, mul_l)
            forecasts.append(forecast)
            if backcast is not None:
                block_input = backcast

        forecast_sum = torch.stack(forecasts, dim=0).sum(dim=0)
        forecast = self.fc(forecast_sum)  # [B, N, horizon]

        if forecast.size(-1) == 1:
            return forecast.unsqueeze(1).squeeze(-1), attention  # [B, 1, N]
        return forecast.permute(0, 2, 1).contiguous(), attention  # [B, H, N]
