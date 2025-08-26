import torch
import torch.nn as nn
import torch.nn.functional as F


class MDNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_mixtures, dropout=0.0):
        super(MDNLayer, self).__init__()
        self.out_dim = out_dim
        self.num_mixtures = num_mixtures

        self.dropout = nn.Dropout(dropout)

        self.fc_mu = nn.Linear(in_dim, num_mixtures * out_dim)
        self.fc_sigma = nn.Linear(in_dim, num_mixtures * out_dim)
        self.fc_alpha = nn.Linear(in_dim, num_mixtures)

    def forward(self, x):
        x = self.dropout(x)
        mu = self.fc_mu(x)
        sigma = torch.exp(self.fc_sigma(x)).clamp(min=1e-6)  # ensures strictly positive
        alpha = F.softmax(self.fc_alpha(x), dim=1)
        return mu, sigma, alpha


class LSTM_MDN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, num_mixtures, output_dim, dropout=0.3
    ):
        super(LSTM_MDN, self).__init__()
        self.num_mixtures = num_mixtures
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mdn = MDNLayer(hidden_dim, output_dim, num_mixtures)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # last timestep

        h = self.fc(last_output)
        h = self.batch_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        mu, sigma, alpha = self.mdn(h)
        return mu, sigma, alpha


class GRU_MDN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, num_mixtures, output_dim, dropout=0.3
    ):
        super(GRU_MDN, self).__init__()
        self.num_mixtures = num_mixtures
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mdn = MDNLayer(hidden_dim, output_dim, num_mixtures)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]  # Use last timestep

        h = self.fc(last_output)
        h = self.batch_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        mu, sigma, alpha = self.mdn(h)
        return mu, sigma, alpha


class RNN_MDN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, num_layers, num_mixtures, output_dim, dropout=0.3
    ):
        super(RNN_MDN, self).__init__()
        self.num_mixtures = num_mixtures
        self.hidden_dim = hidden_dim

        self.rnn = nn.RNN(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.mdn = MDNLayer(hidden_dim, output_dim, num_mixtures)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_output = rnn_out[:, -1, :]  # Last time step

        h = self.fc(last_output)
        h = self.batch_norm(h)
        h = F.relu(h)
        h = self.dropout(h)

        mu, sigma, alpha = self.mdn(h)
        return mu, sigma, alpha


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TCN_MDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_mixtures, output_dim, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(
                TCNBlock(
                    in_channels=input_dim if i == 0 else hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=2**i,
                )
            )

        self.num_mixtures = num_mixtures
        self.tcn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.mdn = MDNLayer(hidden_dim, output_dim, num_mixtures)

    def forward(self, x):  # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.tcn(x)
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
        return self.mdn(x)


class Transformer_MDN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_heads,
        num_layers,
        num_mixtures,
        output_dim,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.num_mixtures = num_mixtures
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dropout = nn.Dropout(dropout)
        self.mdn = MDNLayer(hidden_dim, output_dim, num_mixtures)

    def forward(self, x):  # x: (B, T, F)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_dropout(x)

        mu, sigma, alpha = self.mdn(x)
        return mu, sigma, alpha
