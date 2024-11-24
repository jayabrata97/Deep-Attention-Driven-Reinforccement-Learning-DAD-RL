# Encoder module contains encoders for Ego, SV both self-attention and cross attention

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys

device_ids = [0, 1]
device_1 = f"cuda:{device_ids[0]}"
device_2 = f"cuda:{device_ids[1]}"
torch.cuda.empty_cache()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):
        """
        Args:
        d_model:      dimension of embeddings
        dropout:      randomly zeroes-out some of the input
        max_length:   max sequence length
        """
        super().__init__()
        pe = torch.zeros(max_length, d_model)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)
        pe = pe.unsqueeze(0)
        # buffers are saved in state_dict but not trained by the optimizer
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x:      embeddings (batch_size, seq_length, d_model)

        Returns:
                embeddings + positional encodings (batch_size, seq_length, d_model)
        """
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)  # fixed for batch first
        return x


class State_Encoder_v3(nn.Module):

    def __init__(
        self,
        hidden_size,
        lstm_input_size,
        lstm_num_layers,
        device="cuda:0",
        pos_enc: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.device = torch.device(device)

        self.LSTM = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        ).to(device=self.device)

        if pos_enc:
            self.postional_encoding = PositionalEncoding(d_model=hidden_size).to(
                device=self.device
            )
        else:
            self.postional_encoding = None

        self.MA_encoder = nn.MultiheadAttention(
            embed_dim=56, num_heads=1, batch_first=True, add_zero_attn=True
        ).to(device=self.device)
        ## the add_zero_attn=True option attaches a dummy zero input at the end of key and value seq (useful when no sv vehicles present)
        self.linear = nn.Sequential(nn.Linear(76, 128), nn.ReLU()).to(
            device=self.device
        )
        self.layer_norm = nn.LayerNorm(56, elementwise_affine=False).to(
            device=self.device
        )

    def forward(self, sv_states, ev_states) -> torch.Tensor:
        """
        Ego to surrounding vehicle encoder

        Args:
            sv_states (torch.Tensor): Surrounding vehicles states; input shape: (N: number of surrounding vehicles, seq_length=5, feature_dim=6)
            ev_states (torch.Tensor): Ego vehicle states; input shape: (1, seq_length=5, feature_dim=20)

        Returns:
            Self attention encodings (torch.Tensor): Output shape:()
        """

        max_vehicles = 6
        batch_dim = ev_states.shape[0]
        ev_feature_dim = 20
        sv_feature_dim = 6

        ev_subset = ev_states[:, :, :sv_feature_dim]

        ### ego encoding
        _, (ev_encoding, _) = self.LSTM(ev_subset)
        ev_encoding = ev_encoding.transpose(0, 1)

        ### sv encoding
        sv_states_flatten = torch.flatten(sv_states, end_dim=1)
        _, (sv_encoding, _) = self.LSTM(sv_states_flatten)
        sv_encoding = torch.unflatten(sv_encoding, 1, (max_vehicles, -1)).squeeze(0)
        sv_encoding = sv_encoding.transpose(0, 1)

        ############################## Attention layers ###############################
        if self.postional_encoding is not None:
            sv_encoding = self.postional_encoding(sv_encoding)

        query = ev_encoding
        keys = sv_encoding
        values = sv_encoding

        key_mask = torch.zeros(size=(batch_dim, max_vehicles), device=self.device)
        key_mask[:,] = sv_states[:, :, -1, 0]  # first feature is padding (in version 3)
        key_mask = (
            key_mask.bool()
        )  # a True value indicates that the corresponding key value will be ignored for the purpose of attention.
        attn_output, attn_output_weights = self.MA_encoder(
            query=query, key=keys, value=values, key_padding_mask=key_mask
        )
        attn_output = attn_output.squeeze(1)
        attn_output_weights = attn_output_weights.squeeze(1)
        ego_interactive_enc = self.layer_norm(
            torch.add(ev_encoding.squeeze(1), attn_output)
        )

        state_encoding = self.linear(
            torch.concat((ev_states[:, -1], ego_interactive_enc), dim=1)
        )  # concat last timestep of ev_states and the ego_int_enc

        return state_encoding, attn_output_weights


class ContextCNN_v2(nn.Module):
    def __init__(self, observation_space, features_dim=128, device="cuda:0"):
        super().__init__()
        self.features_dim = features_dim

        # n_input_channels = observation_space['context'].shape[0]
        n_input_channels = 2

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 4, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        ).to(device=device)
        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None], device=device).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU()).to(
            device=device
        )

    def forward(self, input):
        x = self.cnn(input)
        x = self.linear(x)
        return x
