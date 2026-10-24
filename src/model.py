import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HandPoseCNNLSTM(nn.Module):
    """
    CNN‑LSTM model for hand‑pose sequence classification.

    The model expects a batch of sequences where each frame is represented by a
    flat vector of hand‑landmark coordinates (e.g. 21 landmarks × 3 axes = 63).

    Architecture
    ------------
    1. **Temporal Convolution** (1‑D CNN) extracts short‑range temporal patterns
       from the raw landmark sequence.
    2. **Bidirectional LSTM** captures long‑range dependencies.
    3. **Fully‑connected classifier** maps the final hidden state to class logits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of a single frame (number of landmarks × coordinate axes).
    cnn_out_channels : int, default 128
        Number of output channels of the temporal convolution.
    cnn_kernel_size : int, default 3
        Kernel size of the temporal convolution.
    lstm_hidden_size : int, default 256
        Hidden size of the LSTM.
    lstm_num_layers : int, default 2
        Number of stacked LSTM layers.
    num_classes : int
        Number of output gesture classes.
    dropout : float, default 0.5
        Dropout probability applied after the CNN and between LSTM layers.
    bidirectional : bool, default True
        Whether to use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        cnn_out_channels: int = 128,
        cnn_kernel_size: int = 3,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # 1‑D Convolution over the temporal dimension (seq_len)
        # Input shape for Conv1d: (batch, in_channels, seq_len)
        # ------------------------------------------------------------------
        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_out_channels,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2,
            bias=False,
        )
        self.cnn_bn = nn.BatchNorm1d(cnn_out_channels)
        self.cnn_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # LSTM that processes the sequence of CNN features
        # Input shape for LSTM: (batch, seq_len, cnn_out_channels)
        # ------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)

        # ------------------------------------------------------------------
        # Final classification layer
        # ------------------------------------------------------------------
        self.classifier = nn.Linear(lstm_output_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights with Kaiming normal for conv layers and
        Xavier uniform for linear layers."""
        nn.init.kaiming_normal_(self.cnn.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.cnn_bn.weight, 1)
        nn.init.constant_(self.cnn_bn.bias, 0)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.constant_(param.data, 0)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, input_dim).

        Returns
        -------
        torch.Tensor
            Logits tensor of shape (batch, num_classes).
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected input tensor to have 3 dimensions (batch, seq_len, input_dim), got {x.shape}"
            )

        batch_size, seq_len, input_dim = x.shape

        # ------------------------------------------------------------------
        # Temporal convolution
        # ------------------------------------------------------------------
        # Conv1d expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)                     # (batch, input_dim, seq_len)
        x_cnn = self.cnn(x_cnn)                       # (batch, cnn_out_channels, seq_len)
        x_cnn = self.cnn_bn(x_cnn)
        x_cnn = F.relu(x_cnn)
        x_cnn = self.cnn_dropout(x_cnn)
        x_cnn = x_cnn.permute(0, 2, 1)                # (batch, seq_len, cnn_out_channels)

        # ------------------------------------------------------------------
        # LSTM
        # ------------------------------------------------------------------
        lstm_out, (h_n, _) = self.lstm(x_cnn)         # lstm_out: (batch, seq_len, hidden*dir)
        # Use the last time‑step output from the forward direction and,
        # if bidirectional, the first time‑step output from the backward direction.
        if self.lstm.bidirectional:
            # Concatenate forward last hidden and backward first hidden
            forward_last = h_n[-2]                    # (batch, hidden)
            backward_first = h_n[-1]                  # (batch, hidden)
            lstm_feat = torch.cat([forward_last, backward_first], dim=1)
        else:
            lstm_feat = h_n[-1]                        # (batch, hidden)

        # ------------------------------------------------------------------
        # Classification
        # ------------------------------------------------------------------
        logits = self.classifier(lstm_feat)           # (batch, num_classes)
        return logits


def get_hand_pose_cnn_lstm(
    input_dim: int,
    num_classes: int,
    cnn_out_channels: int = 128,
    cnn_kernel_size: int = 3,
    lstm_hidden_size: int = 256,
    lstm_num_layers: int = 2,
    dropout: float = 0.5,
    bidirectional: bool = True,
) -> HandPoseCNNLSTM:
    """
    Convenience factory that returns a ready‑to‑use HandPoseCNNLSTM instance
    with sensible default hyper‑parameters.

    Parameters
    ----------
    input_dim : int
        Dimensionality of a single frame (e.g. 63 for 21 3‑D landmarks).
    num_classes : int
        Number of gesture classes.
    cnn_out_channels, cnn_kernel_size, lstm_hidden_size, lstm_num_layers, dropout, bidirectional
        See :class:`HandPoseCNNLSTM`.

    Returns
    -------
    HandPoseCNNLSTM
        Instantiated model.
    """
    return HandPoseCNNLSTM(
        input_dim=input_dim,
        num_classes=num_classes,
        cnn_out_channels=cnn_out_channels,
        cnn_kernel_size=cnn_kernel_size,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        dropout=dropout,
        bidirectional=bidirectional,
    )


# ----------------------------------------------------------------------
# Simple sanity‑check when the module is executed directly.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Configuration for a dummy run
    BATCH_SIZE = 4
    SEQ_LEN = 30
    INPUT_DIM = 63          # 21 landmarks × 3 coordinates
    NUM_CLASSES = 20

    # Create random input tensor
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # Instantiate model
    model = get_hand_pose_cnn_lstm(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
    )

    # Forward pass
    logits = model(dummy_input)

    assert logits.shape == (BATCH_SIZE, NUM_CLASSES), (
        f"Logits shape mismatch: expected {(BATCH_SIZE, NUM_CLASSES)}, got {logits.shape}"
    )
    print("HandPoseCNNLSTM sanity check passed.")