from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


@dataclass
class CnnResult:
    model: str
    subject: str
    accuracy: float
    macro_f1: float
    confusion: np.ndarray
    history: list[dict[str, float]]


def _require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the CNN model. Install it with `pip install torch`.") from exc
    return torch, nn, DataLoader, TensorDataset


def build_torch_model(model_name: str, n_channels: int, n_classes: int = 4):
    """Build a compact EEG neural decoder."""
    torch, nn, _, _ = _require_torch()

    class EEGNetSmall(nn.Module):
        def __init__(self, n_channels: int, n_classes: int = 4):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, 31), padding=(0, 15), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.35),
                nn.Conv2d(16, 32, kernel_size=(1, 15), padding=(0, 7), bias=False),
                nn.BatchNorm2d(32),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(32, n_classes),
            )

        def forward(self, x):
            return self.net(x)

    class TemporalResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float = 0.25):
            super().__init__()
            padding = dilation * 3
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=7, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.act = nn.ELU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class RawSignalShortResNet(nn.Module):
        def __init__(self, n_channels: int, n_classes: int = 4):
            super().__init__()
            self.spatial = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(n_channels, 1), bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
            )
            self.temporal = nn.Sequential(
                nn.Conv1d(16, 32, kernel_size=15, padding=7, bias=False),
                nn.BatchNorm1d(32),
                nn.ELU(),
                TemporalResidualBlock(32, dilation=1),
                TemporalResidualBlock(32, dilation=2),
                TemporalResidualBlock(32, dilation=4),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(32, n_classes))

        def forward(self, x):
            x = self.spatial(x).squeeze(2)
            x = self.temporal(x)
            return self.head(x)

    class Square(nn.Module):
        def forward(self, x):
            return x * x

    class SafeLog(nn.Module):
        def forward(self, x):
            return torch.log(torch.clamp(x, min=1e-6))

    class ShallowConvNet(nn.Module):
        """Motor-imagery CNN shaped after the classic ShallowConvNet family.

        The temporal filter learns band-power-like features, the spatial filter
        mixes electrodes, and square/log activations mimic CSP-style log-variance
        features while remaining trainable end to end.
        """

        def __init__(self, n_channels: int, n_classes: int = 4):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12), bias=False),
                nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False),
                nn.BatchNorm2d(40),
                Square(),
                nn.AvgPool2d(kernel_size=(1, 35), stride=(1, 7)),
                SafeLog(),
                nn.Dropout(0.5),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier = nn.Linear(40, n_classes)

        def forward(self, x):
            return self.classifier(self.features(x))

    class TemporalConvBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float = 0.3):
            super().__init__()
            padding = dilation * 2
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=5, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
                nn.ELU(),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return x + self.net(x)

    class EEGTCNet(nn.Module):
        """Compact EEGNet front-end followed by temporal convolution blocks.

        This follows the EEG-TCNet idea: learn EEGNet-style spectral/spatial
        filters first, then model temporal dependencies with dilated TCN blocks.
        It is small enough for real-time inference and usually more data-efficient
        than transformer-style models on BCI IV 2a.
        """

        def __init__(self, n_channels: int, n_classes: int = 4):
            super().__init__()
            self.frontend = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=(1, 31), padding=(0, 15), bias=False),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, kernel_size=(n_channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.35),
                nn.Conv2d(16, 16, kernel_size=(1, 15), padding=(0, 7), groups=16, bias=False),
                nn.Conv2d(16, 24, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(24),
                nn.ELU(),
                nn.AvgPool2d(kernel_size=(1, 4)),
                nn.Dropout(0.35),
            )
            self.tcn = nn.Sequential(
                TemporalConvBlock(24, dilation=1),
                TemporalConvBlock(24, dilation=2),
                TemporalConvBlock(24, dilation=4),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            self.classifier = nn.Linear(24, n_classes)

        def forward(self, x):
            x = self.frontend(x).squeeze(2)
            x = self.tcn(x)
            return self.classifier(x)

    if model_name == "cnn":
        return EEGNetSmall(n_channels=n_channels, n_classes=n_classes)
    if model_name == "raw_resnet":
        return RawSignalShortResNet(n_channels=n_channels, n_classes=n_classes)
    if model_name == "shallow_convnet":
        return ShallowConvNet(n_channels=n_channels, n_classes=n_classes)
    if model_name == "eeg_tcnet":
        return EEGTCNet(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown torch EEG model: {model_name}")


TORCH_MODEL_NAMES = ("cnn", "raw_resnet", "shallow_convnet", "eeg_tcnet")


def train_cnn_subjects(
    X: np.ndarray,              # EEG data: shape (samples, channels, time_steps)
    y: np.ndarray,              # Labels: shape (samples,), values 0-3
    subjects: np.ndarray,       # Subject IDs: shape (samples,)
    random_state: int = 42,     # Seed for reproducibility
    test_size: float = 0.25,    # 25% of data for testing
    epochs: int = 8,            # 8 training iterations
    batch_size: int = 32,       # Process 32 samples at a time
    model_name: str = "cnn",    # One of TORCH_MODEL_NAMES
) -> list[CnnResult]:           # Returns list of results per subject
    # Get PyTorch imports (raises error if not installed)
    torch, nn, DataLoader, TensorDataset = _require_torch()
    # Set random seed for reproducible results
    torch.manual_seed(random_state)

    # Store results for each subject
    results: list[CnnResult] = []
    # Use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train one model per subject
    for subject in sorted(set(subjects.astype(str))):
        # Get all data for this subject
        mask = subjects.astype(str) == subject
        # Split into train (75%) and test (25%)
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=test_size, random_state=random_state, stratify=y[mask]
        )
        # Calculate mean and std from training data only
        # Average across samples and time
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        # Std dev + small value to avoid division by zero
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-6
        # Normalize both sets using training statistics
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Convert numpy arrays to PyTorch tensors
        # Add channel dimension
        train_ds = TensorDataset(
            torch.tensor(X_train[:, None, :, :], dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        test_tensor = torch.tensor(X_test[:, None, :, :], dtype=torch.float32).to(device)
        # Create model and move to device
        model = build_torch_model(model_name, n_channels=X.shape[1]).to(device)
        # Adam optimizer: adapts learning rate per parameter
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        # Cross-entropy loss for classification
        loss_fn = nn.CrossEntropyLoss()
        # DataLoader: batches, shuffles, and feeds data
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Training loop
        history: list[dict[str, float]] = []
        for epoch in range(1, epochs + 1):
            # Set to training mode (enables dropout)
            model.train()
            losses = []
            # Loop through batches
            for xb, yb in loader:
                # Move batch to device
                xb = xb.to(device)
                yb = yb.to(device)
                # Clear previous gradients
                optimizer.zero_grad()
                # Forward pass: predict
                loss = loss_fn(model(xb), yb)
                # Backward pass: compute gradients
                loss.backward()
                # Update weights
                optimizer.step()
                # Store loss
                losses.append(float(loss.detach().cpu()))
            # Record average loss for this epoch
            history.append({"epoch": float(epoch), "loss": float(np.mean(losses))})

        # Evaluation phase
        # Set to evaluation mode (disables dropout)
        model.eval()
        # Don't compute gradients
        with torch.no_grad():
            # Predict class
            y_pred = model(test_tensor).argmax(dim=1).cpu().numpy()

        # Create result object
        results.append(
            CnnResult(
                model=model_name,
                subject=subject,
                accuracy=float(accuracy_score(y_test, y_pred)),
                macro_f1=float(f1_score(y_test, y_pred, average="macro")),
                confusion=confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]),
                history=history,
            )
        )
    # Return all results
    return results
