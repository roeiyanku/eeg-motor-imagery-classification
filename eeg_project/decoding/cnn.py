from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from ..results import EvalResult


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


def fit_torch_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_name: str,
    random_state: int,
    epochs: int,
    batch_size: int = 32,
    val_frac: float | None = None,
    patience: int | None = None,
    use_scheduler: bool = False,
):
    """Train an EEG torch model and return ``(model, mean, std, device, history)``.

    Data is per-channel z-scored using statistics computed from ``X`` here; the
    returned ``mean``/``std`` must be applied to any evaluation set before
    :func:`predict_torch_model`. When ``val_frac`` is given, a stratified
    validation split drives best-checkpoint early stopping (``patience``) and an
    optional cosine LR schedule, so both the quick within-subject trainer and the
    competition benchmark share one loop.
    """
    torch, nn, DataLoader, TensorDataset = _require_torch()
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True) + 1e-6
    Xn = ((X - mean) / std).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if val_frac:
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xn, y, test_size=val_frac, random_state=random_state, stratify=y
        )
        val_tensor = torch.tensor(X_val[:, None, :, :], dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    else:
        X_tr, y_tr = Xn, y
        val_tensor = y_val_t = None

    train_ds = TensorDataset(
        torch.tensor(X_tr[:, None, :, :], dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = build_torch_model(model_name, n_channels=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None
    loss_fn = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_since_best = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if scheduler is not None:
            scheduler.step()
        record = {"epoch": float(epoch), "loss": float(np.mean(losses))}

        if val_tensor is not None:
            model.eval()
            with torch.no_grad():
                val_acc = float((model(val_tensor).argmax(dim=1) == y_val_t).float().mean())
            record["val_acc"] = val_acc
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_since_best = 0
            elif patience is not None:
                epochs_since_best += 1
                if epochs_since_best >= patience:
                    history.append(record)
                    break
        history.append(record)

    if val_tensor is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, mean, std, device, history


def predict_torch_model(model, X: np.ndarray, mean, std, device) -> np.ndarray:
    """Z-score ``X`` with the training ``mean``/``std`` and return class predictions."""
    torch, *_ = _require_torch()
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        tensor = torch.tensor(Xn[:, None, :, :], dtype=torch.float32).to(device)
        return model(tensor).argmax(dim=1).cpu().numpy()


def train_cnn_subjects(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    random_state: int = 42,
    test_size: float = 0.25,
    epochs: int = 8,
    batch_size: int = 32,
    model_name: str = "cnn",
) -> list[EvalResult]:
    """Train one CNN per subject on a stratified split and score the held-out part."""
    results: list[EvalResult] = []
    for subject in sorted(set(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        X_train, X_test, y_train, y_test = train_test_split(
            X[mask], y[mask], test_size=test_size, random_state=random_state, stratify=y[mask]
        )
        model, mean, std, device, history = fit_torch_model(
            X_train,
            y_train,
            model_name=model_name,
            random_state=random_state,
            epochs=epochs,
            batch_size=batch_size,
        )
        y_pred = predict_torch_model(model, X_test, mean, std, device)
        results.append(
            EvalResult(
                model=model_name,
                subject=subject,
                accuracy=float(accuracy_score(y_test, y_pred)),
                macro_f1=float(f1_score(y_test, y_pred, average="macro")),
                confusion=confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]),
                history=history,
            )
        )
    return results
