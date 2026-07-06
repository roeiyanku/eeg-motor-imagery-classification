"""Personal EEG calibration recording and model training utilities."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES
from .decoders import build_decoder


DEFAULT_CALIBRATION_CLASSES = ("left_hand", "right_hand", "feet", "tongue", "rest")


def _require_pylsl():
    try:
        from pylsl import StreamInlet, resolve_byprop, resolve_streams
    except ImportError as exc:
        raise RuntimeError(
            "Live EEG calibration requires pylsl. Install it with "
            "`python -m pip install pylsl`, then start your EEG LSL stream."
        ) from exc
    return StreamInlet, resolve_byprop, resolve_streams


def _resolve_eeg_stream(name: str | None, timeout: float):
    """Return the first matching LSL EEG stream info."""
    _, resolve_byprop, resolve_streams = _require_pylsl()
    if name:
        streams = resolve_byprop("name", name, timeout=timeout)
    else:
        streams = resolve_byprop("type", "EEG", timeout=timeout)
        if not streams:
            streams = [stream for stream in resolve_streams(timeout) if stream.type() == "EEG"]
    if not streams:
        detail = f" named {name!r}" if name else ""
        raise RuntimeError(f"No LSL EEG stream{detail} found within {timeout:.1f} seconds.")
    return streams[0]


def _select_channels(sample: list[float], channel_indices: list[int]) -> np.ndarray:
    arr = np.asarray(sample, dtype=np.float64)
    return arr[channel_indices]


def _fit_window_length(samples: list[np.ndarray], n_channels: int, n_samples: int) -> np.ndarray:
    """Return a channels x samples trial, padding or truncating as needed."""
    if samples:
        trial = np.stack(samples, axis=1)
    else:
        trial = np.zeros((n_channels, 0), dtype=np.float64)
    if trial.shape[1] >= n_samples:
        return trial[:, :n_samples]

    padded = np.zeros((n_channels, n_samples), dtype=np.float64)
    padded[:, : trial.shape[1]] = trial
    return padded


def record_lsl_calibration(
    output: Path,
    stream_name: str | None = None,
    channel_indices: list[int] | None = None,
    classes: tuple[str, ...] = DEFAULT_CALIBRATION_CLASSES,
    trials_per_class: int = 20,
    rest_seconds: float = 2.0,
    cue_seconds: float = 1.0,
    imagery_seconds: float = 4.0,
    stream_timeout: float = 10.0,
) -> None:
    """Record a simple cued calibration dataset from an LSL EEG stream.

    Each trial has a rest period, a cue display period, and an imagery recording
    period. The saved NPZ is intentionally simple: ``X`` has shape
    ``(trials, channels, samples)`` and ``y`` contains integer class labels.
    """
    StreamInlet, _, _ = _require_pylsl()
    info = _resolve_eeg_stream(stream_name, stream_timeout)
    sfreq = float(info.nominal_srate())
    if sfreq <= 0:
        raise RuntimeError("The LSL stream does not report a fixed nominal sampling rate.")

    n_stream_channels = int(info.channel_count())
    if channel_indices is None:
        channel_indices = list(range(min(22, n_stream_channels)))
    if len(channel_indices) == 0:
        raise ValueError("At least one EEG channel must be selected.")
    if max(channel_indices) >= n_stream_channels or min(channel_indices) < 0:
        raise ValueError(f"Channel indices must be between 0 and {n_stream_channels - 1}.")

    n_samples = int(round(imagery_seconds * sfreq))
    inlet = StreamInlet(info, max_chunklen=max(1, int(0.25 * sfreq)))
    rng = np.random.default_rng(0)
    schedule = [(label, name) for label, name in enumerate(classes) for _ in range(trials_per_class)]
    rng.shuffle(schedule)

    print(
        f"Connected to '{info.name()}' ({n_stream_channels} channels at {sfreq:.1f} Hz). "
        f"Recording {len(schedule)} trials."
    )
    print("Press Ctrl+C to abort. Try to stay still; jaw/eye movements strongly contaminate EEG.")

    X: list[np.ndarray] = []
    y: list[int] = []
    try:
        for trial_idx, (label, class_name) in enumerate(schedule, start=1):
            print(f"\nTrial {trial_idx}/{len(schedule)}")
            print("REST")
            time.sleep(rest_seconds)
            print(f"CUE: {class_name}")
            time.sleep(cue_seconds)
            print("IMAGINE NOW")

            samples: list[np.ndarray] = []
            start = time.monotonic()
            while time.monotonic() - start < imagery_seconds:
                chunk, _ = inlet.pull_chunk(timeout=0.2)
                for sample in chunk:
                    samples.append(_select_channels(sample, channel_indices))

            X.append(_fit_window_length(samples, len(channel_indices), n_samples))
            y.append(label)
            print(f"captured {len(samples)} samples")
    except KeyboardInterrupt:
        print("\nCalibration recording stopped early.")

    if not X:
        raise RuntimeError("No calibration trials were recorded.")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        X=np.stack(X).astype(np.float32),
        y=np.asarray(y, dtype=np.int64),
        class_names=np.asarray(classes),
        sfreq=np.asarray(sfreq, dtype=np.float64),
        channel_indices=np.asarray(channel_indices, dtype=np.int64),
        imagery_seconds=np.asarray(imagery_seconds, dtype=np.float64),
    )
    print(f"\nSaved calibration dataset to: {output}")


def train_calibration_decoder(
    input_path: Path,
    output_model: Path,
    model_name: str = "riemann",
    random_state: int = 42,
) -> None:
    """Train a decoder from a personal calibration NPZ and save it with joblib."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Saving calibration models requires joblib. Install `joblib`.") from exc

    data = np.load(input_path, allow_pickle=True)
    X = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    sfreq = float(np.asarray(data["sfreq"]).item())
    class_names = [str(item) for item in data["class_names"]]

    decoder = build_decoder(model_name, sfreq=sfreq, random_state=random_state)
    decoder.fit(X, y)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "decoder": decoder,
            "model_name": model_name,
            "class_names": class_names,
            "sfreq": sfreq,
            "channel_indices": np.asarray(data["channel_indices"], dtype=np.int64).tolist(),
            "window_seconds": float(np.asarray(data["imagery_seconds"]).item()),
        },
        output_model,
    )
    counts = {class_names[i]: int(np.sum(y == i)) for i in range(len(class_names))}
    print(f"Saved calibration model to: {output_model}")
    print(f"Training trials by class: {counts}")


def load_calibration_model(path: Path) -> dict:
    """Load a saved personal calibration model bundle."""
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("Loading calibration models requires joblib. Install `joblib`.") from exc
    return joblib.load(path)


def movement_vector(class_name: str) -> np.ndarray:
    """Map a calibration class name to cursor velocity; rest/unknown -> no movement."""
    mapping = {
        "left_hand": np.array([-1.0, 0.0]),
        "right_hand": np.array([1.0, 0.0]),
        "feet": np.array([0.0, -1.0]),
        "tongue": np.array([0.0, 1.0]),
        "rest": np.array([0.0, 0.0]),
    }
    return mapping.get(class_name, np.zeros(2, dtype=np.float64))


def class_names_for_dataset2a() -> list[str]:
    return list(CLASS_NAMES)
