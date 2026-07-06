"""Live EEG cursor demo using Lab Streaming Layer (LSL).

This module is the bridge from a real EEG stream into the existing motor
imagery decoder. It trains a decoder from the local Dataset 2a calibration
recording, listens to an LSL EEG stream, keeps a rolling window, and converts
decoded class probabilities into a virtual cursor velocity.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, DATA_DIR
from .calibration import class_names_for_dataset2a, load_calibration_model, movement_vector
from .cursor_demo import CLASS_DIRECTION, DIRECTION_LABEL, _load_raw_epoched
from .decoders import build_decoder


@dataclass
class LiveState:
    cursor: np.ndarray
    decoded: int
    probs: np.ndarray


class RollingEegBuffer:
    """Fixed-size sample buffer that returns windows as channels x samples."""

    def __init__(self, n_channels: int, n_samples: int):
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.samples: deque[np.ndarray] = deque(maxlen=n_samples)

    def extend(self, chunk: list[list[float]], channel_indices: list[int]) -> None:
        for sample in chunk:
            arr = np.asarray(sample, dtype=np.float64)
            self.samples.append(arr[channel_indices])

    def ready(self) -> bool:
        return len(self.samples) == self.n_samples

    def window(self) -> np.ndarray:
        if not self.ready():
            raise RuntimeError("EEG buffer is not full yet.")
        return np.stack(self.samples, axis=1)


def _require_pylsl():
    try:
        from pylsl import StreamInlet, resolve_byprop, resolve_streams
    except ImportError as exc:
        raise RuntimeError(
            "Live EEG input requires pylsl. Install it with "
            "`python -m pip install pylsl`, then start your EEG LSL stream."
        ) from exc
    return StreamInlet, resolve_byprop, resolve_streams


def train_live_decoder(
    subject: str,
    model: str,
    data_dir: Path,
    sfreq: float,
    win_seconds: float,
):
    """Train the selected decoder on a 2 s motor-imagery window from a T file."""
    win = int(round(win_seconds * sfreq))
    X_train_full, y_train, _ = _load_raw_epoched(
        data_dir / f"{subject}T.gdf",
        {"769": 0, "770": 1, "771": 2, "772": 3},
        resample=sfreq,
    )
    start = int(round(0.5 * sfreq))
    X_train = X_train_full[:, :, start : start + win]

    decoder = build_decoder(model, sfreq=sfreq)
    decoder.fit(X_train, y_train)
    return decoder, win


def resolve_eeg_stream(name: str | None, timeout: float):
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


def run_live_lsl(
    subject: str = "A03",
    model: str = "riemann",
    data_dir: Path = DATA_DIR,
    calibration_model: Path | None = None,
    stream_name: str | None = None,
    channel_indices: list[int] | None = None,
    win_seconds: float = 2.0,
    step_seconds: float = 0.12,
    speed: float = 0.16,
    duration: float | None = None,
    stream_timeout: float = 10.0,
) -> None:
    """Connect to LSL and print continuous cursor-state updates."""
    StreamInlet, _, _ = _require_pylsl()
    info = resolve_eeg_stream(stream_name, stream_timeout)
    stream_sfreq = float(info.nominal_srate())
    if stream_sfreq <= 0:
        raise RuntimeError("The LSL stream does not report a fixed nominal sampling rate.")

    n_stream_channels = int(info.channel_count())
    if calibration_model:
        bundle = load_calibration_model(calibration_model)
        decoder = bundle["decoder"]
        class_names = list(bundle["class_names"])
        channel_indices = list(bundle["channel_indices"])
        win = int(round(float(bundle["window_seconds"]) * stream_sfreq))
        print(f"Loaded personal calibration model: {calibration_model}")
    else:
        if channel_indices is None:
            channel_indices = list(range(min(22, n_stream_channels)))
        if len(channel_indices) != 22:
            raise ValueError(
                f"The Dataset 2a decoder expects 22 EEG channels, but {len(channel_indices)} were selected. "
                "Pass exactly 22 indices with --channels, or use --calibration-model."
            )
        print(f"Training {model} decoder on {subject} at {stream_sfreq:.1f} Hz...")
        decoder, win = train_live_decoder(subject, model, data_dir, stream_sfreq, win_seconds)
        class_names = class_names_for_dataset2a()

    if max(channel_indices) >= n_stream_channels or min(channel_indices) < 0:
        raise ValueError(f"Channel indices must be between 0 and {n_stream_channels - 1}.")

    print(
        f"Connected to LSL EEG stream '{info.name()}' "
        f"({n_stream_channels} channels at {stream_sfreq:.1f} Hz)."
    )

    inlet = StreamInlet(info, max_chunklen=max(1, int(step_seconds * stream_sfreq)))
    buffer = RollingEegBuffer(n_channels=len(channel_indices), n_samples=win)
    cursor = np.zeros(2, dtype=np.float64)
    next_decode = time.monotonic()
    start_time = time.monotonic()

    print("Streaming. Press Ctrl+C to stop.")
    try:
        while duration is None or time.monotonic() - start_time < duration:
            chunk, _ = inlet.pull_chunk(timeout=0.2)
            if chunk:
                buffer.extend(chunk, channel_indices)
            if not buffer.ready() or time.monotonic() < next_decode:
                continue

            window = buffer.window()[None, :, :]
            probs = decoder.predict_proba(window)[0]
            decoded = int(np.argmax(probs))
            velocity = np.zeros(2, dtype=np.float64)
            for class_idx, prob in enumerate(probs):
                if class_idx < len(class_names):
                    velocity += prob * movement_vector(class_names[class_idx])
                elif class_idx in CLASS_DIRECTION:
                    velocity += prob * CLASS_DIRECTION[class_idx]
            cursor = np.clip(cursor + speed * velocity, -1.2, 1.2)

            probs_text = " ".join(
                f"{class_names[i] if i < len(class_names) else i}={probs[i]:.2f}"
                for i in range(len(probs))
            )
            decoded_label = class_names[decoded] if decoded < len(class_names) else str(decoded)
            print(
                f"decoded={decoded_label:>10s} "
                f"cursor=({cursor[0]:+.2f}, {cursor[1]:+.2f})  {probs_text}",
                flush=True,
            )
            next_decode = time.monotonic() + step_seconds
    except KeyboardInterrupt:
        print("\nLive demo stopped.")
