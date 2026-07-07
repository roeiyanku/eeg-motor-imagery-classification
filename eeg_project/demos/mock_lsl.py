"""Publish a synthetic EEG stream over LSL to test the live pipeline without hardware.

Run this in one terminal:

    python pipeline.py mock-stream

then, in another terminal, run ``calibrate-gui`` or ``live-demo`` -- they
auto-discover the first ``type="EEG"`` stream. This exercises the whole live
path (buffering, windowing, decoding, smoothing, online alignment) with zero
hardware. When a real headset arrives you swap its LSL bridge app in for this
and nothing downstream changes.

The signal is band-limited noise plus a few mu/beta oscillators -- realistic
enough to drive the plumbing, but it is NOT real motor imagery, so decoded
predictions will be roughly random (the cursor drifts). That is expected: this
validates the pipe, not the accuracy.
"""
from __future__ import annotations

import time

import numpy as np


def _require_pylsl():
    try:
        from pylsl import StreamInfo, StreamOutlet, cf_float32
    except ImportError as exc:
        raise RuntimeError(
            "Mock LSL streaming requires pylsl. Install it with "
            "`python -m pip install pylsl`."
        ) from exc
    return StreamInfo, StreamOutlet, cf_float32


def run_mock_stream(
    name: str = "MockEEG",
    n_channels: int = 22,
    sfreq: float = 250.0,
    duration: float | None = None,
    seed: int = 0,
) -> None:
    """Broadcast a synthetic EEG stream at ``sfreq`` Hz until stopped.

    ``n_channels`` defaults to 22 to match the Dataset 2a decoder path; pass a
    different count when testing a personal calibration model.
    """
    StreamInfo, StreamOutlet, cf_float32 = _require_pylsl()

    info = StreamInfo(
        name=name,
        type="EEG",
        channel_count=n_channels,
        nominal_srate=sfreq,
        channel_format=cf_float32,
        source_id="mock-eeg-0",
    )
    channels = info.desc().append_child("channels")
    for i in range(n_channels):
        ch = channels.append_child("channel")
        ch.append_child_value("label", f"CH{i + 1}")
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    outlet = StreamOutlet(info, max_buffered=360)

    rng = np.random.default_rng(seed)
    # Per-channel oscillators in the mu/beta range plus broadband noise.
    freqs = rng.uniform(8.0, 24.0, size=n_channels)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_channels)
    amp = rng.uniform(5.0, 15.0, size=n_channels)  # microvolt-scale

    print(
        f"Mock EEG stream '{name}': {n_channels} channels @ {sfreq:.0f} Hz "
        f"(type=EEG). Leave this running; start live-demo/calibrate-gui in "
        f"another terminal. Ctrl+C to stop."
    )

    period = 1.0 / sfreq
    start = time.monotonic()
    emitted = 0
    try:
        while duration is None or time.monotonic() - start < duration:
            # Catch up to real time so the stream keeps the declared rate.
            target = int((time.monotonic() - start) * sfreq)
            while emitted < target:
                t = emitted * period
                sample = amp * np.sin(2.0 * np.pi * freqs * t + phases)
                sample += rng.standard_normal(n_channels) * 8.0
                outlet.push_sample(sample.astype(np.float32))
                emitted += 1
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    print(f"\nMock stream stopped after {emitted} samples "
          f"({emitted / sfreq:.1f} s).")
