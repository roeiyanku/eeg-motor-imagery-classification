from __future__ import annotations

import numpy as np


def sliding_windows(
    X: np.ndarray,
    sfreq: float,
    window_seconds: float = 4.0,
    stride_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice continuous EEG into overlapping windows.

    Parameters
    ----------
    X:
        Continuous EEG shaped ``(channels, samples)``.
    sfreq:
        Sampling frequency in Hz.
    window_seconds:
        Window duration. The seizure-detection setup from the talk used 4 s.
    stride_seconds:
        Step between consecutive windows. The talk used 1 s for real-time
        inference cadence.

    Returns
    -------
    windows:
        Array shaped ``(n_windows, channels, window_samples)``.
    starts:
        Start sample for each window in the original signal.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected X with shape (channels, samples), got {X.shape}")
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    window_samples = int(round(window_seconds * sfreq))
    stride_samples = int(round(stride_seconds * sfreq))
    if window_samples <= 0 or stride_samples <= 0:
        raise ValueError("window_seconds and stride_seconds must produce at least one sample")
    if X.shape[1] < window_samples:
        return np.empty((0, X.shape[0], window_samples), dtype=X.dtype), np.empty(0, dtype=int)

    starts = np.arange(0, X.shape[1] - window_samples + 1, stride_samples, dtype=int)
    windows = np.stack([X[:, start : start + window_samples] for start in starts])
    return windows, starts


def event_overlap_labels(
    starts: np.ndarray,
    window_samples: int,
    events: list[tuple[int, int]],
    min_overlap_seconds: float,
    sfreq: float,
) -> np.ndarray:
    """Label windows positive when they overlap an event by enough time.

    ``events`` contains ``(start_sample, stop_sample)`` intervals. This is a
    practical helper for seizure-style event detection, where labels describe
    time spans rather than one class per trial.
    """
    min_overlap = int(round(min_overlap_seconds * sfreq))
    labels = np.zeros(len(starts), dtype=np.int64)
    for idx, start in enumerate(np.asarray(starts, dtype=int)):
        stop = start + window_samples
        for event_start, event_stop in events:
            overlap = min(stop, event_stop) - max(start, event_start)
            if overlap >= min_overlap:
                labels[idx] = 1
                break
    return labels
