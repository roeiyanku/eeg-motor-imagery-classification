from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from .config import ANNOTATION_TO_CLASS, CLASS_NAMES, CLASS_TO_LABEL, MNE_EVENT_ID


@dataclass(frozen=True)
class EpochConfig:
    data_dir: Path
    tmin: float = 0.5
    tmax: float = 4.0
    l_freq: float = 8.0
    h_freq: float = 30.0
    resample: float | None = 125.0
    subjects: tuple[str, ...] | None = None
    eog_correction: str = "none"


def training_files(data_dir: Path, subjects: tuple[str, ...] | None = None) -> list[Path]:
    files = sorted(data_dir.glob("A*T.gdf"))
    if subjects:
        wanted = {s.upper() for s in subjects}
        files = [path for path in files if path.stem[:3].upper() in wanted]
    return files


def eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    names = [name for name in raw.ch_names if not name.upper().startswith("EOG")]
    if len(names) != 22:
        raise ValueError(f"Expected 22 EEG channels after removing EOG, got {len(names)}")
    return names


def annotation_counts(raw: mne.io.BaseRaw) -> dict[str, int]:
    descriptions = raw.annotations.description.astype(str)
    keys = sorted(set(descriptions), key=lambda item: (len(item), item))
    return {key: int(np.sum(descriptions == key)) for key in keys}


def load_raw(path: Path, preload: bool = False) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_gdf(path, preload=preload, verbose="ERROR")
    eog_names = [name for name in raw.ch_names if name.upper().startswith("EOG")]
    if eog_names:
        raw.set_channel_types({name: "eog" for name in eog_names}, verbose="ERROR")
    return raw


def _annotation_sample_mask(raw: mne.io.BaseRaw, descriptions: set[str]) -> np.ndarray:
    """Return a sample mask for annotations whose descriptions match."""
    mask = np.zeros(raw.n_times, dtype=bool)
    sfreq = float(raw.info["sfreq"])
    for annotation in raw.annotations:
        if str(annotation["description"]) not in descriptions:
            continue
        start = max(0, int(round(float(annotation["onset"]) * sfreq)))
        stop = min(raw.n_times, int(round((float(annotation["onset"]) + float(annotation["duration"])) * sfreq)))
        if stop > start:
            mask[start:stop] = True
    return mask


def apply_eog_regression(raw: mne.io.BaseRaw) -> None:
    """Subtract EOG-predicted activity from EEG channels using calibration blocks.

    Dataset 2a starts with eyes-open, eyes-closed, and eye-movement recordings.
    We use those annotated samples to estimate a linear EOG-to-EEG leakage model,
    then subtract the predicted leakage from the full continuous EEG signal.
    """
    if not raw.preload:
        raise RuntimeError("EOG regression requires preloaded raw data.")

    eeg_names = eeg_channel_names(raw)
    eog_names = [name for name in raw.ch_names if name.upper().startswith("EOG")]
    if not eog_names:
        raise ValueError("No EOG channels found for EOG regression.")

    calibration_mask = _annotation_sample_mask(raw, {"276", "277", "1072"})
    if int(np.sum(calibration_mask)) < len(eog_names) + 1:
        raise ValueError("Not enough EOG calibration samples found for regression.")

    eeg_idx = [raw.ch_names.index(name) for name in eeg_names]
    eog_idx = [raw.ch_names.index(name) for name in eog_names]

    eog_cal = raw.get_data(picks=eog_idx)[:, calibration_mask].T
    eeg_cal = raw.get_data(picks=eeg_idx)[:, calibration_mask].T
    eog_mean = eog_cal.mean(axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(eog_cal - eog_mean, eeg_cal, rcond=None)

    eog_all = raw.get_data(picks=eog_idx).T
    correction = (eog_all - eog_mean) @ beta
    raw._data[eeg_idx, :] -= correction.T


def extract_subject_epochs(path: Path, config: EpochConfig) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    raw = load_raw(path, preload=True)
    if config.eog_correction == "regression":
        apply_eog_regression(raw)
    picks = eeg_channel_names(raw)
    raw.pick(picks)
    raw.filter(config.l_freq, config.h_freq, fir_design="firwin", verbose="ERROR")
    if config.resample:
        raw.resample(config.resample, verbose="ERROR")

    event_id = {code: MNE_EVENT_ID[name] for code, name in ANNOTATION_TO_CLASS.items()}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="ERROR")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=MNE_EVENT_ID,
        tmin=config.tmin,
        tmax=config.tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        verbose="ERROR",
    )
    labels = epochs.events[:, 2] - 1
    return epochs.get_data(copy=True).astype(np.float32), labels.astype(np.int64), epochs.ch_names, float(epochs.info["sfreq"])


def prepare_dataset(config: EpochConfig) -> dict[str, np.ndarray | list[str] | float]:
    files = training_files(config.data_dir, config.subjects)
    if not files:
        raise FileNotFoundError(f"No training GDF files found in {config.data_dir}")

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_subjects: list[np.ndarray] = []
    ch_names: list[str] | None = None
    sfreq: float | None = None

    for path in files:
        subject = path.stem[:3]
        x, y, names, subject_sfreq = extract_subject_epochs(path, config)
        if ch_names is None:
            ch_names = names
        elif ch_names != names:
            raise ValueError(f"Channel mismatch for {path.name}")
        sfreq = subject_sfreq
        all_x.append(x)
        all_y.append(y)
        all_subjects.append(np.array([subject] * len(y)))

    return {
        "X": np.concatenate(all_x),
        "y": np.concatenate(all_y),
        "subjects": np.concatenate(all_subjects),
        "ch_names": ch_names or [],
        "class_names": CLASS_NAMES,
        "sfreq": float(sfreq or 0.0),
        "tmin": config.tmin,
        "tmax": config.tmax,
        "l_freq": config.l_freq,
        "h_freq": config.h_freq,
        "eog_correction": config.eog_correction,
    }


def save_prepared_dataset(dataset: dict[str, np.ndarray | list[str] | float], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)


def load_prepared_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {path}. Run `python pipeline.py prepare` first.")
    return dict(np.load(path, allow_pickle=True))


def validate_label_counts(y: np.ndarray, subjects: np.ndarray) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for subject in sorted(set(subjects.astype(str))):
        mask = subjects.astype(str) == subject
        summary[subject] = {
            CLASS_NAMES[label]: int(np.sum(y[mask] == label))
            for label in range(len(CLASS_NAMES))
        }
    return summary
