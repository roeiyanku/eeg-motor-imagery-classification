"""Competition-style benchmark: train on the ``*T`` set, test on ``*E``.

This mirrors the BCI Competition IV 2a evaluation protocol used to report the
winning kappa of ~0.57 (Ang et al., 2008). For each subject we train on all
labeled trials from the calibration file (``A0XT.gdf``) and score on the
held-out evaluation file (``A0XE.gdf``), whose true labels live in
``true_labels/A0XE.mat``. Kappa is averaged across the nine subjects so the
number is directly comparable to the published benchmark.
"""
from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import scipy.io as sio
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.pipeline import Pipeline

from ..io.config import DATA_DIR
from ..decoding.cnn import TORCH_MODEL_NAMES, fit_torch_model, predict_torch_model
from ..io.data import eeg_channel_names, load_raw
from ..decoding.decoders import DECODER_NAMES, BandPass, build_decoder, _bandpass
from ..decoding.models import classical_models
from ..results import EvalResult

TRUE_LABELS_DIR = Path("data") / "true_labels"
SUBJECTS = tuple(f"A0{i}" for i in range(1, 10))
CLASSICAL_BENCHMARK_NAMES = ("logistic_regression", "svm", "random_forest")
BENCHMARK_MODEL_NAMES = DECODER_NAMES + CLASSICAL_BENCHMARK_NAMES + TORCH_MODEL_NAMES

# Broadband epoching shared by every decoder; each decoder re-filters internally.
BROAD_L_FREQ = 4.0
BROAD_H_FREQ = 40.0


def _load_true_labels(path: Path) -> np.ndarray:
    """Return 0-indexed class labels from a ``true_labels`` MAT file."""
    mat = sio.loadmat(path)
    labels = np.asarray(mat["classlabel"]).ravel().astype(int)
    return labels - 1


def _epochs_from_file(
    path: Path,
    cue_event_ids: dict[str, int],
    tmin: float,
    tmax: float,
    resample: float | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Load broadband EEG epochs and their annotation-derived event codes."""
    raw = load_raw(path, preload=True)
    raw.pick(eeg_channel_names(raw))
    raw.filter(BROAD_L_FREQ, BROAD_H_FREQ, fir_design="firwin", verbose="ERROR")
    if resample:
        raw.resample(resample, verbose="ERROR")

    events, _ = mne.events_from_annotations(raw, event_id=cue_event_ids, verbose="ERROR")
    epochs = mne.Epochs(
        raw,
        events,
        event_id=cue_event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        picks="eeg",
        verbose="ERROR",
    )
    X = epochs.get_data(copy=True).astype(np.float64)
    codes = epochs.events[:, 2]
    return X, codes, float(epochs.info["sfreq"])


def load_subject_train_eval(
    subject: str,
    data_dir: Path = DATA_DIR,
    tmin: float = 0.5,
    tmax: float = 4.0,
    resample: float | None = 125.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return ``(X_train, y_train, X_eval, y_eval, sfreq)`` for one subject."""
    train_ids = {"769": 0, "770": 1, "771": 2, "772": 3}
    X_train, codes, sfreq = _epochs_from_file(
        data_dir / f"{subject}T.gdf", train_ids, tmin, tmax, resample
    )
    y_train = codes.astype(int)  # already 0-indexed by cue_event_ids mapping

    eval_ids = {"783": 0}  # evaluation cues are all "unknown"; labels come from MAT
    X_eval, _, _ = _epochs_from_file(
        data_dir / f"{subject}E.gdf", eval_ids, tmin, tmax, resample
    )
    y_eval = _load_true_labels(TRUE_LABELS_DIR / f"{subject}E.mat")

    if len(y_eval) != len(X_eval):
        raise ValueError(
            f"{subject}: {len(X_eval)} eval epochs but {len(y_eval)} labels; "
            "check that the E.gdf and true-label file correspond."
        )
    return X_train, y_train, X_eval, y_eval, sfreq


def load_subject_eval(
    subject: str,
    data_dir: Path = DATA_DIR,
    tmin: float = 0.5,
    tmax: float = 4.0,
    resample: float | None = 125.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(X_eval, y_eval, sfreq)`` for one subject's evaluation file."""
    eval_ids = {"783": 0}
    X_eval, _, sfreq = _epochs_from_file(
        data_dir / f"{subject}E.gdf", eval_ids, tmin, tmax, resample
    )
    y_eval = _load_true_labels(TRUE_LABELS_DIR / f"{subject}E.mat")
    if len(y_eval) != len(X_eval):
        raise ValueError(
            f"{subject}: {len(X_eval)} eval epochs but {len(y_eval)} labels; "
            "check that the E.gdf and true-label file correspond."
        )
    return X_eval, y_eval, sfreq


def _build_classical_benchmark_model(name: str, sfreq: float, random_state: int) -> Pipeline:
    """Return a classical model with benchmark-time 8-30 Hz filtering."""
    model = classical_models(random_state)[name]
    return Pipeline([("band", BandPass(sfreq, 8.0, 30.0)), *model.steps])


def _predict_cnn_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    sfreq: float,
    random_state: int,
    epochs: int,
    batch_size: int = 32,
    val_frac: float = 0.2,
    patience: int = 25,
    model_name: str = "cnn",
) -> np.ndarray:
    """Train the compact CNN on T epochs and predict E epochs for one subject.

    Band-passes to 8-30 Hz, then delegates to the shared torch trainer with a
    stratified validation split, cosine LR schedule, and best-checkpoint early
    stopping, so predictions come from the epoch that generalised best rather
    than from a fixed, arbitrarily short number of passes over the data.
    """
    X_train = _bandpass(X_train, sfreq, 8.0, 30.0)
    X_eval = _bandpass(X_eval, sfreq, 8.0, 30.0)
    model, mean, std, device, _ = fit_torch_model(
        X_train,
        y_train,
        model_name=model_name,
        random_state=random_state,
        epochs=epochs,
        batch_size=batch_size,
        val_frac=val_frac,
        patience=patience,
        use_scheduler=True,
    )
    return predict_torch_model(model, X_eval, mean, std, device)


def run_benchmark(
    model_names: list[str],
    data_dir: Path = DATA_DIR,
    subjects: tuple[str, ...] = SUBJECTS,
    tmin: float = 0.5,
    tmax: float = 4.0,
    resample: float | None = 125.0,
    random_state: int = 42,
    cnn_epochs: int = 120,
) -> list[EvalResult]:
    """Train each selected model on ``T`` and score on ``E`` for every subject."""
    mne.set_log_level("WARNING")
    results: list[EvalResult] = []
    for subject in subjects:
        X_train, y_train, X_eval, y_eval, sfreq = load_subject_train_eval(
            subject, data_dir=data_dir, tmin=tmin, tmax=tmax, resample=resample
        )
        print(
            f"{subject}: train={X_train.shape} eval={X_eval.shape} sfreq={sfreq:.0f}Hz",
            flush=True,
        )
        for name in model_names:
            if name in DECODER_NAMES:
                model = build_decoder(name, sfreq=sfreq, random_state=random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_eval)
            elif name in CLASSICAL_BENCHMARK_NAMES:
                model = _build_classical_benchmark_model(name, sfreq, random_state)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_eval)
            elif name in TORCH_MODEL_NAMES:
                y_pred = _predict_cnn_benchmark(
                    X_train,
                    y_train,
                    X_eval,
                    sfreq,
                    random_state=random_state,
                    epochs=cnn_epochs,
                    model_name=name,
                )
            else:
                raise ValueError(f"Unknown benchmark model: {name}")
            acc = float(accuracy_score(y_eval, y_pred))
            kappa = float(cohen_kappa_score(y_eval, y_pred))
            results.append(
                EvalResult(
                    model=name,
                    subject=subject,
                    accuracy=acc,
                    kappa=kappa,
                    confusion=confusion_matrix(y_eval, y_pred, labels=[0, 1, 2, 3]),
                )
            )
            print(f"    {name:10s} acc={acc:.3f} kappa={kappa:.3f}", flush=True)
    return results


def run_pooled_benchmark(
    model_names: list[str],
    data_dir: Path = DATA_DIR,
    subjects: tuple[str, ...] = SUBJECTS,
    tmin: float = 0.5,
    tmax: float = 4.0,
    resample: float | None = 125.0,
    random_state: int = 42,
) -> list[EvalResult]:
    """Train one pooled model on all subjects' ``T`` files, then score each ``E`` file."""
    unsupported = [name for name in model_names if name not in DECODER_NAMES and name not in CLASSICAL_BENCHMARK_NAMES]
    if unsupported:
        raise ValueError(
            "Pooled benchmark currently supports classical/decoder models only, not neural models: "
            f"{unsupported}"
        )

    mne.set_log_level("WARNING")
    train_ids = {"769": 0, "770": 1, "771": 2, "772": 3}
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    sfreq: float | None = None
    for subject in subjects:
        X_train, codes, subject_sfreq = _epochs_from_file(
            data_dir / f"{subject}T.gdf", train_ids, tmin, tmax, resample
        )
        X_parts.append(X_train)
        y_parts.append(codes.astype(int))
        sfreq = subject_sfreq
        print(f"{subject}: pooled train part={X_train.shape}", flush=True)

    X_pool = np.concatenate(X_parts)
    y_pool = np.concatenate(y_parts)
    if sfreq is None:
        raise ValueError("No subjects were provided for pooled benchmark.")
    print(f"Pooled train={X_pool.shape} sfreq={sfreq:.0f}Hz", flush=True)

    results: list[EvalResult] = []
    for name in model_names:
        if name in DECODER_NAMES:
            model = build_decoder(name, sfreq=sfreq, random_state=random_state)
        elif name in CLASSICAL_BENCHMARK_NAMES:
            model = _build_classical_benchmark_model(name, sfreq, random_state)
        else:
            raise ValueError(f"Unknown pooled benchmark model: {name}")

        print(f"Training pooled {name}...", flush=True)
        model.fit(X_pool, y_pool)
        for subject in subjects:
            X_eval, y_eval, _ = load_subject_eval(
                subject, data_dir=data_dir, tmin=tmin, tmax=tmax, resample=resample
            )
            y_pred = model.predict(X_eval)
            acc = float(accuracy_score(y_eval, y_pred))
            kappa = float(cohen_kappa_score(y_eval, y_pred))
            results.append(
                EvalResult(
                    model=f"pooled_{name}",
                    subject=subject,
                    accuracy=acc,
                    kappa=kappa,
                    confusion=confusion_matrix(y_eval, y_pred, labels=[0, 1, 2, 3]),
                )
            )
            print(f"    {subject} {name:18s} acc={acc:.3f} kappa={kappa:.3f}", flush=True)
    return results
