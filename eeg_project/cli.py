from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mne
import numpy as np

from .config import DATA_DIR, EVENT_DESCRIPTIONS, PREPARED_FILE, RESULTS_DIR
from .cnn import TORCH_MODEL_NAMES, train_cnn_subjects
from .data import (
    EpochConfig,
    annotation_counts,
    load_prepared_dataset,
    load_raw,
    prepare_dataset,
    save_prepared_dataset,
    training_files,
    validate_label_counts,
)
from .models import evaluate_classical_subjects
from .benchmark import BENCHMARK_MODEL_NAMES, run_benchmark, run_pooled_benchmark
from .calibration import (
    DEFAULT_CALIBRATION_CLASSES,
    record_lsl_calibration,
    record_lsl_calibration_gui,
    train_calibration_decoder,
)
from .decoders import DECODER_NAMES
from .reporting import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_model_comparison,
    write_benchmark,
    write_cnn_history,
    write_metrics,
    write_summary,
)


CLASSICAL_MODEL_NAMES = ["logistic_regression", "svm", "random_forest"]
ALL_MODEL_NAMES = CLASSICAL_MODEL_NAMES + list(TORCH_MODEL_NAMES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BCI Competition IV 2a EEG motor-imagery project pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect", help="Inspect one GDF file.")
    inspect.add_argument("gdf_path", nargs="?", type=Path, default=DATA_DIR / "A01T.gdf")
    inspect.add_argument("--plot", action="store_true", help="Open the MNE raw plot window.")

    prepare = subparsers.add_parser("prepare", help="Extract labeled motor-imagery epochs.")
    prepare.add_argument("--data-dir", type=Path, default=DATA_DIR)
    prepare.add_argument("--output", type=Path, default=PREPARED_FILE)
    prepare.add_argument("--subjects", nargs="*", help="Optional subject IDs, e.g. A01 A02.")
    prepare.add_argument("--tmin", type=float, default=0.5)
    prepare.add_argument("--tmax", type=float, default=4.0)
    prepare.add_argument("--l-freq", type=float, default=8.0)
    prepare.add_argument("--h-freq", type=float, default=30.0)
    prepare.add_argument("--resample", type=float, default=125.0)
    prepare.add_argument(
        "--eog-correction",
        choices=["none", "regression"],
        default="none",
        help="Optionally subtract EOG-predicted activity from EEG before epoching.",
    )

    train = subparsers.add_parser("train", help="Train and evaluate models.")
    train.add_argument("--input", type=Path, default=PREPARED_FILE)
    train.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    train.add_argument("--models", nargs="+", choices=ALL_MODEL_NAMES, default=ALL_MODEL_NAMES)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--test-size", type=float, default=0.25)
    train.add_argument("--cnn-epochs", type=int, default=8)

    bench = subparsers.add_parser(
        "benchmark",
        help="Competition protocol: train on T, test on E, report kappa vs the 2008 winner.",
    )
    bench.add_argument("--data-dir", type=Path, default=DATA_DIR)
    bench.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    bench.add_argument("--models", nargs="+", choices=list(BENCHMARK_MODEL_NAMES), default=list(BENCHMARK_MODEL_NAMES))
    bench.add_argument("--subjects", nargs="*", help="Optional subject IDs, e.g. A01 A02.")
    bench.add_argument("--tmin", type=float, default=0.5)
    bench.add_argument("--tmax", type=float, default=4.0)
    bench.add_argument("--resample", type=float, default=125.0)
    bench.add_argument("--random-state", type=int, default=42)
    bench.add_argument("--cnn-epochs", type=int, default=120)

    pooled = subparsers.add_parser(
        "pooled-benchmark",
        help="Train one model on all subjects' T files pooled together, then test each E file.",
    )
    pooled.add_argument("--data-dir", type=Path, default=DATA_DIR)
    pooled.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    pooled.add_argument(
        "--models",
        nargs="+",
        choices=list(BENCHMARK_MODEL_NAMES),
        default=["riemann_fbcsp_vote"],
    )
    pooled.add_argument("--subjects", nargs="*", help="Optional subject IDs, e.g. A01 A02.")
    pooled.add_argument("--tmin", type=float, default=0.5)
    pooled.add_argument("--tmax", type=float, default=4.0)
    pooled.add_argument("--resample", type=float, default=125.0)
    pooled.add_argument("--random-state", type=int, default=42)

    demo = subparsers.add_parser(
        "demo",
        help="Neuralink-style cursor-control demo: decoded motor imagery drives a cursor.",
    )
    demo.add_argument("--subject", default="A03", help="Subject ID, e.g. A03.")
    demo.add_argument("--model", choices=list(DECODER_NAMES), default="riemann")
    demo.add_argument("--data-dir", type=Path, default=DATA_DIR)
    demo.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    demo.add_argument("--targets", type=int, default=12, help="Number of center-out targets.")
    demo.add_argument("--seed", type=int, default=0)

    live_demo = subparsers.add_parser(
        "live-demo",
        help="Live LSL EEG cursor demo: decode a real EEG stream continuously.",
    )
    live_demo.add_argument("--subject", default="A03", help="Subject ID used for calibration training, e.g. A03.")
    live_demo.add_argument("--model", choices=list(DECODER_NAMES), default="riemann")
    live_demo.add_argument("--data-dir", type=Path, default=DATA_DIR)
    live_demo.add_argument("--calibration-model", type=Path, help="Use a saved personal calibration model instead of Dataset 2a.")
    live_demo.add_argument("--stream-name", help="Optional exact LSL stream name. Defaults to first type=EEG stream.")
    live_demo.add_argument(
        "--channels",
        type=int,
        nargs=22,
        help="Exactly 22 zero-based LSL channel indices in Dataset 2a channel order. Defaults to first 22.",
    )
    live_demo.add_argument("--window-seconds", type=float, default=2.0)
    live_demo.add_argument("--step-seconds", type=float, default=0.12)
    live_demo.add_argument("--speed", type=float, default=0.16)
    live_demo.add_argument("--smoothing-windows", type=int, default=5)
    live_demo.add_argument("--confidence-threshold", type=float, default=0.0)
    live_demo.add_argument("--duration", type=float, help="Optional run duration in seconds. Defaults to until Ctrl+C.")
    live_demo.add_argument("--stream-timeout", type=float, default=10.0)

    replay = subparsers.add_parser(
        "replay-live",
        help="Interactive live-style cursor GUI replaying held-out Dataset 2a EEG.",
    )
    replay.add_argument("--subject", default="A03", help="Subject ID, e.g. A03.")
    replay.add_argument("--model", choices=list(DECODER_NAMES), default="riemann_fbcsp_vote")
    replay.add_argument("--data-dir", type=Path, default=DATA_DIR)
    replay.add_argument("--targets", type=int, default=12)
    replay.add_argument("--window-seconds", type=float, default=2.0)
    replay.add_argument("--step-seconds", type=float, default=0.12)
    replay.add_argument("--speed", type=float, default=0.16)
    replay.add_argument("--smoothing-windows", type=int, default=5)
    replay.add_argument("--confidence-threshold", type=float, default=0.0)
    replay.add_argument("--timeout-windows", type=int, default=45)
    replay.add_argument("--seed", type=int, default=0)

    rec = subparsers.add_parser(
        "calibrate-record",
        help="Record a personal cued calibration dataset from an LSL EEG stream.",
    )
    rec.add_argument("--output", type=Path, default=Path("processed") / "personal_calibration.npz")
    rec.add_argument("--stream-name", help="Optional exact LSL stream name. Defaults to first type=EEG stream.")
    rec.add_argument("--channels", type=int, nargs="+", help="Zero-based LSL channel indices to record.")
    rec.add_argument("--classes", nargs="+", default=list(DEFAULT_CALIBRATION_CLASSES))
    rec.add_argument("--trials-per-class", type=int, default=20)
    rec.add_argument("--rest-seconds", type=float, default=2.0)
    rec.add_argument("--cue-seconds", type=float, default=1.0)
    rec.add_argument("--imagery-seconds", type=float, default=4.0)
    rec.add_argument("--stream-timeout", type=float, default=10.0)

    rec_gui = subparsers.add_parser(
        "calibrate-gui",
        help="Record a personal calibration dataset with a simple arrow/cue GUI.",
    )
    rec_gui.add_argument("--output", type=Path, default=Path("processed") / "personal_calibration.npz")
    rec_gui.add_argument("--stream-name", help="Optional exact LSL stream name. Defaults to first type=EEG stream.")
    rec_gui.add_argument("--channels", type=int, nargs="+", help="Zero-based LSL channel indices to record.")
    rec_gui.add_argument("--classes", nargs="+", default=list(DEFAULT_CALIBRATION_CLASSES))
    rec_gui.add_argument("--trials-per-class", type=int, default=20)
    rec_gui.add_argument("--rest-seconds", type=float, default=2.0)
    rec_gui.add_argument("--cue-seconds", type=float, default=1.0)
    rec_gui.add_argument("--imagery-seconds", type=float, default=4.0)
    rec_gui.add_argument("--stream-timeout", type=float, default=10.0)

    cal_train = subparsers.add_parser(
        "calibrate-train",
        help="Train and save a decoder from a personal calibration NPZ.",
    )
    cal_train.add_argument("--input", type=Path, default=Path("processed") / "personal_calibration.npz")
    cal_train.add_argument("--output", type=Path, default=Path("models") / "personal_riemann.joblib")
    cal_train.add_argument("--model", choices=list(DECODER_NAMES), default="riemann")
    cal_train.add_argument("--random-state", type=int, default=42)

    return parser


def inspect_file(path: Path, plot: bool) -> None:
    raw = load_raw(path, preload=plot)
    print(raw.info)
    print(f"\nFile: {path}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Samples: {raw.n_times}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")
    print("\nAnnotation counts:")
    for code, count in annotation_counts(raw).items():
        description = EVENT_DESCRIPTIONS.get(code, "unknown")
        print(f"  {code:>5} ({description}): {count}")
    if plot:
        raw.plot(block=True)


def prepare(args: argparse.Namespace) -> None:
    subjects = tuple(args.subjects) if args.subjects else None
    files = training_files(args.data_dir, subjects)
    print(f"Found {len(files)} training files:")
    for path in files:
        print(f"  {path.name}")

    config = EpochConfig(
        data_dir=args.data_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        resample=args.resample,
        subjects=subjects,
        eog_correction=args.eog_correction,
    )
    dataset = prepare_dataset(config)
    save_prepared_dataset(dataset, args.output)

    X = dataset["X"]
    y = dataset["y"]
    subjects_array = dataset["subjects"]
    print(f"\nSaved: {args.output}")
    print(f"Epochs shape: {X.shape} (trials, EEG channels, time samples)")
    print(f"Labels shape: {y.shape}")
    print(f"EOG correction: {dataset['eog_correction']}")
    print(f"Subjects: {sorted(set(subjects_array.astype(str)))}")
    print("\nClass counts per subject:")
    for subject, counts in validate_label_counts(y, subjects_array).items():
        print(f"  {subject}: {counts}")


def train(args: argparse.Namespace) -> None:
    mne.set_log_level("WARNING")
    dataset = load_prepared_dataset(args.input)
    X = dataset["X"]
    y = dataset["y"]
    subjects = dataset["subjects"].astype(str)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_classical = [name for name in args.models if name in CLASSICAL_MODEL_NAMES]
    results = []
    if selected_classical:
        results.extend(
            evaluate_classical_subjects(
                X,
                y,
                subjects,
                selected_classical,
                random_state=args.random_state,
                test_size=args.test_size,
            )
        )

    cnn_results = []
    for model_name in TORCH_MODEL_NAMES:
        if model_name in args.models:
            model_results = train_cnn_subjects(
                X,
                y,
                subjects,
                random_state=args.random_state,
                test_size=args.test_size,
                epochs=args.cnn_epochs,
                model_name=model_name,
            )
            cnn_results.extend(model_results)
            results.extend(model_results)

    metrics = write_metrics(results, args.output_dir)
    plot_class_distribution(y, args.output_dir)
    plot_model_comparison(metrics, args.output_dir)
    plot_confusion_matrices(results, args.output_dir)
    write_summary(metrics, args.output_dir)
    write_cnn_history(cnn_results, args.output_dir)

    print(f"Saved results to: {args.output_dir}")
    print(metrics.groupby("model")[["accuracy", "macro_f1"]].mean().round(3))


def benchmark(args: argparse.Namespace) -> None:
    subjects = tuple(s.upper() for s in args.subjects) if args.subjects else None
    kwargs = {"subjects": subjects} if subjects else {}
    results = run_benchmark(
        args.models,
        data_dir=args.data_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        resample=args.resample,
        random_state=args.random_state,
        cnn_epochs=args.cnn_epochs,
        **kwargs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = write_benchmark(results, args.output_dir)
    print(f"\nSaved benchmark results to: {args.output_dir}")
    summary = df.groupby("model")[["accuracy", "kappa"]].mean().round(3).sort_values("kappa", ascending=False)
    print(summary)
    print("\n2008 winning kappa (FBCSP): 0.57")


def pooled_benchmark(args: argparse.Namespace) -> None:
    subjects = tuple(s.upper() for s in args.subjects) if args.subjects else None
    kwargs = {"subjects": subjects} if subjects else {}
    results = run_pooled_benchmark(
        args.models,
        data_dir=args.data_dir,
        tmin=args.tmin,
        tmax=args.tmax,
        resample=args.resample,
        random_state=args.random_state,
        **kwargs,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = write_benchmark(results, args.output_dir)
    print(f"\nSaved pooled benchmark results to: {args.output_dir}")
    summary = df.groupby("model")[["accuracy", "kappa"]].mean().round(3).sort_values("kappa", ascending=False)
    print(summary)


def demo(args: argparse.Namespace) -> None:
    from .cursor_demo import render_gif, simulate_session

    print(f"Training streaming decoder ({args.model}) on {args.subject} and running cursor task...")
    frames, stats = simulate_session(
        subject=args.subject.upper(),
        model=args.model,
        n_targets=args.targets,
        seed=args.seed,
        data_dir=args.data_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f"cursor_demo_{args.subject.upper()}_{args.model}.gif"
    render_gif(frames, stats, out)
    print(
        f"Targets hit: {stats['hits']}/{stats['targets']} "
        f"(hit rate {stats['hit_rate']:.0%})"
    )
    print(f"Saved cursor demo animation to: {out}")


def live_demo(args: argparse.Namespace) -> None:
    from .live_demo import run_live_lsl

    try:
        run_live_lsl(
            subject=args.subject.upper(),
            model=args.model,
            data_dir=args.data_dir,
            calibration_model=args.calibration_model,
            stream_name=args.stream_name,
            channel_indices=args.channels,
            win_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            speed=args.speed,
            smoothing_windows=args.smoothing_windows,
            confidence_threshold=args.confidence_threshold,
            duration=args.duration,
            stream_timeout=args.stream_timeout,
        )
    except RuntimeError as exc:
        print(f"Live demo error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def replay_live(args: argparse.Namespace) -> None:
    from .replay_gui import run_replay_gui

    run_replay_gui(
        subject=args.subject.upper(),
        model=args.model,
        data_dir=args.data_dir,
        n_targets=args.targets,
        win_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        speed=args.speed,
        smoothing_windows=args.smoothing_windows,
        confidence_threshold=args.confidence_threshold,
        timeout_windows=args.timeout_windows,
        seed=args.seed,
    )


def calibrate_record(args: argparse.Namespace) -> None:
    try:
        record_lsl_calibration(
            output=args.output,
            stream_name=args.stream_name,
            channel_indices=args.channels,
            classes=tuple(args.classes),
            trials_per_class=args.trials_per_class,
            rest_seconds=args.rest_seconds,
            cue_seconds=args.cue_seconds,
            imagery_seconds=args.imagery_seconds,
            stream_timeout=args.stream_timeout,
        )
    except RuntimeError as exc:
        print(f"Calibration recording error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def calibrate_gui(args: argparse.Namespace) -> None:
    try:
        record_lsl_calibration_gui(
            output=args.output,
            stream_name=args.stream_name,
            channel_indices=args.channels,
            classes=tuple(args.classes),
            trials_per_class=args.trials_per_class,
            rest_seconds=args.rest_seconds,
            cue_seconds=args.cue_seconds,
            imagery_seconds=args.imagery_seconds,
            stream_timeout=args.stream_timeout,
        )
    except RuntimeError as exc:
        print(f"GUI calibration error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def calibrate_train(args: argparse.Namespace) -> None:
    train_calibration_decoder(
        input_path=args.input,
        output_model=args.output,
        model_name=args.model,
        random_state=args.random_state,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "inspect":
        inspect_file(args.gdf_path, args.plot)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "benchmark":
        benchmark(args)
    elif args.command == "pooled-benchmark":
        pooled_benchmark(args)
    elif args.command == "demo":
        demo(args)
    elif args.command == "live-demo":
        live_demo(args)
    elif args.command == "replay-live":
        replay_live(args)
    elif args.command == "calibrate-record":
        calibrate_record(args)
    elif args.command == "calibrate-gui":
        calibrate_gui(args)
    elif args.command == "calibrate-train":
        calibrate_train(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
