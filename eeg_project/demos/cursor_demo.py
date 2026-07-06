"""Neuralink-style cursor-control demo driven by decoded motor imagery.

This turns the offline 4-class decoder into a *continuous* brain-computer
interface, the way an invasive BCI (e.g. Neuralink) drives a cursor from neural
intent -- only here the "intent" is decoded from non-invasive scalp EEG.

Pipeline:

1. Train a streaming decoder on short (2 s) windows from a subject's ``T`` set.
2. Run a center-out task: a target lights up in one of four directions; the
   matching motor-imagery class is streamed from the held-out ``E`` set as a
   sliding 2 s window, decoded continuously, and used to move the cursor.
3. Class probabilities are turned into a velocity vector, so a confident decode
   pushes the cursor firmly toward its direction. Hit rate and an
   information-transfer estimate are reported, mirroring Neuralink's cursor
   benchmark ("Webgrid").

Directions:
    left_hand  -> left      right_hand -> right
    feet       -> down      tongue     -> up
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..eval.benchmark import BROAD_H_FREQ, BROAD_L_FREQ, TRUE_LABELS_DIR, _load_true_labels
from ..io.config import CLASS_NAMES, DATA_DIR
from ..io.data import eeg_channel_names, load_raw
from ..decoding.decoders import build_decoder

# Class index -> unit velocity direction (x right, y up).
CLASS_DIRECTION = {
    0: np.array([-1.0, 0.0]),  # left_hand  -> left
    1: np.array([1.0, 0.0]),   # right_hand -> right
    2: np.array([0.0, -1.0]),  # feet       -> down
    3: np.array([0.0, 1.0]),   # tongue     -> up
}
DIRECTION_LABEL = {0: "LEFT", 1: "RIGHT", 2: "DOWN", 3: "UP"}


@dataclass
class Frame:
    cursor: np.ndarray
    target_dir: int
    decoded: int
    probs: np.ndarray
    hits: int
    attempts: int
    trail: list = field(default_factory=list)


def _load_raw_epoched(path: Path, cue_ids: dict[str, int], resample: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Load broadband trials as ``(trials, channels, samples)`` from cue onset (0 s) to 4 s."""
    import mne

    mne.set_log_level("ERROR")
    raw = load_raw(path, preload=True)
    raw.pick(eeg_channel_names(raw))
    raw.filter(BROAD_L_FREQ, BROAD_H_FREQ, fir_design="firwin", verbose="ERROR")
    raw.resample(resample, verbose="ERROR")
    events, _ = mne.events_from_annotations(raw, event_id=cue_ids, verbose="ERROR")
    epochs = mne.Epochs(
        raw, events, event_id=cue_ids, tmin=0.0, tmax=4.0,
        baseline=None, preload=True, picks="eeg", verbose="ERROR",
    )
    return epochs.get_data(copy=True).astype(np.float64), epochs.events[:, 2].astype(int), float(epochs.info["sfreq"])


def train_stream_decoder(
    subject: str,
    model: str = "riemann",
    data_dir: Path = DATA_DIR,
    resample: float = 125.0,
    win_seconds: float = 2.0,
):
    """Train a decoder on 2 s windows of the T set; return decoder, E-set windows, sfreq."""
    sfreq = resample
    win = int(win_seconds * sfreq)

    # Training: one representative 2 s window per trial (0.5-2.5 s after cue).
    X_train_full, codes_train, _ = _load_raw_epoched(
        data_dir / f"{subject}T.gdf", {"769": 0, "770": 1, "771": 2, "772": 3}, resample
    )
    start = int(0.5 * sfreq)
    X_train = X_train_full[:, :, start:start + win]
    y_train = codes_train

    decoder = build_decoder(model, sfreq=sfreq)
    decoder.fit(X_train, y_train)

    # Evaluation trials, grouped by true class, for streaming.
    X_eval_full, _, _ = _load_raw_epoched(data_dir / f"{subject}E.gdf", {"783": 0}, resample)
    y_eval = _load_true_labels(TRUE_LABELS_DIR / f"{subject}E.mat")
    trials_by_class: dict[int, list[np.ndarray]] = {c: [] for c in range(4)}
    for trial, label in zip(X_eval_full, y_eval):
        trials_by_class[int(label)].append(trial)

    return decoder, trials_by_class, sfreq, win


def simulate_session(
    subject: str = "A03",
    model: str = "riemann",
    n_targets: int = 12,
    win_seconds: float = 2.0,
    step_seconds: float = 0.12,
    speed: float = 0.16,
    timeout_frames: int = 45,
    seed: int = 0,
    data_dir: Path = DATA_DIR,
) -> tuple[list[Frame], dict]:
    """Run the center-out cursor task and return animation frames + summary stats."""
    rng = np.random.default_rng(seed)
    decoder, trials_by_class, sfreq, win = train_stream_decoder(
        subject, model=model, data_dir=data_dir, win_seconds=win_seconds
    )
    step = max(1, int(step_seconds * sfreq))

    # Round-robin target sequence over the four directions.
    order = []
    dirs = [0, 1, 2, 3]
    for _ in range((n_targets + 3) // 4):
        rng.shuffle(dirs)
        order.extend(dirs)
    order = order[:n_targets]

    cursor_idx = {c: 0 for c in range(4)}  # which eval trial to stream next per class
    frames: list[Frame] = []
    hits = 0
    attempts = 0
    trail: list[np.ndarray] = []

    for target_dir in order:
        attempts += 1
        cursor = np.zeros(2)
        trail = [cursor.copy()]
        reached = False

        trials = trials_by_class[target_dir]
        # Stream sliding windows from consecutive eval trials of this class.
        for f in range(timeout_frames):
            trial = trials[cursor_idx[target_dir] % len(trials)]
            max_start = trial.shape[1] - win
            w_start = min(f * step, max_start)
            if w_start >= max_start:
                cursor_idx[target_dir] += 1  # advance to next trial when window runs out
            window = trial[:, w_start:w_start + win][None, :, :]

            probs = decoder.predict_proba(window)[0]
            decoded = int(np.argmax(probs))

            # Probability-weighted velocity: confident decode -> firm push.
            velocity = sum(probs[c] * CLASS_DIRECTION[c] for c in range(4))
            cursor = cursor + speed * velocity
            cursor = np.clip(cursor, -1.2, 1.2)
            trail.append(cursor.copy())

            frames.append(Frame(cursor.copy(), target_dir, decoded, probs.copy(), hits, attempts, list(trail)))

            # Hit test: cursor projects far enough along the target direction.
            if float(cursor @ CLASS_DIRECTION[target_dir]) >= 1.0:
                reached = True
                break
        if reached:
            hits += 1
            # reflect the successful hit in the last frame's counter
            frames[-1] = Frame(frames[-1].cursor, target_dir, frames[-1].decoded,
                               frames[-1].probs, hits, attempts, frames[-1].trail)

    acc = hits / attempts if attempts else 0.0
    stats = {
        "subject": subject,
        "model": model,
        "targets": attempts,
        "hits": hits,
        "hit_rate": acc,
    }
    return frames, stats


def render_gif(frames: list[Frame], stats: dict, output: Path, fps: int = 12) -> None:
    """Render the session to an animated GIF."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw(i: int):
        ax.clear()
        fr = frames[i]
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        # Target zones on the four sides; active one highlighted.
        zones = {0: (-1.25, 0), 1: (1.25, 0), 2: (0, -1.25), 3: (0, 1.25)}
        for d, (x, y) in zones.items():
            active = d == fr.target_dir
            ax.scatter([x], [y], s=900 if active else 300,
                       c="#e9c46a" if active else "#cccccc",
                       marker="s", edgecolors="black", zorder=1)
            ax.text(x, y, DIRECTION_LABEL[d], ha="center", va="center", fontsize=8, zorder=2)

        # Cursor trail + head.
        trail = np.array(fr.trail)
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], color="#264653", alpha=0.4, lw=1.5, zorder=2)
        ax.scatter([fr.cursor[0]], [fr.cursor[1]], s=260, c="#2a9d8f", edgecolors="black", zorder=3)

        decoded_ok = fr.decoded == fr.target_dir
        ax.set_title(
            f"{stats['subject']} · {stats['model']} · target {DIRECTION_LABEL[fr.target_dir]}"
            f"  |  decoded {DIRECTION_LABEL[fr.decoded]} {'✓' if decoded_ok else '·'}\n"
            f"targets hit {fr.hits}/{fr.attempts}   (thinking → cursor moves)",
            fontsize=10,
        )

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 // fps)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
