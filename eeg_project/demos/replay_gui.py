"""Interactive live-BCI replay using held-out Dataset 2a EEG trials."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, CENTER, Canvas, Tk, ttk

import numpy as np

from ..io.config import DATA_DIR
from .cursor_demo import CLASS_DIRECTION, DIRECTION_LABEL, train_stream_decoder


@dataclass
class ReplayStats:
    targets: int = 0
    hits: int = 0
    windows: int = 0
    correct_windows: int = 0


def _target_order(n_targets: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    dirs = [0, 1, 2, 3]
    order: list[int] = []
    for _ in range((n_targets + 3) // 4):
        rng.shuffle(dirs)
        order.extend(dirs)
    return order[:n_targets]


def _draw_scene(
    canvas: Canvas,
    cursor: np.ndarray,
    target_dir: int,
    decoded: int | None,
    probs: np.ndarray,
    stats: ReplayStats,
    status: str,
) -> None:
    canvas.delete("all")
    width = int(canvas.winfo_width() or 720)
    height = int(canvas.winfo_height() or 520)
    cx = width // 2
    cy = height // 2
    scale = min(width, height) * 0.32

    canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
    canvas.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill="#d8dee9", outline="")

    zones = {
        0: (cx - scale, cy, "<"),
        1: (cx + scale, cy, ">"),
        2: (cx, cy + scale, "v"),
        3: (cx, cy - scale, "^"),
    }
    for direction, (x, y, arrow) in zones.items():
        active = direction == target_dir
        color = "#f2cc8f" if active else "#34495e"
        canvas.create_rectangle(x - 42, y - 42, x + 42, y + 42, fill=color, outline="#f8f9fa", width=2)
        canvas.create_text(x, y, text=arrow, fill="#101820" if active else "#ecf0f1", font=("Segoe UI", 34, "bold"))

    cursor_x = cx + float(cursor[0]) * scale
    cursor_y = cy - float(cursor[1]) * scale
    canvas.create_oval(cursor_x - 16, cursor_y - 16, cursor_x + 16, cursor_y + 16, fill="#2a9d8f", outline="#ecf0f1", width=2)

    decoded_label = "..." if decoded is None else DIRECTION_LABEL[decoded]
    target_label = DIRECTION_LABEL[target_dir]
    hit_rate = stats.hits / stats.targets if stats.targets else 0.0
    window_acc = stats.correct_windows / stats.windows if stats.windows else 0.0
    probs_text = "  ".join(f"{DIRECTION_LABEL[i]}={probs[i]:.2f}" for i in range(4)) if probs.size else ""
    canvas.create_text(
        cx,
        38,
        text=f"Target: {target_label}    Decoded: {decoded_label}    {status}",
        fill="#f8f9fa",
        font=("Segoe UI", 18, "bold"),
    )
    canvas.create_text(
        cx,
        height - 62,
        text=f"targets hit {stats.hits}/{stats.targets} ({hit_rate:.0%})    window accuracy {window_acc:.0%}",
        fill="#d8dee9",
        font=("Segoe UI", 13),
    )
    canvas.create_text(cx, height - 30, text=probs_text, fill="#d8dee9", font=("Segoe UI", 12))


def _draw_final_scene(canvas: Canvas, stats: ReplayStats) -> None:
    canvas.delete("all")
    width = int(canvas.winfo_width() or 720)
    height = int(canvas.winfo_height() or 520)
    cx = width // 2
    cy = height // 2
    hit_rate = stats.hits / stats.targets if stats.targets else 0.0
    window_acc = stats.correct_windows / stats.windows if stats.windows else 0.0

    canvas.create_rectangle(0, 0, width, height, fill="#101820", outline="")
    canvas.create_text(
        cx,
        cy - 82,
        text="Finished",
        fill="#f8f9fa",
        font=("Segoe UI", 28, "bold"),
    )
    canvas.create_text(
        cx,
        cy - 20,
        text=f"Targets hit: {stats.hits}/{stats.targets} ({hit_rate:.0%})",
        fill="#f2cc8f",
        font=("Segoe UI", 22, "bold"),
    )
    canvas.create_text(
        cx,
        cy + 28,
        text=f"Window accuracy: {window_acc:.0%}",
        fill="#d8dee9",
        font=("Segoe UI", 20, "bold"),
    )
    canvas.create_text(
        cx,
        height - 38,
        text="Close the window to return to the terminal.",
        fill="#9fb3c8",
        font=("Segoe UI", 12),
    )


def run_replay_gui(
    subject: str = "A03",
    model: str = "riemann_fbcsp_vote",
    data_dir: Path = DATA_DIR,
    n_targets: int = 12,
    win_seconds: float = 2.0,
    step_seconds: float = 0.12,
    speed: float = 0.16,
    smoothing_windows: int = 5,
    confidence_threshold: float = 0.0,
    timeout_windows: int = 45,
    seed: int = 0,
) -> None:
    """Replay held-out evaluation EEG through the live cursor GUI."""
    decoder, trials_by_class, sfreq, win = train_stream_decoder(
        subject=subject,
        model=model,
        data_dir=data_dir,
        win_seconds=win_seconds,
    )
    step = max(1, int(round(step_seconds * sfreq)))
    timeout_windows = max(1, timeout_windows)
    n_targets = max(1, n_targets)
    order = _target_order(n_targets, seed)
    trial_idx = {direction: 0 for direction in range(4)}

    root = Tk()
    root.title(f"Dataset 2a live replay - {subject} - {model}")
    root.geometry("900x650")
    frame = ttk.Frame(root, padding=14)
    frame.pack(fill=BOTH, expand=True)
    title = ttk.Label(
        frame,
        text="Dataset 2a Replay-Live BCI Simulator",
        anchor=CENTER,
        font=("Segoe UI", 20, "bold"),
    )
    title.pack(fill=BOTH, pady=(0, 8))
    canvas = Canvas(frame, width=860, height=560, highlightthickness=0)
    canvas.pack(fill=BOTH, expand=True)

    stats = ReplayStats()
    zero_probs = np.zeros(4, dtype=np.float64)
    _draw_scene(canvas, np.zeros(2), order[0], None, zero_probs, stats, "training complete")
    root.update()
    time.sleep(0.8)

    try:
        for target_dir in order:
            stats.targets += 1
            cursor = np.zeros(2, dtype=np.float64)
            probs_history: deque[np.ndarray] = deque(maxlen=max(1, smoothing_windows))
            reached = False
            trials = trials_by_class[target_dir]
            trial = trials[trial_idx[target_dir] % len(trials)]
            trial_idx[target_dir] += 1

            _draw_scene(canvas, cursor, target_dir, None, zero_probs, stats, "cue")
            root.update()
            time.sleep(0.5)

            max_start = max(0, trial.shape[1] - win)
            for window_idx in range(timeout_windows):
                root.update_idletasks()
                root.update()
                w_start = min(window_idx * step, max_start)
                window = trial[:, w_start : w_start + win][None, :, :]
                probs = decoder.predict_proba(window)[0]
                probs_history.append(probs)
                smooth_probs = np.mean(np.stack(probs_history), axis=0)
                decoded = int(np.argmax(smooth_probs))
                confidence = float(np.max(smooth_probs))

                stats.windows += 1
                if decoded == target_dir:
                    stats.correct_windows += 1

                velocity = np.zeros(2, dtype=np.float64)
                if confidence >= confidence_threshold:
                    velocity = sum(smooth_probs[c] * CLASS_DIRECTION[c] for c in range(4))
                cursor = np.clip(cursor + speed * velocity, -1.2, 1.2)
                status = f"conf={confidence:.2f}"
                _draw_scene(canvas, cursor, target_dir, decoded, smooth_probs, stats, status)

                if float(cursor @ CLASS_DIRECTION[target_dir]) >= 1.0:
                    stats.hits += 1
                    reached = True
                    _draw_scene(canvas, cursor, target_dir, decoded, smooth_probs, stats, "hit")
                    break

                time.sleep(step_seconds)

            if not reached:
                _draw_scene(canvas, cursor, target_dir, decoded, smooth_probs, stats, "timeout")
                time.sleep(0.35)

        hit_rate = stats.hits / stats.targets if stats.targets else 0.0
        window_acc = stats.correct_windows / stats.windows if stats.windows else 0.0
        print(
            f"Replay finished: targets hit {stats.hits}/{stats.targets} "
            f"({hit_rate:.0%}), window accuracy {window_acc:.0%}"
        )
        _draw_final_scene(canvas, stats)
        root.mainloop()
    except KeyboardInterrupt:
        root.destroy()
