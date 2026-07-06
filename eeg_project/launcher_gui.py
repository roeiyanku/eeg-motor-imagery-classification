"""Small Tkinter launcher for the project demo workflows."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from tkinter import BOTH, LEFT, RIGHT, StringVar, Tk, ttk

from .decoders import DECODER_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "pipeline.py"


class LauncherApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("EEG Motor Imagery BCI")
        self.root.geometry("720x480")
        self.root.minsize(640, 430)

        self.subject = StringVar(value="A03")
        self.model = StringVar(value="riemann_fbcsp_vote")
        self.targets = StringVar(value="12")
        self.calibration_model = StringVar(value="")
        self.status = StringVar(value="Ready")

        self._build()

    def _build(self) -> None:
        root_frame = ttk.Frame(self.root, padding=18)
        root_frame.pack(fill=BOTH, expand=True)

        title = ttk.Label(root_frame, text="EEG Motor Imagery BCI", font=("Segoe UI", 22, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(root_frame, text="Launch replay, example, calibration, and live workflows.")
        subtitle.pack(anchor="w", pady=(2, 16))

        settings = ttk.LabelFrame(root_frame, text="Settings", padding=12)
        settings.pack(fill="x", pady=(0, 14))
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Subject").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(settings, textvariable=self.subject, width=12).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(settings, text="Model").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=4)
        model_box = ttk.Combobox(settings, textvariable=self.model, values=list(DECODER_NAMES), state="readonly", width=28)
        model_box.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(settings, text="Targets").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(settings, textvariable=self.targets, width=12).grid(row=2, column=1, sticky="w", pady=4)

        ttk.Label(settings, text="Calibration model").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=4)
        ttk.Entry(settings, textvariable=self.calibration_model).grid(row=3, column=1, sticky="ew", pady=4)

        actions = ttk.LabelFrame(root_frame, text="Actions", padding=12)
        actions.pack(fill="x", pady=(0, 14))
        for i in range(2):
            actions.columnconfigure(i, weight=1)

        ttk.Button(actions, text="Show Live Replay", command=self.show_live_replay).grid(row=0, column=0, sticky="ew", padx=(0, 8), pady=5)
        ttk.Button(actions, text="Save GIF Example", command=self.save_gif_example).grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=5)
        ttk.Button(actions, text="Start Calibration GUI", command=self.start_calibration_gui).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=5)
        ttk.Button(actions, text="Start Live EEG Demo", command=self.start_live_demo).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=5)

        footer = ttk.Frame(root_frame)
        footer.pack(fill=BOTH, expand=True)
        ttk.Label(footer, textvariable=self.status).pack(side=LEFT, anchor="s")
        ttk.Button(footer, text="Close", command=self.root.destroy).pack(side=RIGHT, anchor="s")

    def _subject(self) -> str:
        value = self.subject.get().strip().upper()
        return value or "A03"

    def _targets(self) -> str:
        try:
            return str(max(1, int(self.targets.get())))
        except ValueError:
            self.targets.set("12")
            return "12"

    def _run(self, args: list[str], label: str) -> None:
        cmd = [sys.executable, str(PIPELINE), *args]
        subprocess.Popen(cmd, cwd=PROJECT_ROOT)
        self.status.set(f"Started: {label}")

    def show_live_replay(self) -> None:
        self._run(
            [
                "replay-live",
                "--subject",
                self._subject(),
                "--model",
                self.model.get(),
                "--targets",
                self._targets(),
            ],
            "Dataset 2a live replay",
        )

    def save_gif_example(self) -> None:
        self._run(
            [
                "demo",
                "--subject",
                self._subject(),
                "--model",
                self.model.get(),
                "--targets",
                self._targets(),
            ],
            "offline GIF example",
        )

    def start_calibration_gui(self) -> None:
        self._run(["calibrate-gui"], "personal calibration GUI")

    def start_live_demo(self) -> None:
        model_path = self.calibration_model.get().strip()
        args = ["live-demo", "--subject", self._subject(), "--model", self.model.get()]
        if model_path:
            args.extend(["--calibration-model", model_path])
        self._run(args, "live EEG demo")

    def run(self) -> None:
        self.root.mainloop()


def run_launcher_gui() -> None:
    LauncherApp().run()
