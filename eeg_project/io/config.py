from __future__ import annotations

from pathlib import Path

DATA_DIR = Path("data") / "BCICIV_2a_gdf"
PREPARED_DIR = Path("processed")
RESULTS_DIR = Path("results")
PREPARED_FILE = PREPARED_DIR / "dataset_2a_epochs.npz"

ANNOTATION_TO_CLASS = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}

MNE_EVENT_ID = {
    "left_hand": 1,
    "right_hand": 2,
    "feet": 3,
    "tongue": 4,
}

CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]
CLASS_TO_LABEL = {name: i for i, name in enumerate(CLASS_NAMES)}

EVENT_DESCRIPTIONS = {
    "276": "eyes_open",
    "277": "eyes_closed",
    "768": "start_of_trial",
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
    "783": "unknown_cue",
    "1023": "rejected_trial",
    "1072": "eye_movements",
    "32766": "start_of_new_run",
}

