from __future__ import annotations

import argparse
from pathlib import Path

import mne


DEFAULT_GDF = Path("BCICIV_2a_gdf") / "A01E.gdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a BCI Competition IV Dataset 2a GDF file."
    )
    parser.add_argument(
        "gdf_path",
        nargs="?",
        type=Path,
        default=DEFAULT_GDF,
        help=f"Path to a .gdf file (default: {DEFAULT_GDF})",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load the signal data into memory instead of reading metadata lazily.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Open MNE's interactive raw-signal plot window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.gdf_path.exists():
        raise FileNotFoundError(f"GDF file not found: {args.gdf_path}")

    raw = mne.io.read_raw_gdf(args.gdf_path, preload=args.preload, verbose="ERROR")

    print(raw.info)
    print(f"\nLoaded: {args.gdf_path}")
    print(f"Channels: {len(raw.ch_names)}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Samples: {raw.n_times}")
    print(f"Duration: {raw.times[-1]:.2f} seconds")

    if args.plot:
        raw.plot()


if __name__ == "__main__":
    main()
