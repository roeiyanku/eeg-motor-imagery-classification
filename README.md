# EEG Motor Imagery Classification

This project classifies motor-imagery EEG trials from **BCI Competition IV Dataset 2a**. The four classes are imagined movement of the left hand, right hand, both feet, and tongue.

The project has two goals:

1. Build a reliable benchmark pipeline for Dataset 2a.
2. Use the best decoder as a foundation for a future real-time personal BCI demo.

## Dataset

Dataset 2a contains 9 subjects. Each subject has:

- a labeled calibration recording: `A01T.gdf` ... `A09T.gdf`
- an evaluation recording: `A01E.gdf` ... `A09E.gdf`
- official evaluation labels stored locally in `data/true_labels/`
- 22 EEG channels and 3 EOG channels

Important event codes:

```text
769 = left hand
770 = right hand
771 = both feet
772 = tongue
783 = unknown cue in evaluation files
```

Raw `.gdf` files should be placed in:

```text
data/BCICIV_2a_gdf/
```

The raw EEG files are ignored by Git because they are large dataset files.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In this workspace, the project was run with:

```powershell
.\.venv\Scripts\python.exe
```

## Main Commands

Inspect a raw GDF file:

```powershell
.\.venv\Scripts\python.exe pipeline.py inspect data/BCICIV_2a_gdf/A01T.gdf
```

Prepare the labeled training epochs:

```powershell
.\.venv\Scripts\python.exe pipeline.py prepare
```

Run the official train-T/test-E benchmark:

```powershell
.\.venv\Scripts\python.exe pipeline.py benchmark --models riemann fbcsp riemann_fbcsp_vote
```

Run an offline cursor-control demo:

```powershell
.\.venv\Scripts\python.exe pipeline.py demo --subject A03 --model riemann --targets 12
```

Open the launcher GUI for the main demo workflows:

```powershell
.\.venv\Scripts\python.exe pipeline.py gui
```

Run an interactive live-style replay GUI without EEG hardware:

```powershell
.\.venv\Scripts\python.exe pipeline.py replay-live --subject A03 --model riemann_fbcsp_vote --targets 12
```

Run a live LSL demo with a real EEG stream:

```powershell
.\.venv\Scripts\python.exe pipeline.py live-demo --subject A03 --model riemann
```

Record a personal calibration dataset from an LSL EEG stream:

```powershell
.\.venv\Scripts\python.exe pipeline.py calibrate-record --trials-per-class 20 --channels 0 1 2 3 4 5 6 7
```

Or use the arrow/cue GUI, which is closer to the original motor-imagery protocol:

```powershell
.\.venv\Scripts\python.exe pipeline.py calibrate-gui --trials-per-class 20 --channels 0 1 2 3 4 5 6 7
```

Train a personal decoder from that calibration dataset:

```powershell
.\.venv\Scripts\python.exe pipeline.py calibrate-train --model riemann --output models/personal_riemann.joblib
```

Run the live demo with your personal decoder:

```powershell
.\.venv\Scripts\python.exe pipeline.py live-demo --calibration-model models/personal_riemann.joblib
```

For a steadier cursor, use smoothing and a confidence threshold:

```powershell
.\.venv\Scripts\python.exe pipeline.py live-demo --calibration-model models/personal_riemann.joblib --smoothing-windows 8 --confidence-threshold 0.45
```

## Preprocessing

The default EEG preprocessing pipeline:

- loads `.gdf` files with MNE
- marks EOG channels
- keeps the 22 EEG channels
- excludes the 3 EOG channels from model input
- filters EEG to 8-30 Hz for motor-imagery mu/beta rhythms
- resamples to 125 Hz
- extracts epochs from 0.5s to 4.0s after cue onset
- maps labels to `left_hand`, `right_hand`, `feet`, and `tongue`

For benchmark experiments, the raw recordings are first loaded as broadband 4-40 Hz epochs, and each decoder applies its own internal filtering.

## Implemented Models

Classical models:

- CSP + Logistic Regression
- CSP + SVM
- CSP + Random Forest
- CSP + LDA
- FBCSP + mutual-information feature selection + shrinkage LDA
- Riemannian filter-bank tangent-space decoder
- Riemannian variants with wider filter banks and logistic regression
- `riemann_fbcsp_vote`: soft-vote ensemble of Riemannian + FBCSP

Neural models:

- compact EEGNet-style CNN
- raw short ResNet
- ShallowConvNet
- EEG-TCNet

## Benchmark Protocol

The main benchmark follows the BCI Competition IV 2a evaluation protocol:

```text
train on A0XT.gdf
test on A0XE.gdf
score using data/true_labels/A0XE.mat
average across 9 subjects
```

The main score is **Cohen's kappa**, because it corrects accuracy for chance agreement. For 4 balanced classes, random guessing is around 25% accuracy, so kappa is more informative than raw accuracy alone.

The 2008 competition winner, Ang et al.'s FBCSP method, reported approximately:

```text
mean kappa ~= 0.57
```

## Current Results

Main 9-subject benchmark:

```text
model             mean accuracy   mean kappa
riemann           0.712           0.616
fbcsp             0.675           0.566
shallow_convnet   0.627           0.503
```

Best result after adding a soft-vote ensemble:

```text
model                 mean accuracy   mean kappa
riemann_fbcsp_vote    0.735           0.647
```

This is the strongest result so far and is clearly above the 2008 FBCSP reference kappa of about 0.57.

We also tested naive cross-subject pooling: one `riemann_fbcsp_vote` model trained on all 9 subjects' `T` files and tested on each subject's `E` file. This performed much worse:

```text
model                         mean accuracy   mean kappa
pooled_riemann_fbcsp_vote      0.520           0.360
```

This supports the main live-BCI conclusion: EEG motor-imagery models need subject-specific calibration. More data from other people does not automatically help unless we use domain adaptation or personal fine-tuning.

Saved result tables:

```text
results/benchmark_all_subjects_riemann_fbcsp_vote.csv
results/benchmark_all_subjects_riemann_fbcsp_shallow_convnet.csv
results/benchmark_all_subjects_riemann_wide_lr.csv
results/benchmark_pooled_riemann_fbcsp_vote.csv
```

## What Worked

The strongest methods were classical EEG-specific methods:

- Riemannian tangent-space decoding
- FBCSP
- a soft-vote ensemble of both

These worked well because Dataset 2a is relatively small: each subject has 288 calibration trials, or about 72 trials per class. Classical spatial-filtering methods are data-efficient and encode useful EEG structure directly.

## What Worked Less Well

Deep learning models were useful experiments, but they did not beat the classical methods on average.

ShallowConvNet was the best neural baseline, but it was less stable across subjects. EEG-TCNet ran correctly but was not competitive in its first untuned version.

This is a meaningful result: for low-data motor-imagery EEG, a well-designed classical pipeline can outperform more complex neural networks.

## Challenges

Important challenges during the project:

- preserving a fair train-T/test-E benchmark protocol
- avoiding accidental data leakage
- dealing with subject-to-subject variability
- handling limited data for deep learning
- memory issues when running MNE/OpenBLAS over all subjects in one long process
- deciding between benchmark performance and future live-BCI usability

The memory issue was handled by running subject-level benchmarks in fresh Python processes and limiting BLAS thread counts:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
```

## Live BCI Direction

The project already includes live-demo infrastructure using Lab Streaming Layer (LSL). The future personal BCI workflow is:

1. collect personal calibration EEG with labeled cues
2. train a decoder on the same user and headset
3. add a `rest` / `no-control` class
4. use sliding windows for real-time prediction
5. smooth predictions over recent windows
6. drive a cursor or command interface

The repository now supports the first two practical live steps:

```text
gui              -> launcher with buttons for replay, example GIF, calibration, and live EEG
replay-live      -> interactive live-style GUI using held-out Dataset 2a EEG
calibrate-record -> saves processed/personal_calibration.npz
calibrate-gui    -> arrow/cue GUI for the same calibration dataset
calibrate-train  -> saves models/personal_riemann.joblib
live-demo --calibration-model models/personal_riemann.joblib
```

`gui` is the easiest entry point for a presentation or grading meeting. It opens a small control panel with "Show Live Replay", "Save GIF Example", "Start Calibration GUI", and "Start Live EEG Demo" buttons.

`replay-live` is useful before hardware is available: it trains on a subject's Dataset 2a calibration file, replays held-out evaluation trials as if they were a live EEG stream, decodes sliding windows, and moves a cursor in a GUI.

The personal calibration workflow supports an explicit `rest` class by default. In live cursor control, `rest` maps to zero velocity, so the model is no longer forced to always output a movement command. The live demo also supports probability smoothing and confidence thresholding, which helps prevent noisy low-confidence windows from moving the cursor.

For a real-time system, `riemann` is the simplest strong decoder. `riemann_fbcsp_vote` is the best benchmark model, but it is heavier because it runs two decoders.

## Report / Submission Notes

A detailed professor-facing project history is in:

```text
docs/PROJECT_HISTORY.md
```

It includes:

- dataset description
- research questions
- code summary
- algorithms tested
- benchmark results
- challenges and how they were handled
- discussion of what worked and what did not
- future live-BCI direction
