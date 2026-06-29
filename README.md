# EEG Motor Imagery Classification

This project explores motor-imagery classification from EEG signals using the BCI Competition IV Dataset 2a. The goal is to classify which movement a participant imagines: left hand, right hand, both feet, or tongue.

## Project Goals

- Load and inspect EEG recordings in GDF format.
- Preprocess EEG signals for motor-imagery classification.
- Compare classical machine-learning models such as Logistic Regression, SVM, and Random Forest.
- Compare classical models with a CNN-based approach.
- Analyze which preprocessing and feature-extraction choices improve performance.

## Dataset

The project uses BCI Competition IV Dataset 2a.

- 9 subjects: `A01` to `A09`
- Training and evaluation sessions for each subject
- 22 EEG channels and 3 EOG channels
- 250 Hz sampling frequency
- 4 motor-imagery classes

Download page: https://www.bbci.de/competition/iv/#dataset2a

Place the GDF files in:

```text
BCICIV_2a_gdf/
```

The raw `.gdf` files are intentionally ignored by Git because they are large data files. Keep them locally, or document a separate download step for collaborators.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Inspect the default file, `BCICIV_2a_gdf/A01E.gdf`:

```powershell
python pipeline.py
```

Inspect a specific file:

```powershell
python pipeline.py BCICIV_2a_gdf/A01T.gdf
```

Open the interactive MNE plot:

```powershell
python pipeline.py BCICIV_2a_gdf/A01T.gdf --preload --plot
```

## Planned Workflow

1. Load GDF files with MNE.
2. Extract events and labels from annotations.
3. Apply EEG preprocessing, including filtering and epoching.
4. Extract features such as band power or CSP features.
5. Train and evaluate classical ML models.
6. Train and evaluate a CNN baseline.
7. Compare results across models and subjects.

## Repository Contents

```text
.
├── pipeline.py
├── requirements.txt
├── README.md
├── desc_2a.pdf
└── BCICIV_2a_gdf/        # local dataset files, ignored by Git
```
