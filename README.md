# Soccer Movement Intelligence Pipeline

An end-to-end soccer analytics platform built on [StatsBomb open data](https://github.com/statsbomb/open-data). It combines event-based match analysis, machine learning for Expected Possession Value (EPV), Voronoi-based defensive space control, PPDA pressing metrics, and MediaPipe skeletal pose analysis — all visualized in an interactive Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Pipeline (Step by Step)](#running-the-pipeline-step-by-step)
  - [Step 1 — Ingest Data](#step-1--ingest-data)
  - [Step 2 — Extract Features](#step-2--extract-features)
  - [Step 3 — Train the EPV Model](#step-3--train-the-epv-model)
  - [Step 4 — Launch the Dashboard](#step-4--launch-the-dashboard)
- [Modules](#modules)
- [Key Metrics Explained](#key-metrics-explained)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Data Source](#data-source)

---

## Project Overview

| Module | What it does |
|--------|-------------|
| **Ingestion** | Pulls StatsBomb match events and 360° freeze frames via `statsbombpy` and stores them as Parquet files |
| **Features** | Engineers possession features: ball position, pressure index, distance/angle to goal, defender counts, time-to-space |
| **EPV Model** | Trains an XGBoost binary classifier to estimate the probability a possession leads to a shot within N events |
| **Defensive** | Computes Voronoi space control per player, PPDA pressing intensity, and a composite defensive pressure score |
| **Pose Analysis** | Uses MediaPipe Pose Landmarker to extract 33 skeletal keypoints from images/videos; computes sprint asymmetry index, knee flexion angles, and stride lengths |
| **Dashboard** | A 3-tab Streamlit app: Match Explorer (EPV heatmap + events), Pose Analyzer (biomechanics), Player Comparison (cross-match metrics) |

---

## Architecture

```
StatsBomb API
     │
     ▼
src/ingestion.py ──► data/raw/events_{match_id}.parquet
                 ──► data/raw/frames_{match_id}.parquet
     │
     ▼
src/features.py ──► data/processed/features_{match_id}.parquet
     │
     ▼
src/epv_model.py ──► data/models/epv_model.joblib
                 ──► data/models/epv_heatmap.png
     │
     ▼
src/defensive.py ── (runs on demand per match)
src/pose_analysis.py ── (runs on uploaded image/video)
     │
     ▼
app/streamlit_app.py ── Interactive Dashboard
```

---

## Prerequisites

- **Python 3.10 or 3.11** (MediaPipe requires < 3.12 as of 2025)
- **macOS / Linux** (Windows works but MediaPipe installation may vary)
- **Homebrew** (macOS only, for system dependencies)
- **~2 GB disk space** for StatsBomb data + model files

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Soccer-Intelligence-Pipeline.git
cd Soccer-Intelligence-Pipeline
```

### 2. Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (macOS only) Install OpenMP for XGBoost

XGBoost on macOS requires OpenMP via Homebrew:

```bash
brew install libomp
```

Without this, you will see:
```
XGBoostError: Library not loaded: @rpath/libomp.dylib
```

---

## Running the Pipeline (Step by Step)

### Step 1 — Ingest Data

Pull match events and 360° freeze frames from StatsBomb. The default is the **FIFA World Cup 2022** (all 64 matches).

```bash
python -m src.ingestion
```

This saves files to:
```
data/raw/events_{match_id}.parquet
data/raw/frames_{match_id}.parquet
```

> **To change the competition**, edit the bottom of `src/ingestion.py`:
> ```python
> COMPETITION_ID = 43   # FIFA World Cup 2022
> SEASON_ID     = 106
> ```
> Other available competitions (free): La Liga (11/90), NWSL (49/3), UEFA Euro 2020 (55/43).

---

### Step 2 — Extract Features

Build the possession feature matrix for each ingested match:

```bash
python -m src.features
```

Output files:
```
data/processed/features_{match_id}.parquet
```

Each file contains columns: `ball_x`, `ball_y`, `distance_to_goal`, `angle_to_goal`, `pressure_index`, `n_attackers_ahead`, `n_defenders_between`, `time_to_space`, `shot_within_5`.

---

### Step 3 — Train the EPV Model

Train an XGBoost classifier on all processed feature files:

```bash
python -m src.epv_model
```

This will:
- Load all `data/processed/features_*.parquet` files
- Split 80/20 train/test
- Train an XGBoost classifier with class imbalance handling
- Print ROC-AUC and Average Precision scores
- Save the model to `data/models/epv_model.joblib`
- Save a pitch EPV heatmap to `data/models/epv_heatmap.png`

Expected output (FIFA World Cup 2022, 64 matches):
```
EPV model: ROC-AUC=0.890  Avg-Precision=0.258  (train=163109  test=40778)
```

---

### Step 4 — Launch the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open your browser at `http://localhost:8501`.

#### Dashboard tabs:

| Tab | Features |
|-----|----------|
| **Match Explorer** | Select any ingested match → EPV pitch heatmap, top events table, defensive pressure score breakdown |
| **Pose Analyzer** | Upload a soccer player image → extract MediaPipe keypoints → view knee flexion angles, stride lengths, sprint asymmetry index |
| **Player Comparison** | Select multiple matches → compare defensive metrics side by side in a table |

---

## Modules

### `src/ingestion.py`

```python
from src.ingestion import ingest_competition

ingest_competition(competition_id=43, season_id=106)  # FIFA World Cup 2022
```

Key functions:
- `get_competitions()` — list all available StatsBomb competitions
- `get_matches(competition_id, season_id)` — list matches for a competition
- `ingest_match(match_id)` — ingest events + freeze frames for one match
- `ingest_competition(competition_id, season_id)` — ingest all matches

---

### `src/features.py`

```python
from src.features import extract_possession_features, save_features
from src.ingestion import load_events, load_freeze_frames

events = load_events(match_id)
frames = load_freeze_frames(match_id)
features = extract_possession_features(events, frames)
save_features(features, match_id)
```

---

### `src/epv_model.py`

```python
from src.epv_model import train, save_model, load_model, predict_epv, build_pitch_heatmap

model, metrics = train()           # trains on all processed parquet files
save_model(model)                  # saves to data/models/epv_model.joblib
model = load_model()               # load for inference

epv_probs = predict_epv(model, features_df)          # per-event EPV
xx, yy, epv_grid = build_pitch_heatmap(model)        # 2D pitch surface
```

---

### `src/defensive.py`

```python
from src.defensive import compute_ppda, compute_voronoi_areas, compute_defensive_pressure_score
from src.ingestion import load_events, load_freeze_frames

events = load_events(match_id)
frames = load_freeze_frames(match_id)

ppda  = compute_ppda(events)
score = compute_defensive_pressure_score(events, frames)
```

---

### `src/pose_analysis.py`

```python
from src.pose_analysis import (
    extract_keypoints_from_image,
    compute_knee_flexion_angle,
    compute_sprint_asymmetry_index,
    compute_pose_metrics_per_frame,
)

# Single image
kp_df = extract_keypoints_from_image("player.jpg")
left_angle  = compute_knee_flexion_angle(kp_df, side="left")
right_angle = compute_knee_flexion_angle(kp_df, side="right")

# Video sequence
from src.pose_analysis import extract_keypoints_from_video
kp_seq = extract_keypoints_from_video("sprint.mp4", sample_every_n_frames=5)
metrics = compute_pose_metrics_per_frame(kp_seq)
asym    = compute_sprint_asymmetry_index(kp_seq)
```

> The MediaPipe model (`pose_landmarker_full.task`) is downloaded automatically to `/tmp/` on first run (~25 MB).

---

## Key Metrics Explained

### EPV — Expected Possession Value
The probability (0–1) that the current possession leads to a shot within the next N events, given ball position, defensive pressure, and player positioning. Higher EPV zones on the heatmap represent more dangerous areas of the pitch.

### Sprint Asymmetry Index
```
asymmetry = |mean_left_stride - mean_right_stride| / avg_stride × 100 (%)
```
Values above ~10% are a common threshold for elevated injury risk in biomechanics research.

### Knee Flexion Angle
The hip–knee–ankle angle at ground contact (degrees). Measured per frame. Lower values indicate more bent knees; typical sprinting range is 140–170°.

### PPDA — Passes Allowed Per Defensive Action
```
PPDA = opponent_passes_in_defensive_zone / team_defensive_actions_in_zone
```
Lower PPDA = more aggressive pressing. Elite pressing teams typically score below 8.

### Voronoi Space Control
Each cell of a discretized pitch grid is assigned to the nearest player. The total cell area per player is their "controlled space" in square yards for that freeze frame.

### Defensive Pressure Score (0–100)
A weighted composite of:
- PPDA score (40%) — inverted so lower PPDA → higher score
- Voronoi score (30%) — less opponent space → higher score
- Defensive action density in opponent half (30%)

---

## Project Structure

```
Soccer-Intelligence-Pipeline/
├── app/
│   └── streamlit_app.py          # Streamlit dashboard (3 tabs)
├── data/
│   ├── raw/                      # StatsBomb Parquet files (gitignored)
│   │   ├── events_{id}.parquet
│   │   └── frames_{id}.parquet
│   ├── processed/                # Feature files (gitignored)
│   │   └── features_{id}.parquet
│   └── models/                   # Trained models (gitignored)
│       ├── epv_model.joblib
│       └── epv_heatmap.png
├── notebooks/                    # Exploratory Jupyter notebooks
├── src/
│   ├── __init__.py
│   ├── ingestion.py              # StatsBomb data pull
│   ├── features.py               # Feature engineering
│   ├── epv_model.py              # XGBoost EPV model
│   ├── defensive.py              # PPDA, Voronoi, pressure score
│   └── pose_analysis.py          # MediaPipe pose extraction
├── tests/                        # Unit tests
├── requirements.txt
├── README.md
└── CLAUDE.md                     # Developer guidelines
```

---

## Configuration

| Setting | Where to change | Default |
|---------|----------------|---------|
| Competition to ingest | `src/ingestion.py` → `COMPETITION_ID`, `SEASON_ID` | 43 / 106 (FIFA WC 2022) |
| Feature window (shot within N events) | `src/features.py` → `SHOT_WINDOW` | 5 |
| EPV model hyperparameters | `src/epv_model.py` → `xgb.XGBClassifier(...)` | n_estimators=300, max_depth=5 |
| Voronoi grid resolution | `src/defensive.py` → `resolution` param | 120 |
| Video frame sampling | `src/pose_analysis.py` → `sample_every_n_frames` | 5 |
| Pitch dimensions | `src/defensive.py` → `PITCH_LENGTH`, `PITCH_WIDTH` | 120 × 80 yards |

---

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=term-missing

# Run a specific module's tests
pytest tests/test_features.py

# Lint check
flake8 src/ app/
black src/ app/ --check
```

---

## Troubleshooting

### `XGBoostError: Library not loaded: @rpath/libomp.dylib`
**macOS only.** Install OpenMP:
```bash
brew install libomp
```

### `AttributeError: module 'mediapipe' has no attribute 'solutions'`
You are using MediaPipe 0.10+ which removed the legacy API. This project already uses the new Tasks API. Ensure you installed the correct version:
```bash
pip install "mediapipe>=0.10.0"
```

### `FileNotFoundError: No feature files found in data/processed`
You need to run ingestion and feature extraction before training:
```bash
python -m src.ingestion
python -m src.features
```

### `No pose detected` in Pose Analyzer
- Use a clear, well-lit full-body image of a player
- The player should be visible from at least hip to feet
- JPEG or PNG formats work best
- Free images: [Unsplash — soccer](https://unsplash.com/s/photos/soccer-player), [Pexels — football](https://www.pexels.com/search/football%20player/)

### StatsBomb rate limiting
If ingesting many matches, `statsbombpy` may be rate-limited. The ingestion script includes automatic retry logic. Run it again if it stops mid-way — already-ingested matches are skipped.

---

## Data Source

All match data comes from the **[StatsBomb Open Data](https://github.com/statsbomb/open-data)** repository — free to use for research and educational purposes under the StatsBomb Open Data License.

Available competitions (free tier):

| Competition | ID | Season ID | Matches |
|-------------|-----|-----------|---------|
| FIFA World Cup 2022 | 43 | 106 | 64 |
| La Liga 2015/16 | 11 | 26 | 38 |
| UEFA Euro 2020 | 55 | 43 | 51 |
| NWSL 2018 | 49 | 3 | 92 |
| FA Women's Super League | 37 | 42 | varies |

---

## License

This project is for educational and research use. Match data is provided by StatsBomb under their [Open Data License](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
