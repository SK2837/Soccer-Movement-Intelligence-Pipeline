# Soccer Movement Intelligence Pipeline

End-to-end player tracking analytics system built on StatsBomb open data. Demonstrates EPV modeling, off-ball movement analytics, Voronoi space control, and MediaPipe skeletal pose extraction — deployed as an interactive Streamlit dashboard.

## Modules

| Module | Description |
|--------|-------------|
| **Ingestion** | Pull StatsBomb events + 360° freeze frames via `statsbombpy` |
| **Features** | Velocity vectors, pressure index, off-ball run distance, time-to-space |
| **EPV Model** | XGBoost classifier for Expected Possession Value; pitch surface heatmap |
| **Pose Analysis** | MediaPipe 33-keypoint extraction, sprint asymmetry index, knee flexion angles |
| **Defensive** | Voronoi space control, PPDA, defensive pressure score |
| **Dashboard** | 3-tab Streamlit app: match explorer, pose analyzer, player comparison |

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Pull StatsBomb data and build features
python src/ingestion.py

# Train EPV model
python src/epv_model.py

# Launch dashboard
streamlit run app/streamlit_app.py
```

## Project Structure

```
├── data/
│   ├── raw/          # StatsBomb parquet files
│   └── processed/    # Feature-engineered outputs
├── notebooks/        # Exploratory analysis notebooks
├── src/              # Core pipeline modules
├── app/              # Streamlit dashboard
└── tests/            # Unit tests
```

## Key Metrics

- **EPV (Expected Possession Value)**: Probability a possession leads to a shot, conditioned on ball + player positions
- **Sprint Asymmetry Index**: `|left_stride - right_stride| / avg_stride` — injury risk proxy
- **Knee Flexion Angle**: Hip-knee-ankle angle at ground contact
- **PPDA**: Passes Allowed Per Defensive Action — pressing intensity metric
- **Voronoi Space Control**: Pitch area controlled per player per frame

## Data Source

[StatsBomb Open Data](https://github.com/statsbomb/open-data) — free, no license required. Includes 360° tracking for NWSL, FIFA World Cup, La Liga, and more.

## Live Demo

*Deploy to [Streamlit Community Cloud](https://streamlit.io/cloud) and add link here.*
