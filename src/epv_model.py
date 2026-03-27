"""Expected Possession Value (EPV) model using XGBoost.

Trains a binary classifier on possession feature matrices to predict the
probability that a possession leads to a shot within N events.
Produces pitch surface heatmaps of EPV across ball positions.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent.parent / "data" / "models"

FEATURE_COLS = [
    "ball_x",
    "ball_y",
    "distance_to_goal",
    "angle_to_goal",
    "pressure_index",
    "n_attackers_ahead",
    "n_defenders_between",
    "time_to_space",
]


def load_all_features(processed_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and concatenate all feature parquet files.

    Args:
        processed_dir: Directory containing features_*.parquet files.

    Returns:
        Combined DataFrame with all matches.
    """
    d = processed_dir or PROCESSED_DIR
    paths = sorted(d.glob("features_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature files found in {d}")
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded %d rows from %d feature files", len(df), len(paths))
    return df


def _get_label_col(df: pd.DataFrame) -> str:
    """Return the shot_within_N column name."""
    cols = [c for c in df.columns if c.startswith("shot_within_")]
    if not cols:
        raise ValueError("No shot_within_N column found in features DataFrame")
    return cols[0]


def train(
    df: Optional[pd.DataFrame] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Train XGBoost EPV classifier.

    Args:
        df: Feature DataFrame. If None, loads all processed parquet files.
        test_size: Fraction of data held out for evaluation.
        random_state: Random seed.

    Returns:
        Tuple of (trained model, evaluation metrics dict).
    """
    try:
        import xgboost as xgb
    except ImportError:
        raise ImportError("xgboost is required: pip install xgboost")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

    if df is None:
        df = load_all_features()

    label_col = _get_label_col(df)
    data = df[FEATURE_COLS + [label_col]].dropna()
    X = data[FEATURE_COLS].values
    y = data[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pos_weight = float((y_train == 0).sum()) / max((y_train == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "avg_precision": float(average_precision_score(y_test, y_prob)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "pos_rate": float(y.mean()),
    }
    logger.info(
        "EPV model: ROC-AUC=%.3f  Avg-Precision=%.3f  (train=%d  test=%d)",
        metrics["roc_auc"],
        metrics["avg_precision"],
        metrics["n_train"],
        metrics["n_test"],
    )
    return model, metrics


def save_model(model, path: Optional[Path] = None) -> Path:
    """Persist trained model with joblib.

    Args:
        model: Trained XGBoost classifier.
        path: Output path. Defaults to data/models/epv_model.joblib.

    Returns:
        Path to saved file.
    """
    import joblib

    out = path or (MODELS_DIR / "epv_model.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    logger.info("Saved EPV model to %s", out)
    return out


def load_model(path: Optional[Path] = None):
    """Load a previously saved EPV model.

    Args:
        path: Path to joblib file. Defaults to data/models/epv_model.joblib.

    Returns:
        Loaded model.
    """
    import joblib

    p = path or (MODELS_DIR / "epv_model.joblib")
    if not p.exists():
        raise FileNotFoundError(f"No saved model at {p}. Run train() first.")
    return joblib.load(p)


def predict_epv(model, features_df: pd.DataFrame) -> pd.Series:
    """Return EPV probability for each row in features_df.

    Args:
        model: Trained XGBoost classifier.
        features_df: DataFrame with FEATURE_COLS columns.

    Returns:
        Series of EPV probabilities (0–1).
    """
    X = features_df[FEATURE_COLS].fillna(0).values
    return pd.Series(model.predict_proba(X)[:, 1], index=features_df.index)


def build_pitch_heatmap(
    model,
    resolution: int = 50,
    fixed_features: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate an EPV surface over the pitch by varying ball_x and ball_y.

    All other features are held at their median (or overridden via fixed_features).

    Args:
        model: Trained XGBoost classifier.
        resolution: Grid resolution (resolution x resolution cells).
        fixed_features: Optional dict overriding median values for other features.

    Returns:
        Tuple of (x_grid, y_grid, epv_grid) numpy arrays for plotting with
        matplotlib pcolormesh or contourf.
    """
    xs = np.linspace(0, 120, resolution)
    ys = np.linspace(0, 80, resolution)
    xx, yy = np.meshgrid(xs, ys)

    defaults = {
        "distance_to_goal": 40.0,
        "angle_to_goal": 10.0,
        "pressure_index": 0.5,
        "n_attackers_ahead": 3,
        "n_defenders_between": 4,
        "time_to_space": 2.0,
    }
    if fixed_features:
        defaults.update(fixed_features)

    n = resolution * resolution
    grid_df = pd.DataFrame(
        {
            "ball_x": xx.ravel(),
            "ball_y": yy.ravel(),
            **{k: np.full(n, v) for k, v in defaults.items()},
        }
    )[FEATURE_COLS]

    epv = model.predict_proba(grid_df.values)[:, 1].reshape(resolution, resolution)
    return xx, yy, epv


def plot_heatmap(
    model,
    resolution: int = 50,
    title: str = "EPV Pitch Heatmap",
    save_path: Optional[Path] = None,
):
    """Plot EPV surface heatmap over a pitch outline.

    Args:
        model: Trained XGBoost classifier.
        resolution: Grid resolution.
        title: Plot title.
        save_path: If provided, save figure to this path instead of showing.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    xx, yy, epv = build_pitch_heatmap(model, resolution)

    fig, ax = plt.subplots(figsize=(12, 8))
    pcm = ax.pcolormesh(xx, yy, epv, cmap="RdYlGn", vmin=0, vmax=epv.max())
    plt.colorbar(pcm, ax=ax, label="EPV (P(shot within N events))")

    # Pitch outline
    ax.add_patch(patches.Rectangle((0, 0), 120, 80, fill=False, edgecolor="white", lw=2))
    ax.plot([60, 60], [0, 80], color="white", lw=1)
    ax.add_patch(patches.Circle((60, 40), 10, fill=False, edgecolor="white", lw=1))
    # Penalty areas
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, fill=False, edgecolor="white", lw=1))
    ax.add_patch(patches.Rectangle((0, 18), 18, 44, fill=False, edgecolor="white", lw=1))
    # Goals
    ax.add_patch(patches.Rectangle((120, 36), 2, 8, fill=False, edgecolor="yellow", lw=2))
    ax.add_patch(patches.Rectangle((-2, 36), 2, 8, fill=False, edgecolor="yellow", lw=2))

    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14)
    ax.axis("off")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved heatmap to %s", save_path)
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model, metrics = train()
    print("\nEPV Model Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_model(model)

    heatmap_path = MODELS_DIR / "epv_heatmap.png"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_heatmap(model, save_path=heatmap_path)
    print(f"\nHeatmap saved to {heatmap_path}")
