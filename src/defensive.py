"""Defensive analytics: Voronoi space control, PPDA, and defensive pressure score.

Computes:
- Voronoi space control: pitch area owned by each player per freeze frame
- PPDA: Passes Allowed Per Defensive Action (pressing intensity)
- Defensive pressure score: composite per-match metric
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0


# ---------------------------------------------------------------------------
# 1. Voronoi space control
# ---------------------------------------------------------------------------

def compute_voronoi_areas(
    frame_df: pd.DataFrame,
    resolution: int = 120,
) -> pd.DataFrame:
    """Compute Voronoi-based pitch area controlled by each player in a freeze frame.

    Each pitch cell is assigned to the nearest player. Area is returned in
    square yards.

    Args:
        frame_df: Rows for a single event with columns x, y, teammate.
            Each row is one player.
        resolution: Grid resolution along the longer pitch axis.

    Returns:
        Copy of frame_df with added column voronoi_area_yards2.
    """
    df = frame_df.copy().reset_index(drop=True)
    players = df[["x", "y"]].dropna().values

    if len(players) == 0:
        df["voronoi_area_yards2"] = 0.0
        return df

    step = PITCH_LENGTH / resolution
    ry = int(PITCH_WIDTH / step)

    xs = np.linspace(0, PITCH_LENGTH, resolution)
    ys = np.linspace(0, PITCH_WIDTH, ry)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N_cells, 2)

    # For each cell, find the nearest player index
    diffs = grid[:, None, :] - players[None, :, :]  # (N_cells, N_players, 2)
    dists = np.linalg.norm(diffs, axis=2)            # (N_cells, N_players)
    nearest = np.argmin(dists, axis=1)               # (N_cells,)

    cell_area = step * step
    areas = np.bincount(nearest, minlength=len(players)) * cell_area

    # Map areas back to original df rows (only rows with valid positions)
    valid_idx = df[["x", "y"]].dropna().index.tolist()
    df["voronoi_area_yards2"] = 0.0
    for i, idx in enumerate(valid_idx):
        df.at[idx, "voronoi_area_yards2"] = areas[i]

    return df


def aggregate_voronoi(
    freeze_frames_df: pd.DataFrame,
    resolution: int = 120,
) -> pd.DataFrame:
    """Compute mean Voronoi area per teammate group across all freeze frames.

    Args:
        freeze_frames_df: Full match freeze frames (columns: id, x, y, teammate).
        resolution: Grid resolution for Voronoi computation.

    Returns:
        DataFrame with columns: teammate, mean_voronoi_area_yards2,
        total_voronoi_area_yards2, n_frames.
    """
    records = []
    for event_id, group in freeze_frames_df.groupby("id"):
        result = compute_voronoi_areas(group, resolution=resolution)
        for teammate_flag in [True, False]:
            subset = result[result["teammate"] == teammate_flag]
            if subset.empty:
                continue
            records.append(
                {
                    "event_id": event_id,
                    "teammate": teammate_flag,
                    "mean_voronoi_area": subset["voronoi_area_yards2"].mean(),
                    "total_voronoi_area": subset["voronoi_area_yards2"].sum(),
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["teammate", "mean_voronoi_area_yards2", "total_voronoi_area_yards2", "n_frames"]
        )

    df = pd.DataFrame(records)
    summary = (
        df.groupby("teammate")
        .agg(
            mean_voronoi_area_yards2=("mean_voronoi_area", "mean"),
            total_voronoi_area_yards2=("total_voronoi_area", "mean"),
            n_frames=("event_id", "count"),
        )
        .reset_index()
    )
    logger.info("Voronoi aggregation complete over %d frames", len(records) // 2)
    return summary


# ---------------------------------------------------------------------------
# 2. PPDA — Passes Allowed Per Defensive Action
# ---------------------------------------------------------------------------

DEFENSIVE_ACTION_TYPES = {"Pressure", "Tackle", "Interception", "Block", "Duel"}
PPDA_ZONE_X_MIN = 60.0  # Only count actions in opponent's half


def compute_ppda(events_df: pd.DataFrame) -> float:
    """Compute PPDA (Passes Allowed Per Defensive Action) for the home team.

    PPDA = (opponent passes in defensive zone) / (team defensive actions in zone)

    A lower PPDA means more aggressive pressing.

    Args:
        events_df: Match events DataFrame with columns: type, team, location,
            possession_team.

    Returns:
        PPDA scalar (or np.inf if no defensive actions recorded).
    """
    def _type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t) if pd.notna(t) else ""

    def _team_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t) if pd.notna(t) else ""

    df = events_df.copy()
    df["_type"] = df["type"].apply(_type_name)

    teams = df["team"].apply(_team_name).unique()
    if len(teams) < 2:
        return np.nan
    home_team = teams[0]

    def _x(loc):
        if isinstance(loc, (list, np.ndarray)) and len(loc) >= 1:
            return float(loc[0])
        return np.nan

    df["_x"] = df["location"].apply(_x)

    # Opponent passes in home team's defensive zone (x < PPDA_ZONE_X_MIN for home)
    opp_passes = df[
        (df["team"].apply(_team_name) != home_team)
        & (df["_type"] == "Pass")
        & (df["_x"] < PPDA_ZONE_X_MIN)
    ]

    # Home team defensive actions in same zone
    home_def = df[
        (df["team"].apply(_team_name) == home_team)
        & (df["_type"].isin(DEFENSIVE_ACTION_TYPES))
        & (df["_x"] < PPDA_ZONE_X_MIN)
    ]

    if len(home_def) == 0:
        return np.inf

    ppda = len(opp_passes) / len(home_def)
    logger.info(
        "PPDA: %.2f  (opp_passes=%d  def_actions=%d)", ppda, len(opp_passes), len(home_def)
    )
    return ppda


# ---------------------------------------------------------------------------
# 3. Defensive pressure score
# ---------------------------------------------------------------------------

def compute_defensive_pressure_score(
    events_df: pd.DataFrame,
    freeze_frames_df: pd.DataFrame,
    ppda_weight: float = 0.4,
    voronoi_weight: float = 0.3,
    action_density_weight: float = 0.3,
) -> dict:
    """Composite defensive pressure score for a match (0–100 scale).

    Combines:
    - PPDA-based component (lower PPDA → higher pressing intensity)
    - Voronoi-based component (less space ceded to opponents → better defense)
    - Defensive action density (actions per minute in opponent half)

    Args:
        events_df: Match events DataFrame.
        freeze_frames_df: Freeze frames DataFrame.
        ppda_weight: Weight for PPDA component.
        voronoi_weight: Weight for Voronoi component.
        action_density_weight: Weight for action density component.

    Returns:
        Dict with keys: ppda, voronoi_opponent_area, action_density,
        defensive_pressure_score.
    """
    # PPDA component (invert and normalize; cap at 20 for normalization)
    ppda = compute_ppda(events_df)
    ppda_score = 0.0
    if np.isfinite(ppda) and ppda > 0:
        ppda_score = max(0.0, min(100.0, (1 - ppda / 20.0) * 100))

    # Voronoi component (opponent space — lower is better for defense)
    voronoi_score = 50.0
    if not freeze_frames_df.empty:
        summary = aggregate_voronoi(freeze_frames_df, resolution=60)
        opp_row = summary[summary["teammate"] == False]
        if not opp_row.empty:
            opp_area = float(opp_row["total_voronoi_area_yards2"].iloc[0])
            total_area = PITCH_LENGTH * PITCH_WIDTH
            voronoi_score = max(0.0, min(100.0, (1 - opp_area / total_area) * 100))

    # Action density component
    def _type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t) if pd.notna(t) else ""

    def _x(loc):
        if isinstance(loc, (list, np.ndarray)) and len(loc) >= 1:
            return float(loc[0])
        return np.nan

    df = events_df.copy()
    df["_type"] = df["type"].apply(_type_name)
    df["_x"] = df["location"].apply(_x)

    def_actions_opp_half = df[
        df["_type"].isin(DEFENSIVE_ACTION_TYPES) & (df["_x"] >= PPDA_ZONE_X_MIN)
    ]
    total_events = max(len(events_df), 1)
    action_density_score = min(100.0, (len(def_actions_opp_half) / total_events) * 1000)

    composite = (
        ppda_weight * ppda_score
        + voronoi_weight * voronoi_score
        + action_density_weight * action_density_score
    )

    result = {
        "ppda": round(ppda, 3) if np.isfinite(ppda) else None,
        "ppda_score": round(ppda_score, 1),
        "voronoi_score": round(voronoi_score, 1),
        "action_density_score": round(action_density_score, 1),
        "defensive_pressure_score": round(composite, 1),
    }
    logger.info("Defensive pressure score: %.1f", composite)
    return result


# ---------------------------------------------------------------------------
# 4. Batch analysis across matches
# ---------------------------------------------------------------------------

def analyze_match_defense(match_id: int) -> dict:
    """Run full defensive analysis for a single match.

    Loads data from cache and returns a metrics dict.

    Args:
        match_id: StatsBomb match ID (must be already ingested).

    Returns:
        Dict with match_id and all defensive metrics.
    """
    from src.ingestion import load_events, load_freeze_frames

    events = load_events(match_id)
    frames = load_freeze_frames(match_id)
    metrics = compute_defensive_pressure_score(events, frames)
    metrics["match_id"] = match_id
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.ingestion import load_events, load_freeze_frames

    MATCH_ID = 3857256
    events = load_events(MATCH_ID)
    frames = load_freeze_frames(MATCH_ID)

    print("\n--- PPDA ---")
    ppda = compute_ppda(events)
    print(f"PPDA: {ppda:.2f}")

    print("\n--- Voronoi Space Control ---")
    voronoi = aggregate_voronoi(frames, resolution=60)
    print(voronoi.to_string(index=False))

    print("\n--- Defensive Pressure Score ---")
    score = compute_defensive_pressure_score(events, frames)
    for k, v in score.items():
        print(f"  {k}: {v}")
