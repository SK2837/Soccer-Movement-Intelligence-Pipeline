"""Feature engineering from StatsBomb events and 360° freeze frames.

Computes:
- Player velocity vectors from sequential freeze frame positions
- Defensive pressure index per frame
- Off-ball run distance per possession sequence
- Time-to-space metric
- Full possession feature matrix for EPV model input
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# StatsBomb pitch dimensions (yards)
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = np.array([120.0, 40.0])
GOAL_POST_LEFT = np.array([120.0, 36.0])
GOAL_POST_RIGHT = np.array([120.0, 44.0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_xy(location) -> tuple[Optional[float], Optional[float]]:
    """Extract x, y from a StatsBomb location (list or numpy array)."""
    if isinstance(location, (list, np.ndarray)) and len(location) >= 2:
        return float(location[0]), float(location[1])
    return None, None


def _add_ball_xy(events_df: pd.DataFrame) -> pd.DataFrame:
    """Add ball_x and ball_y columns from the location column."""
    df = events_df.copy()
    if "location" in df.columns:
        df["ball_x"] = df["location"].apply(lambda l: _extract_xy(l)[0])
        df["ball_y"] = df["location"].apply(lambda l: _extract_xy(l)[1])
    else:
        df["ball_x"] = np.nan
        df["ball_y"] = np.nan
    return df


# ---------------------------------------------------------------------------
# 1. Velocity vectors
# ---------------------------------------------------------------------------

def compute_velocity_vectors(freeze_frames_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate velocity vectors for each player across consecutive freeze frames.

    StatsBomb freeze frames are per-event snapshots, not continuous tracking.
    Velocity is approximated as the positional delta between consecutive events
    that share a player slot (matched by teammate flag + position order).

    Args:
        freeze_frames_df: DataFrame with columns id (event_id), x, y, teammate.

    Returns:
        Copy of freeze_frames_df with added columns: vx, vy, speed.
    """
    df = freeze_frames_df.copy()
    df["vx"] = 0.0
    df["vy"] = 0.0
    df["speed"] = 0.0

    if "id" not in df.columns or df.empty:
        return df

    event_ids = df["id"].unique()
    if len(event_ids) < 2:
        return df

    for i in range(1, len(event_ids)):
        prev_id = event_ids[i - 1]
        curr_id = event_ids[i]

        for teammate_flag in [True, False]:
            prev_pts = (
                df[(df["id"] == prev_id) & (df["teammate"] == teammate_flag)][["x", "y"]]
                .values
            )
            curr_pts = (
                df[(df["id"] == curr_id) & (df["teammate"] == teammate_flag)][["x", "y"]]
                .values
            )
            if len(prev_pts) == 0 or len(curr_pts) == 0:
                continue

            idxs = df.index[
                (df["id"] == curr_id) & (df["teammate"] == teammate_flag)
            ].tolist()

            n = min(len(prev_pts), len(curr_pts), len(idxs))
            for j in range(n):
                vx = float(curr_pts[j, 0] - prev_pts[j, 0])
                vy = float(curr_pts[j, 1] - prev_pts[j, 1])
                df.at[idxs[j], "vx"] = vx
                df.at[idxs[j], "vy"] = vy
                df.at[idxs[j], "speed"] = np.hypot(vx, vy)

    logger.info("Computed velocity vectors for %d freeze frame rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Defensive pressure index
# ---------------------------------------------------------------------------

def compute_pressure_index(
    frame_df: pd.DataFrame,
    ball_x: float,
    ball_y: float,
    radius: float = 10.0,
) -> float:
    """Compute defensive pressure on the ball carrier for one freeze frame.

    Pressure = sum(1 / distance) for all defenders within `radius` yards of
    the ball. Higher value means more pressure.

    Args:
        frame_df: Rows for a single event (columns: x, y, teammate).
        ball_x: Ball x coordinate.
        ball_y: Ball y coordinate.
        radius: Only defenders within this distance contribute (yards).

    Returns:
        Scalar pressure index >= 0.
    """
    defenders = frame_df[frame_df["teammate"] == False][["x", "y"]].dropna()
    if defenders.empty:
        return 0.0

    ball = np.array([ball_x, ball_y])
    dists = np.linalg.norm(defenders.values - ball, axis=1)
    nearby = dists[dists < radius]
    if len(nearby) == 0:
        return 0.0

    return float(np.sum(1.0 / np.maximum(nearby, 0.1)))


def add_pressure_index(
    events_df: pd.DataFrame,
    freeze_frames_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add a pressure_index column to events_df for every event with a freeze frame.

    Args:
        events_df: Match events (must have id, location columns).
        freeze_frames_df: Freeze frames (must have id, x, y, teammate columns).

    Returns:
        events_df copy with a new pressure_index column (NaN where no frame exists).
    """
    df = _add_ball_xy(events_df)
    frame_lookup = {eid: grp for eid, grp in freeze_frames_df.groupby("id")}

    def _pressure(row):
        frame = frame_lookup.get(row["id"])
        if frame is None or pd.isna(row.get("ball_x")) or pd.isna(row.get("ball_y")):
            return np.nan
        return compute_pressure_index(frame, row["ball_x"], row["ball_y"])

    df["pressure_index"] = df.apply(_pressure, axis=1)
    return df


# ---------------------------------------------------------------------------
# 3. Spatial metrics
# ---------------------------------------------------------------------------

def compute_distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance from (x, y) to the center of the goal (yards).

    Args:
        x: Pitch x coordinate.
        y: Pitch y coordinate.

    Returns:
        Distance in yards.
    """
    return float(np.linalg.norm(np.array([x, y]) - GOAL_CENTER))


def compute_angle_to_goal(x: float, y: float) -> float:
    """Angle subtended by the goal mouth from position (x, y) in degrees.

    Uses 8-yard-wide goal posts at (120, 36) and (120, 44).
    A larger angle = better shooting position.

    Args:
        x: Pitch x coordinate.
        y: Pitch y coordinate.

    Returns:
        Angle in degrees.
    """
    pos = np.array([x, y])
    vec_l = GOAL_POST_LEFT - pos
    vec_r = GOAL_POST_RIGHT - pos
    cos_a = np.dot(vec_l, vec_r) / (
        np.linalg.norm(vec_l) * np.linalg.norm(vec_r) + 1e-9
    )
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def count_players_ahead(
    frame_df: pd.DataFrame,
    ball_x: float,
    attack_direction: str = "right",
) -> tuple[int, int]:
    """Count attacking players ahead of the ball and defenders between ball and goal.

    Args:
        frame_df: Freeze frame rows for one event (columns: x, y, teammate).
        ball_x: Ball x coordinate.
        attack_direction: "right" if team attacks toward x=120, else "left".

    Returns:
        (n_attackers_ahead, n_defenders_between_ball_and_goal).
    """
    attackers = frame_df[frame_df["teammate"] == True]
    defenders = frame_df[frame_df["teammate"] == False]

    if attack_direction == "right":
        n_ahead = int((attackers["x"] > ball_x).sum())
        n_def = int(((defenders["x"] > ball_x) & (defenders["x"] < PITCH_LENGTH)).sum())
    else:
        n_ahead = int((attackers["x"] < ball_x).sum())
        n_def = int(((defenders["x"] < ball_x) & (defenders["x"] > 0)).sum())

    return n_ahead, n_def


# ---------------------------------------------------------------------------
# 4. Off-ball run distance
# ---------------------------------------------------------------------------

def compute_off_ball_run_distance(
    events_df: pd.DataFrame,
    freeze_frames_df: pd.DataFrame,
) -> pd.DataFrame:
    """Cumulative off-ball run distance per player slot per possession.

    Sums positional deltas for non-ball-carrier teammates across consecutive
    freeze frames within each possession sequence.

    Args:
        events_df: Match events (must have id, possession columns).
        freeze_frames_df: Freeze frames (must have id, x, y, teammate columns).

    Returns:
        DataFrame with columns: possession_id, player_slot, teammate,
        run_distance_yards.
    """
    if freeze_frames_df.empty or events_df.empty:
        return pd.DataFrame(
            columns=["possession_id", "player_slot", "teammate", "run_distance_yards"]
        )

    ff = freeze_frames_df.copy()

    if "possession" in events_df.columns and "id" in events_df.columns:
        poss_map = events_df.set_index("id")["possession"].to_dict()
        ff["possession"] = ff["id"].map(poss_map)
    else:
        ff["possession"] = 0

    records = []
    for poss, group in ff.groupby("possession"):
        event_ids = group["id"].unique()
        for teammate_flag in [True, False]:
            team_frames = [
                group[(group["id"] == eid) & (group["teammate"] == teammate_flag)][
                    ["x", "y"]
                ].values
                for eid in event_ids
            ]
            max_players = max((len(f) for f in team_frames), default=0)
            for slot in range(max_players):
                total = 0.0
                for i in range(1, len(team_frames)):
                    prev, curr = team_frames[i - 1], team_frames[i]
                    if slot < len(prev) and slot < len(curr):
                        total += float(np.linalg.norm(curr[slot] - prev[slot]))
                records.append(
                    {
                        "possession_id": poss,
                        "player_slot": slot,
                        "teammate": teammate_flag,
                        "run_distance_yards": total,
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Time-to-space
# ---------------------------------------------------------------------------

def compute_time_to_space(
    frame_df: pd.DataFrame,
    ball_x: float,
    ball_y: float,
    ball_speed: float = 5.0,
) -> float:
    """Estimate time for ball carrier to reach open space (nearest gap in defense).

    Computed as: distance to nearest uncontested corridor / ball_speed.

    Args:
        frame_df: Freeze frame rows for one event.
        ball_x: Ball x coordinate.
        ball_y: Ball y coordinate.
        ball_speed: Assumed ball speed in yards/second.

    Returns:
        Time in seconds (lower = more space available).
    """
    defenders = frame_df[frame_df["teammate"] == False][["x", "y"]].dropna()
    if defenders.empty:
        return 0.0

    ball = np.array([ball_x, ball_y])
    dists = np.linalg.norm(defenders.values - ball, axis=1)
    nearest = float(np.min(dists))
    return nearest / max(ball_speed, 0.1)


# ---------------------------------------------------------------------------
# 6. Full possession feature matrix (EPV model input)
# ---------------------------------------------------------------------------

def extract_possession_features(
    events_df: pd.DataFrame,
    freeze_frames_df: pd.DataFrame,
    lookahead: int = 5,
) -> pd.DataFrame:
    """Build one feature row per event for EPV model training/inference.

    For each event with a freeze frame, computes ball position, spatial metrics,
    pressure, player counts, and a binary shot-within-N-events label.

    Args:
        events_df: Match events from ingestion.get_match_events().
        freeze_frames_df: Freeze frames from ingestion.get_freeze_frames().
        lookahead: Number of events to look ahead when labeling shot occurrence.

    Returns:
        DataFrame with columns:
            event_id, possession, index, type_name,
            ball_x, ball_y, distance_to_goal, angle_to_goal,
            pressure_index, n_attackers_ahead, n_defenders_between,
            time_to_space, shot_within_N.
    """
    if freeze_frames_df.empty or events_df.empty:
        return pd.DataFrame()

    df = _add_ball_xy(events_df)
    frame_lookup = {eid: grp for eid, grp in freeze_frames_df.groupby("id")}

    # Extract type name safely from nested dict or plain string
    def _type_name(t):
        if isinstance(t, dict):
            return t.get("name", "")
        return str(t) if pd.notna(t) else ""

    df["type_name"] = df["type"].apply(_type_name) if "type" in df.columns else ""

    # Identify shot event IDs
    shot_ids = set(df[df["type_name"] == "Shot"]["id"].tolist())

    # Build possession-ordered event index for look-ahead labeling
    poss_events: dict = {}
    if "possession" in df.columns and "index" in df.columns:
        for _, row in df[["id", "possession", "index"]].iterrows():
            poss_events.setdefault(row["possession"], []).append(
                (row["index"], row["id"])
            )
        for p in poss_events:
            poss_events[p].sort()

    def _shot_within_n(event_id: str, n: int) -> int:
        if "possession" not in df.columns:
            return 0
        rows = df[df["id"] == event_id]
        if rows.empty:
            return 0
        r = rows.iloc[0]
        future_ids = [
            eid
            for (idx, eid) in poss_events.get(r["possession"], [])
            if r["index"] < idx <= r["index"] + n
        ]
        return int(any(eid in shot_ids for eid in future_ids))

    records = []
    for event_id, frame_df in frame_lookup.items():
        row = df[df["id"] == event_id]
        if row.empty:
            continue
        row = row.iloc[0]

        bx, by = row.get("ball_x", np.nan), row.get("ball_y", np.nan)
        if pd.isna(bx) or pd.isna(by):
            continue

        pressure = compute_pressure_index(frame_df, bx, by)
        dist_goal = compute_distance_to_goal(bx, by)
        angle_goal = compute_angle_to_goal(bx, by)
        n_ahead, n_def = count_players_ahead(frame_df, bx)
        t2s = compute_time_to_space(frame_df, bx, by)

        records.append(
            {
                "event_id": event_id,
                "possession": row.get("possession", -1),
                "index": row.get("index", -1),
                "type_name": row.get("type_name", ""),
                "ball_x": bx,
                "ball_y": by,
                "distance_to_goal": dist_goal,
                "angle_to_goal": angle_goal,
                "pressure_index": pressure,
                "n_attackers_ahead": n_ahead,
                "n_defenders_between": n_def,
                "time_to_space": t2s,
                f"shot_within_{lookahead}": _shot_within_n(event_id, lookahead),
            }
        )

    result = pd.DataFrame(records)
    logger.info("Extracted %d possession feature rows", len(result))
    return result


def save_features(features_df: pd.DataFrame, match_id: int) -> None:
    """Save feature matrix to data/processed/ as parquet.

    Args:
        features_df: Output of extract_possession_features().
        match_id: Used to name the output file.
    """
    from pathlib import Path

    out_dir = Path(__file__).parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"features_{match_id}.parquet"
    features_df.to_parquet(path, index=False)
    logger.info("Saved features to %s", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from src.ingestion import load_events, load_freeze_frames

    # Requires data already ingested — run ingestion.py first
    MATCH_ID = 3857256

    events = load_events(MATCH_ID)
    frames = load_freeze_frames(MATCH_ID)

    features = extract_possession_features(events, frames)
    print(features.head(10).to_string())
    print(f"\nShape: {features.shape}")
    print(f"\nShot label distribution:\n{features.iloc[:, -1].value_counts()}")

    save_features(features, MATCH_ID)

