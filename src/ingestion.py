"""StatsBomb open data ingestion pipeline.

Pulls competition list, match events, and 360° freeze frames using statsbombpy,
then saves raw data as parquet files to data/raw/.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from statsbombpy import sb

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def get_competitions() -> pd.DataFrame:
    """Return all available StatsBomb open data competitions.

    Returns:
        DataFrame with columns: competition_id, season_id, competition_name,
        country_name, season_name, competition_gender.
    """
    return sb.competitions()


def get_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    """Return all matches for a given competition and season.

    Args:
        competition_id: StatsBomb competition ID.
        season_id: StatsBomb season ID.

    Returns:
        DataFrame with one row per match.
    """
    return sb.matches(competition_id=competition_id, season_id=season_id)


def get_match_events(match_id: int) -> pd.DataFrame:
    """Pull all events for a single match.

    Args:
        match_id: StatsBomb match ID.

    Returns:
        DataFrame with one row per event (passes, shots, pressures, etc.).
    """
    events = sb.events(match_id=match_id)
    logger.info("Loaded %d events for match %d", len(events), match_id)
    return events


def get_freeze_frames(match_id: int) -> pd.DataFrame:
    """Pull 360° freeze frames for a single match.

    Each row is one player position snapshot associated with an event.

    Args:
        match_id: StatsBomb match ID.

    Returns:
        DataFrame with columns: id (event id), teammate, actor, keeper,
        location (list [x, y]).
    """
    frames = sb.frames(match_id=match_id)
    if frames is None or frames.empty:
        logger.warning("No freeze frames available for match %d", match_id)
        return pd.DataFrame()

    # Explode nested location lists into separate x/y columns
    if "location" in frames.columns:
        frames = _expand_location(frames)

    logger.info("Loaded %d freeze frame rows for match %d", len(frames), match_id)
    return frames


def _expand_location(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a 'location' column of [x, y] lists into x/y float columns.

    Args:
        df: DataFrame with a 'location' column containing [x, y] lists.

    Returns:
        DataFrame with added 'x' and 'y' columns; original 'location' dropped.
    """
    df = df.copy()
    df["x"] = df["location"].apply(lambda loc: loc[0] if isinstance(loc, list) else None)
    df["y"] = df["location"].apply(lambda loc: loc[1] if isinstance(loc, list) else None)
    return df.drop(columns=["location"])


def save_events(events: pd.DataFrame, match_id: int) -> Path:
    """Persist match events DataFrame as parquet.

    Args:
        events: Events DataFrame from get_match_events().
        match_id: Used to name the output file.

    Returns:
        Path to the saved parquet file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"events_{match_id}.parquet"
    events.to_parquet(path, index=False)
    logger.info("Saved events to %s", path)
    return path


def save_freeze_frames(frames: pd.DataFrame, match_id: int) -> Path:
    """Persist freeze frames DataFrame as parquet.

    Args:
        frames: Freeze frames DataFrame from get_freeze_frames().
        match_id: Used to name the output file.

    Returns:
        Path to the saved parquet file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"frames_{match_id}.parquet"
    frames.to_parquet(path, index=False)
    logger.info("Saved freeze frames to %s", path)
    return path


def load_events(match_id: int) -> pd.DataFrame:
    """Load saved match events from parquet.

    Args:
        match_id: Match ID used when saving.

    Returns:
        Events DataFrame.

    Raises:
        FileNotFoundError: If parquet file does not exist for this match_id.
    """
    path = RAW_DIR / f"events_{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No saved events found at {path}. Run ingestion first.")
    return pd.read_parquet(path)


def load_freeze_frames(match_id: int) -> pd.DataFrame:
    """Load saved freeze frames from parquet.

    Args:
        match_id: Match ID used when saving.

    Returns:
        Freeze frames DataFrame.

    Raises:
        FileNotFoundError: If parquet file does not exist for this match_id.
    """
    path = RAW_DIR / f"frames_{match_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No saved frames found at {path}. Run ingestion first.")
    return pd.read_parquet(path)


def ingest_match(match_id: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full ingestion pipeline for a single match.

    Downloads events and freeze frames, saves to parquet, returns both DataFrames.
    Skips download if parquet files already exist unless force=True.

    Args:
        match_id: StatsBomb match ID.
        force: If True, re-download even if files already exist.

    Returns:
        Tuple of (events DataFrame, freeze frames DataFrame).
    """
    events_path = RAW_DIR / f"events_{match_id}.parquet"
    frames_path = RAW_DIR / f"frames_{match_id}.parquet"

    if not force and events_path.exists() and frames_path.exists():
        logger.info("Match %d already ingested. Loading from cache.", match_id)
        return load_events(match_id), load_freeze_frames(match_id)

    events = get_match_events(match_id)
    frames = get_freeze_frames(match_id)

    save_events(events, match_id)
    if not frames.empty:
        save_freeze_frames(frames, match_id)

    return events, frames


def ingest_competition(
    competition_id: int,
    season_id: int,
    max_matches: Optional[int] = None,
    force: bool = False,
) -> list[int]:
    """Ingest all (or up to max_matches) matches for a competition/season.

    Args:
        competition_id: StatsBomb competition ID.
        season_id: StatsBomb season ID.
        max_matches: Optional cap on number of matches to ingest.
        force: If True, re-download even if files already exist.

    Returns:
        List of match IDs that were ingested.
    """
    matches = get_matches(competition_id, season_id)
    match_ids = matches["match_id"].tolist()

    if max_matches is not None:
        match_ids = match_ids[:max_matches]

    logger.info(
        "Ingesting %d matches for competition %d season %d",
        len(match_ids),
        competition_id,
        season_id,
    )

    for match_id in match_ids:
        try:
            ingest_match(match_id, force=force)
        except Exception as exc:
            logger.error("Failed to ingest match %d: %s", match_id, exc)

    return match_ids


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    comps = get_competitions()
    print(comps[["competition_id", "season_id", "competition_name", "season_name"]].head(20))

    # Default: ingest 3 matches from La Liga 2020/21 (competition_id=11, season_id=90)
    COMPETITION_ID = 11
    SEASON_ID = 90
    ingest_competition(COMPETITION_ID, SEASON_ID, max_matches=3)
