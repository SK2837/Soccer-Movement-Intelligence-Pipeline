"""Soccer Movement Intelligence Pipeline — Streamlit Dashboard.

Three tabs:
1. Match Explorer: EPV heatmap, possession features, defensive metrics
2. Pose Analyzer: Sprint asymmetry and knee flexion from uploaded video/image
3. Player Comparison: Side-by-side defensive and EPV metrics across matches
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ensure src is importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion import load_events, load_freeze_frames, get_matches
from src.features import extract_possession_features
from src.defensive import compute_ppda, aggregate_voronoi, compute_defensive_pressure_score
from src.epv_model import (
    FEATURE_COLS,
    PROCESSED_DIR,
    MODELS_DIR,
    load_model,
    predict_epv,
    build_pitch_heatmap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_data
def get_available_matches() -> list[int]:
    raw = ROOT / "data" / "raw"
    return sorted(int(p.stem.replace("events_", "")) for p in raw.glob("events_*.parquet"))


@st.cache_data
def get_match_labels() -> dict[int, str]:
    """Return {match_id: 'Home vs Away (date)'} for all ingested matches."""
    try:
        matches = get_matches(competition_id=43, season_id=106)
        return {
            int(row["match_id"]): f"{row['home_team']} vs {row['away_team']} ({row['match_date']})"
            for _, row in matches.iterrows()
        }
    except Exception:
        return {}


@st.cache_resource
def get_epv_model():
    try:
        return load_model()
    except FileNotFoundError:
        return None


@st.cache_data
def load_features_cached(match_id: int) -> pd.DataFrame:
    path = PROCESSED_DIR / f"features_{match_id}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    events = load_events(match_id)
    frames = load_freeze_frames(match_id)
    return extract_possession_features(events, frames)


@st.cache_data
def load_defensive_metrics(match_id: int) -> dict:
    events = load_events(match_id)
    frames = load_freeze_frames(match_id)
    return compute_defensive_pressure_score(events, frames)


def _draw_pitch(ax, color="white"):
    ax.add_patch(patches.Rectangle((0, 0), 120, 80, fill=False, edgecolor=color, lw=2))
    ax.plot([60, 60], [0, 80], color=color, lw=1)
    ax.add_patch(patches.Circle((60, 40), 10, fill=False, edgecolor=color, lw=1))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44, fill=False, edgecolor=color, lw=1))
    ax.add_patch(patches.Rectangle((0, 18), 18, 44, fill=False, edgecolor=color, lw=1))
    ax.add_patch(patches.Rectangle((114, 30), 6, 20, fill=False, edgecolor=color, lw=1))
    ax.add_patch(patches.Rectangle((0, 30), 6, 20, fill=False, edgecolor=color, lw=1))
    ax.set_xlim(-2, 122)
    ax.set_ylim(-2, 82)
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Soccer Movement Intelligence",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Soccer Movement Intelligence Pipeline")
st.caption("FIFA World Cup 2022 · StatsBomb Open Data")

tab1, tab2, tab3 = st.tabs(["Match Explorer", "Pose Analyzer", "Player Comparison"])

match_labels = get_match_labels()


def _label(match_id: int) -> str:
    return match_labels.get(match_id, f"Match {match_id}")


# ===========================================================================
# TAB 1 — Match Explorer
# ===========================================================================
with tab1:
    st.header("Match Explorer")

    match_ids = get_available_matches()
    selected_match = st.selectbox(
        "Select Match",
        match_ids,
        format_func=_label,
        key="match_explorer_select",
    )

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader("EPV Pitch Heatmap")
        model = get_epv_model()
        if model is None:
            st.warning("EPV model not found. Run `python -m src.epv_model` to train it.")
        else:
            resolution = st.slider("Heatmap resolution", 30, 80, 50, key="heatmap_res")
            xx, yy, epv = build_pitch_heatmap(model, resolution=resolution)

            fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="#1a1a2e")
            ax.set_facecolor("#1a1a2e")
            pcm = ax.pcolormesh(xx, yy, epv, cmap="RdYlGn", vmin=0, vmax=float(epv.max()))
            plt.colorbar(pcm, ax=ax, label="EPV")
            _draw_pitch(ax)
            ax.set_title("Expected Possession Value Surface", color="white", fontsize=13)
            st.pyplot(fig)
            plt.close(fig)

    with col_right:
        st.subheader("Defensive Metrics")
        def_metrics = load_defensive_metrics(selected_match)

        st.metric("Defensive Pressure Score", f"{def_metrics['defensive_pressure_score']:.1f} / 100")
        st.metric("PPDA", f"{def_metrics['ppda']:.2f}" if def_metrics["ppda"] else "N/A",
                  help="Lower = more aggressive pressing")
        st.metric("PPDA Score", f"{def_metrics['ppda_score']:.1f}")
        st.metric("Voronoi Score", f"{def_metrics['voronoi_score']:.1f}")
        st.metric("Action Density Score", f"{def_metrics['action_density_score']:.1f}")

        st.divider()
        st.subheader("Possession Features")
        features_df = load_features_cached(selected_match)
        if not features_df.empty:
            label_col = [c for c in features_df.columns if c.startswith("shot_within_")][0]
            st.metric("Total Events w/ Freeze Frame", len(features_df))
            st.metric("Shot-Leading Sequences", int(features_df[label_col].sum()))

            fig2, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].hist(features_df["distance_to_goal"].dropna(), bins=30, color="#4CAF50", edgecolor="white")
            axes[0].set_title("Distance to Goal (yards)")
            axes[0].set_xlabel("Yards")
            axes[1].hist(features_df["pressure_index"].dropna(), bins=30, color="#FF5722", edgecolor="white")
            axes[1].set_title("Pressure Index")
            axes[1].set_xlabel("Pressure")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()
    st.subheader("Ball Position — Shot-leading Events")
    if not features_df.empty and model is not None:
        features_df["epv"] = predict_epv(model, features_df)
        label_col = [c for c in features_df.columns if c.startswith("shot_within_")][0]
        shot_events = features_df[features_df[label_col] == 1]

        fig3, ax3 = plt.subplots(figsize=(12, 7), facecolor="#1a1a2e")
        ax3.set_facecolor("#1a1a2e")
        _draw_pitch(ax3)
        sc = ax3.scatter(
            features_df["ball_x"],
            features_df["ball_y"],
            c=features_df["epv"],
            cmap="RdYlGn",
            s=8,
            alpha=0.4,
            vmin=0,
            vmax=features_df["epv"].max(),
        )
        ax3.scatter(
            shot_events["ball_x"],
            shot_events["ball_y"],
            c="red",
            s=40,
            marker="*",
            label="Shot within 5 events",
            zorder=5,
        )
        plt.colorbar(sc, ax=ax3, label="EPV")
        ax3.legend(loc="upper left", fontsize=9, framealpha=0.5)
        ax3.set_title("Ball Positions Coloured by EPV · Red Stars = Shot-leading Events", color="white")
        st.pyplot(fig3)
        plt.close(fig3)


# ===========================================================================
# TAB 2 — Pose Analyzer
# ===========================================================================
with tab2:
    st.header("Pose Analyzer")
    st.info(
        "Upload a player image or video to extract MediaPipe pose keypoints and "
        "compute sprint asymmetry index and knee flexion angles."
    )

    upload_type = st.radio("Input type", ["Image", "Video"], horizontal=True)
    uploaded = st.file_uploader(
        "Upload file",
        type=["jpg", "jpeg", "png"] if upload_type == "Image" else ["mp4", "mov", "avi"],
    )

    if uploaded is not None:
        import tempfile, os

        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            from src.pose_analysis import (
                extract_keypoints_from_image,
                extract_keypoints_from_video,
                compute_pose_metrics_per_frame,
                summarize_pose_metrics,
                compute_knee_flexion_angle,
            )

            with st.spinner("Extracting pose keypoints…"):
                if upload_type == "Image":
                    kp_df = extract_keypoints_from_image(tmp_path)
                    if kp_df is None:
                        st.error("No pose detected in this image.")
                    else:
                        st.success(f"Detected {len(kp_df)} keypoints.")
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Left Knee Flexion", f"{compute_knee_flexion_angle(kp_df, 'left'):.1f}°")
                        with col_b:
                            st.metric("Right Knee Flexion", f"{compute_knee_flexion_angle(kp_df, 'right'):.1f}°")
                        st.dataframe(kp_df[["landmark_name", "x", "y", "z", "visibility"]].round(4))
                else:
                    sample_n = st.slider("Sample every N frames", 1, 30, 5)
                    kp_seq = extract_keypoints_from_video(tmp_path, sample_every_n_frames=sample_n)
                    if kp_seq.empty:
                        st.error("No poses detected in this video.")
                    else:
                        metrics_df = compute_pose_metrics_per_frame(kp_seq)
                        summary = summarize_pose_metrics(metrics_df)

                        st.success(f"Processed {kp_seq['frame_idx'].nunique()} frames.")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Sprint Asymmetry Index", f"{summary.get('sprint_asymmetry_index', float('nan')):.1f}%")
                        c2.metric("Left Knee Flexion (mean)", f"{summary.get('left_knee_flexion_mean', float('nan')):.1f}°")
                        c3.metric("Right Knee Flexion (mean)", f"{summary.get('right_knee_flexion_mean', float('nan')):.1f}°")

                        st.subheader("Knee Flexion Over Time")
                        fig4, ax4 = plt.subplots(figsize=(10, 4))
                        ax4.plot(metrics_df["frame_idx"], metrics_df["left_knee_flexion"], label="Left", color="#2196F3")
                        ax4.plot(metrics_df["frame_idx"], metrics_df["right_knee_flexion"], label="Right", color="#FF5722")
                        ax4.set_xlabel("Frame")
                        ax4.set_ylabel("Angle (°)")
                        ax4.legend()
                        ax4.set_title("Knee Flexion Angles")
                        st.pyplot(fig4)
                        plt.close(fig4)

                        st.subheader("Stride Lengths Over Time")
                        fig5, ax5 = plt.subplots(figsize=(10, 4))
                        ax5.plot(metrics_df["frame_idx"], metrics_df["left_stride_length"], label="Left", color="#4CAF50")
                        ax5.plot(metrics_df["frame_idx"], metrics_df["right_stride_length"], label="Right", color="#9C27B0")
                        ax5.set_xlabel("Frame")
                        ax5.set_ylabel("Stride Length (normalized)")
                        ax5.legend()
                        ax5.set_title("Stride Lengths")
                        st.pyplot(fig5)
                        plt.close(fig5)

        except ImportError as e:
            st.error(f"Missing dependency: {e}")
        finally:
            os.unlink(tmp_path)
    else:
        st.caption("No file uploaded yet. Showing demo with synthetic data.")
        from src.pose_analysis import _generate_synthetic_keypoints, compute_pose_metrics_per_frame, summarize_pose_metrics

        kp_seq = _generate_synthetic_keypoints(n_frames=30)
        metrics_df = compute_pose_metrics_per_frame(kp_seq)
        summary = summarize_pose_metrics(metrics_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Sprint Asymmetry Index (demo)", f"{summary.get('sprint_asymmetry_index', float('nan')):.1f}%")
        c2.metric("Left Knee Flexion (demo)", f"{summary.get('left_knee_flexion_mean', float('nan')):.1f}°")
        c3.metric("Right Knee Flexion (demo)", f"{summary.get('right_knee_flexion_mean', float('nan')):.1f}°")


# ===========================================================================
# TAB 3 — Player Comparison
# ===========================================================================
with tab3:
    st.header("Player Comparison — Defensive & EPV Metrics Across Matches")

    all_match_ids = get_available_matches()
    selected_matches = st.multiselect(
        "Select matches to compare (up to 10)",
        all_match_ids,
        default=all_match_ids[:5],
        format_func=_label,
    )

    if not selected_matches:
        st.info("Select at least one match above.")
    else:
        with st.spinner("Computing metrics…"):
            rows = []
            model = get_epv_model()
            for mid in selected_matches[:10]:
                def_m = load_defensive_metrics(mid)
                feat = load_features_cached(mid)
                mean_epv = float(predict_epv(model, feat).mean()) if (model is not None and not feat.empty) else np.nan
                label_col = next((c for c in feat.columns if c.startswith("shot_within_")), None)
                shot_rate = float(feat[label_col].mean()) if label_col else np.nan
                rows.append(
                    {
                        "match_id": mid,
                        "ppda": def_m["ppda"],
                        "defensive_pressure_score": def_m["defensive_pressure_score"],
                        "voronoi_score": def_m["voronoi_score"],
                        "mean_epv": round(mean_epv, 4) if not np.isnan(mean_epv) else None,
                        "shot_rate": round(shot_rate, 4) if not np.isnan(shot_rate) else None,
                        "n_events": len(feat),
                    }
                )

        comparison_df = pd.DataFrame(rows)
        comparison_df["match_label"] = comparison_df["match_id"].apply(_label)
        comparison_df = comparison_df.set_index("match_label").drop(columns=["match_id"])
        st.dataframe(comparison_df.style.background_gradient(cmap="RdYlGn", axis=0), width="stretch")

        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Defensive Pressure Score")
            fig6, ax6 = plt.subplots(figsize=(7, 4))
            ax6.barh(
                list(comparison_df.index),
                comparison_df["defensive_pressure_score"],
                color="#4CAF50",
            )
            ax6.set_xlabel("Score (0–100)")
            ax6.set_title("Defensive Pressure Score by Match")
            fig6.tight_layout()
            st.pyplot(fig6)
            plt.close(fig6)

        with col2:
            st.subheader("PPDA (lower = more pressing)")
            fig7, ax7 = plt.subplots(figsize=(7, 4))
            ppda_vals = comparison_df["ppda"].fillna(0)
            ax7.barh(
                list(comparison_df.index),
                ppda_vals,
                color="#FF5722",
            )
            ax7.set_xlabel("PPDA")
            ax7.set_title("PPDA by Match")
            fig7.tight_layout()
            st.pyplot(fig7)
            plt.close(fig7)

        if model is not None:
            st.subheader("Mean EPV vs Shot Rate")
            fig8, ax8 = plt.subplots(figsize=(8, 5))
            ax8.scatter(
                comparison_df["mean_epv"],
                comparison_df["shot_rate"],
                s=80,
                color="#2196F3",
                zorder=3,
            )
            for label, row in comparison_df.iterrows():
                ax8.annotate(label, (row["mean_epv"], row["shot_rate"]), fontsize=7, ha="left")
            ax8.set_xlabel("Mean EPV")
            ax8.set_ylabel("Shot Rate (shots within 5 events)")
            ax8.set_title("Mean EPV vs Shot Rate per Match")
            ax8.grid(True, alpha=0.3)
            fig8.tight_layout()
            st.pyplot(fig8)
            plt.close(fig8)
