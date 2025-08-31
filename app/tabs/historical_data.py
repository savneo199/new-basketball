# historical_data.py (no compare feature, optimized)
# - Loads DuckDB -> DataFrame once per container (cache_resource)
# - Caches team & season lists (cache_data)
# - Caps table size for smooth scrolling
# - Lineup click shows quick stats (no add-to-compare)

from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

from helpers.helpers import (
    latest_artifacts,
    rename_columns,
    load_json_file,
    load_duckdb,
    COLLEGE_MAP,
    COLLEGE_MAP_INV,
)
from helpers.archetype_positions import normalize_position, positions_for_archetype
from helpers.court_builder import make_lineup_figure

MAX_TABLE_ROWS = 1000  # cap UI table rows for responsiveness


# ----------------------------
# Data loading & caching
# ----------------------------
@st.cache_resource(show_spinner=False)
def _load_df_and_version(processed_path: Path) -> tuple[pd.DataFrame, str]:
    """Load processed data once per instance and normalize columns once."""
    df = load_duckdb(processed_path)

    if "college" in df.columns:
        cn = df["college"].astype("string").str.strip().str.lower()
        df = df.assign(
            **{
                " college_norm": cn,
                "college_display": cn.map(COLLEGE_MAP).fillna(df["college"]),
            }
        )

    # small fingerprint to invalidate caches if file changes
    data_version = str(processed_path.stat().st_mtime_ns)
    return df, data_version


@st.cache_data(show_spinner=False)
def _team_list(data_version: str, colleges: pd.Series) -> tuple[str, ...]:
    """Cached list of teams (college_display)."""
    return tuple(sorted(colleges.dropna().astype(str).unique().tolist()))


@st.cache_data(show_spinner=False)
def _seasons_for_norm(data_version: str, df: pd.DataFrame, norm: str) -> tuple:
    """Cached seasons for a given normalized college name."""
    return tuple(
        sorted(
            df.loc[df[" college_norm"] == norm, "season"].dropna().astype(str).unique().tolist()
        )
    )


# ----------------------------
# Utilities
# ----------------------------
def _num(v) -> float:
    try:
        if v is None or pd.isna(v):
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _assign_slots_by_position(df_top5: pd.DataFrame):
    slots = ["PG", "SG", "SF", "PF", "C"]
    assigned, leftovers = {}, []

    for _, row in df_top5.iterrows():
        raw_pos = str(row.get("position", "") or "")
        pos = normalize_position(raw_pos)
        if not pos:
            arc = str(row.get("Archetype", "") or "")
            prefs = positions_for_archetype(arc)
            pos = prefs[0] if prefs else ""
        if pos in slots and pos not in assigned:
            assigned[pos] = row
        else:
            leftovers.append(row)

    def pref_chain(pos_guess: str) -> list[str]:
        if pos_guess in ("PG", "SG"):
            return ["PG", "SG", "SF", "PF", "C"]
        if pos_guess == "SF":
            return ["SF", "PF", "SG", "PG", "C"]
        if pos_guess == "PF":
            return ["PF", "C", "SF", "SG", "PG"]
        if pos_guess == "C":
            return ["C", "PF", "SF", "SG", "PG"]
        return ["SF", "PF", "PG", "SG", "C"]

    for row in leftovers:
        raw_pos = str(row.get("position", "") or "")
        guess = normalize_position(raw_pos)
        if not guess:
            arc = str(row.get("Archetype", "") or "")
            prefs = positions_for_archetype(arc)
            guess = prefs[0] if prefs else ""
        for s in pref_chain(guess):
            if s not in assigned:
                assigned[s] = row
                break

    ordered_slots = [s for s in slots if s in assigned]
    ordered_rows = [assigned[s] for s in ordered_slots]
    return ordered_rows, ordered_slots


# ----------------------------
# Main render
# ----------------------------
def render():
    st.subheader("Roster & Metrics")

    paths = latest_artifacts()
    if not paths or not paths.get("processed") or not paths["processed"].exists():
        st.info("Run pipeline to populate processed data.")
        return

    df, data_version = _load_df_and_version(paths["processed"])

    # sanity checks
    if "season" not in df.columns or "college_display" not in df.columns or " college_norm" not in df.columns:
        st.warning("Expected columns not found in processed data.")
        return

    # ---- Team + season selectors (cached) ----
    teams = _team_list(data_version, df["college_display"])
    team_display = st.selectbox("Team", teams, index=0 if teams else None)
    if not team_display:
        return

    selected_norm = COLLEGE_MAP_INV.get(team_display, str(team_display).strip().lower())
    seasons_for_team = _seasons_for_norm(data_version, df, selected_norm)
    if not seasons_for_team:
        st.info("No seasons found for the selected team.")
        return

    season = st.selectbox("Season", seasons_for_team, index=0)
    show_adv = st.checkbox("Show advanced metrics", value=False)

    # ---- Filter only needed rows ----
    filt = df.loc[(df[" college_norm"] == selected_norm) & (df["season"].astype(str) == str(season))]

    # ---- Roster table (lean render) ----
    if show_adv:
        drop_cols = [c for c in [" college_norm", "college_display", "college", "season"] if c in filt.columns]
        view = filt.drop(columns=drop_cols, errors="ignore")
    else:
        minimal_cols = [
            "player_ind",
            "player_number_ind",
            "mins_per_game",
            "Archetype",
            "pts_per_game",
            "ast_per_game",
            "reb_per_game",
            "stl_per_game",
            "blk_per_game",
            "eFG_pct",
            "gp_ind",
        ]
        cols = [c for c in minimal_cols if c in filt.columns]
        view = filt.loc[:, cols]

    view_display = rename_columns(view)

    st.dataframe(
        view_display.head(MAX_TABLE_ROWS),
        hide_index=True,
        use_container_width=True,
        height=480,
    )
    if len(view_display) > MAX_TABLE_ROWS:
        st.caption(f"Showing first {MAX_TABLE_ROWS:,} rows.")
    st.download_button(
        "Download CSV (current selection)",
        data=view_display.to_csv(index=False).encode(),
        file_name=f"{team_display}_{season}_players.csv",
        mime="text/csv",
    )

    # ----------------------------
    # Predict lineup
    # ----------------------------
    st.markdown("---")
    st.subheader("Predict lineup")

    # Choose a minutes column fallback chain
    minutes_col = None
    for cand in ["minutes_tot_ind", "minutes_tot", "minutes", "mins_per_game"]:
        if cand in filt.columns:
            minutes_col = cand
            break

    if not minutes_col:
        st.info("Not enough data to build a lineup (need minutes).")
        return

    # Top 5 by minutes
    top5 = filt.sort_values(minutes_col, ascending=False).head(5).reset_index(drop=True)
    rows, slots_order = _assign_slots_by_position(top5)

    # Build labels, numbers, and stats
    labels, numbers, stats_list, names_for_click = [], [], [], []
    for _, r in enumerate(rows):
        name = r.get("player_ind", "Player")
        arch = r.get("Archetype", "")
        labels.append(f"{name} ({arch})" if arch else str(name))

        # jersey number
        num = ""
        if "player_number_ind" in r.index:
            try:
                num = str(int(float(r["player_number_ind"])))
            except Exception:
                num = str(r["player_number_ind"])
        numbers.append(num)

        # eFG% to %
        efg = r.get("eFG_pct")
        if pd.notna(efg):
            try:
                efg = float(efg)
                if efg <= 1.0:
                    efg *= 100.0
            except Exception:
                efg = None
        else:
            efg = None

        stats_list.append(
            {
                "PTS/Game": r.get("pts_per_game", 0.0),
                "AST/Game": r.get("ast_per_game", 0.0),
                "REB/Game": r.get("reb_per_game", 0.0),
                "STL/Game": r.get("stl_per_game", 0.0),
                "BLK/Game": r.get("blk_per_game", 0.0),
                "eFG%": efg,
            }
        )
        names_for_click.append(str(name))

    fig = make_lineup_figure(
        labels,
        title=f"Possible lineup for {team_display} ({season})",
        slots_order=slots_order,
        numbers=numbers,
        stats=stats_list,
    )

    lineup_plotly_key = f"lineup_click_{selected_norm}_{season}"

    try:
        selected = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            key=lineup_plotly_key,
        )
        if selected:
            idx = selected[0].get("pointIndex", selected[0].get("pointNumber", 0))
            idx = int(idx)
            if 0 <= idx < len(stats_list):
                s = stats_list[idx]
                nm = names_for_click[idx]
                st.markdown(f"**{nm} — quick stats**")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("PTS/Game", f"{float(s['PTS/Game']):.3f}")
                c2.metric("AST/Game", f"{float(s['AST/Game']):.3f}")
                c3.metric("REB/Game", f"{float(s['REB/Game']):.3f}")
                c4.metric("BLK/Game", f"{float(s['BLK/Game']):.3f}")
                efgtxt = f"{float(s['eFG%']):.3f}%" if s["eFG%"] is not None else "—"
                c5.metric("eFG%", efgtxt)
        else:
            st.caption("Tip: click a circle to pin a stat card; hover for details.")
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=f"lineup_chart_{selected_norm}_{season}")
        st.caption("Tip: hover a circle to see stats. (Install `streamlit-plotly-events` to enable click.)")

    # ----------------------------
    # Notes section
    # ----------------------------
    st.markdown("---")
    st.subheader("Notes & Comments")

    notes_file = Path("app_notes.json")
    notes = load_json_file(notes_file) if notes_file.exists() else {}
    key = f"{team_display}__{season}"
    existing_text = notes.get(key, "")
    txt = st.text_area("Write notes for this team/season", value=existing_text, height=160, key="notes_roster")
    if st.button("Save notes", key="save_notes_roster"):
        notes[key] = txt
        notes_file.write_text(json.dumps(notes, indent=2))
        st.success("Notes saved")
