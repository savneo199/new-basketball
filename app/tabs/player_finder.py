from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from helpers.helpers import latest_artifacts, load_parquet

CORE_SHOW = [
    "player_ind", "college", "season", "position", "player_number_ind",
    "minutes_tot_ind", "mins_per_game", "pts_per_game", "reb_per_game", "ast_per_game",
    "stl_per_game", "blk_per_game", "eFG_pct", "USG_pct", "Archetype"
]

def _load_df() -> pd.DataFrame | None:
    paths = latest_artifacts()
    if not paths or not Path(paths["processed"]).exists():
        return None
    try:
        return load_parquet(Path(paths["processed"]))
    except Exception:
        return None

def _search_names(df: pd.DataFrame, query: str) -> list[str]:
    if not query:
        return []
    s = df.get("player_ind") if "player_ind" in df.columns else None
    if s is None:
        return []
    mask = s.astype(str).str.contains(query.strip(), case=False, na=False)
    return sorted(s[mask].dropna().unique().tolist())

def render():
    st.subheader("Player finder")

    df = _load_df()
    if df is None or df.empty:
        st.info("No processed data found. Run the pipeline first.")
        return

    q = st.text_input("Search by player name", placeholder="Type at least 2 characters…")
    suggestions = _search_names(df, q) if len(q.strip()) >= 2 else []

    if suggestions:
        name = st.selectbox("Matches", suggestions, index=0)
        sub = df[df["player_ind"].astype(str) == str(name)].copy()

        # season selector when multiple rows
        season_opt = None
        if "season" in sub.columns:
            uniq = sorted(sub["season"].dropna().unique().tolist())
            if len(uniq) > 1:
                season_opt = st.selectbox("Season", uniq, index=0)
                sub = sub[sub["season"] == season_opt]

        # if multiple rows remain (e.g. different colleges same season), let user pick college too
        if len(sub) > 1 and "college" in sub.columns:
            colleges = sorted(sub["college"].dropna().unique().tolist())
            if len(colleges) > 1:
                college_opt = st.selectbox("College", colleges, index=0)
                sub = sub[sub["college"] == college_opt]

        if sub.empty:
            st.warning("No row found for that selection.")
            return

        row = sub.iloc[0]
        title = f"{row.get('player_ind','Player')} — {row.get('college','')} ({row.get('season','')})"
        st.markdown(f"### {title}")

        # KPI row (per-game)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("PTS", f"{float(row.get('pts_per_game',0) or 0):.1f}")
        c2.metric("REB", f"{float(row.get('reb_per_game',0) or 0):.1f}")
        c3.metric("AST", f"{float(row.get('ast_per_game',0) or 0):.1f}")
        c4.metric("STL", f"{float(row.get('stl_per_game',0) or 0):.1f}")
        c5.metric("BLK", f"{float(row.get('blk_per_game',0) or 0):.1f}")

        # Details table
        cols = [c for c in CORE_SHOW if c in sub.columns]
        pretty = sub[cols].copy()
        st.dataframe(pretty, use_container_width=True)

        # Download
        st.download_button(
            "Download row (CSV)",
            data=pretty.to_csv(index=False).encode(),
            file_name=f"{row.get('player_ind','player')}_{row.get('season','')}.csv",
            mime="text/csv",
        )
    elif q:
        st.info("No matches yet. Keep typing or try a different name.")
