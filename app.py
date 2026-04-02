import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NFL Combine Analytics", layout="wide")

METRICS = [
    "40-yd Dash",
    "Vertical Jump",
    "Bench Press",
    "Broad Jump",
    "3-Cone Drill",
    "20-yd Shuttle",
]

REVERSE_METRICS = ["40-yd Dash", "3-Cone Drill", "20-yd Shuttle"]
DATA_PATH = "data/NFL_Combine_Since_2000.csv"


def get_score_tier(score: float) -> str:
    if score >= 90:
        return "Elite"
    if score >= 80:
        return "Great"
    if score >= 70:
        return "Good"
    if score >= 60:
        return "Solid"
    return "Average"


@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    return df


@st.cache_data
def prepare_and_score_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Player", "Position", "Height", "Weight",
        "40-yd Dash", "Vertical Jump", "Bench Press",
        "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
        "Year"
    ]
    df = raw_df[columns].copy()

    df["metrics_available"] = df[METRICS].notna().sum(axis=1)
    df = df[df["metrics_available"] >= 3].copy()

    position_counts = df["Position"].value_counts()
    valid_positions = position_counts[position_counts >= 20].index
    df = df[df["Position"].isin(valid_positions)].copy()

    for metric in METRICS:
        df[f"{metric}_z"] = df.groupby("Position")[metric].transform(
            lambda x: (x - x.mean()) / x.std()
        )

    for metric in REVERSE_METRICS:
        df[f"{metric}_z"] *= -1

    z_columns = [f"{metric}_z" for metric in METRICS]
    df["Athletic_Score_Raw"] = df[z_columns].mean(axis=1, skipna=True)

    min_score = df["Athletic_Score_Raw"].min()
    max_score = df["Athletic_Score_Raw"].max()
    df["Athletic_Score"] = (
        (df["Athletic_Score_Raw"] - min_score) / (max_score - min_score)
    ) * 100

    df = df.sort_values("Athletic_Score", ascending=False).copy()
    df["Position_Rank"] = (
        df.groupby("Position")["Athletic_Score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    df["Tier"] = df["Athletic_Score"].apply(get_score_tier)
    return df


def find_similar_players(player: pd.Series, df: pd.DataFrame, top_n: int = 5, min_shared_metrics: int = 3) -> pd.DataFrame:
    position = player["Position"]
    position_df = df[df["Position"] == position].copy()
    position_df = position_df[position_df["Player"] != player["Player"]].copy()

    z_columns = [f"{metric}_z" for metric in METRICS]
    player_vector = player[z_columns]

    def calculate_distance(row: pd.Series) -> pd.Series:
        squared_diffs = []
        shared_metric_count = 0

        for col in z_columns:
            if pd.notna(player_vector[col]) and pd.notna(row[col]):
                squared_diffs.append((player_vector[col] - row[col]) ** 2)
                shared_metric_count += 1

        if shared_metric_count < min_shared_metrics:
            return pd.Series([np.nan, shared_metric_count])

        distance = np.sqrt(sum(squared_diffs))
        return pd.Series([distance, shared_metric_count])

    position_df[["Similarity_Distance", "Shared_Metrics"]] = position_df.apply(
        calculate_distance, axis=1
    )

    position_df = position_df.dropna(subset=["Similarity_Distance"]).copy()
    similar_players = position_df.sort_values(
        by=["Similarity_Distance", "Athletic_Score"],
        ascending=[True, False]
    ).head(top_n)

    similar_players = similar_players[[
        "Player", "Position", "Year", "Athletic_Score", "Tier", "Similarity_Distance", "Shared_Metrics"
    ]].copy()
    similar_players["Athletic_Score"] = similar_players["Athletic_Score"].round(2)
    similar_players["Similarity_Distance"] = similar_players["Similarity_Distance"].round(3)
    similar_players["Shared_Metrics"] = similar_players["Shared_Metrics"].astype(int)
    return similar_players


def display_metric_value(value):
    return "N/A" if pd.isna(value) else value


raw_df = load_data(DATA_PATH)
scored_df = prepare_and_score_data(raw_df)

st.title("🏈 NFL Combine Analytics")
st.caption("Position-adjusted scoring, player lookup, and similarity matching.")

with st.sidebar:
    st.header("Filters")
    positions = ["All"] + sorted(scored_df["Position"].unique().tolist())
    selected_position = st.selectbox("Position", positions)
    year_min = int(scored_df["Year"].min())
    year_max = int(scored_df["Year"].max())
    selected_years = st.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
    top_n = st.slider("Top players to display", min_value=5, max_value=25, value=10)

filtered_df = scored_df[
    (scored_df["Year"] >= selected_years[0]) &
    (scored_df["Year"] <= selected_years[1])
].copy()

if selected_position != "All":
    filtered_df = filtered_df[filtered_df["Position"] == selected_position].copy()

left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Leaderboard")
    leaderboard = filtered_df[["Player", "Position", "Year", "Athletic_Score", "Tier", "Position_Rank"]].head(top_n).copy()
    leaderboard["Athletic_Score"] = leaderboard["Athletic_Score"].round(2)
    st.dataframe(leaderboard, use_container_width=True, hide_index=True)

    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = filtered_df.head(top_n).copy()
        plot_df = plot_df.sort_values("Athletic_Score", ascending=True)
        ax.barh(plot_df["Player"], plot_df["Athletic_Score"])
        ax.set_title(f"Top {min(top_n, len(plot_df))} Athletic Scores")
        ax.set_xlabel("Athletic Score (0-100)")
        st.pyplot(fig)
    else:
        st.info("No players match the current filters.")

with right:
    st.subheader("Player Lookup")
    player_query = st.text_input("Search by player name", placeholder="e.g. Anthony Richardson")

    if player_query:
        exact_matches = filtered_df[filtered_df["Player"].str.lower() == player_query.lower()]
        partial_matches = filtered_df[filtered_df["Player"].str.lower().str.contains(player_query.lower(), na=False)]
        results = exact_matches if not exact_matches.empty else partial_matches

        if results.empty:
            raw_matches = raw_df[raw_df["Player"].str.lower().str.contains(player_query.lower(), na=False)]
            if raw_matches.empty:
                st.warning("No player found in the dataset.")
            else:
                st.warning("Player exists in the raw dataset but is not available in the scored dataset.")
                st.dataframe(raw_matches[["Player", "Position", "Year"]].drop_duplicates(), use_container_width=True, hide_index=True)
        else:
            if len(results) > 1:
                selected_name = st.selectbox("Matching players", results["Player"].tolist())
                player = results[results["Player"] == selected_name].iloc[0]
            else:
                player = results.iloc[0]

            st.markdown(f"### {player['Player']}")
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Athletic Score", f"{player['Athletic_Score']:.2f}/100")
            metric_b.metric("Tier", player["Tier"])
            metric_c.metric("Position Rank", int(player["Position_Rank"]))

            st.write(f"**Position:** {player['Position']}")
            st.write(f"**Year:** {int(player['Year'])}")

            st.write("**Combine Metrics**")
            metrics_table = pd.DataFrame({
                "Metric": METRICS,
                "Value": [display_metric_value(player[m]) for m in METRICS],
                "Z-Score": [round(player[f"{m}_z"], 2) if pd.notna(player[f"{m}_z"]) else "N/A" for m in METRICS]
            })
            st.dataframe(metrics_table, use_container_width=True, hide_index=True)

            z_map = {
                metric: player[f"{metric}_z"]
                for metric in METRICS
                if pd.notna(player[f"{metric}_z"])
            }
            if z_map:
                strongest_metric = max(z_map, key=z_map.get)
                weakest_metric = min(z_map, key=z_map.get)
                st.write("**Performance Summary**")
                st.write(f"Strongest metric: {strongest_metric} ({z_map[strongest_metric]:.2f} z-score)")
                st.write(f"Weakest metric: {weakest_metric} ({z_map[weakest_metric]:.2f} z-score)")

            similar_players = find_similar_players(player, scored_df, top_n=5, min_shared_metrics=3)
            st.write("**Most Similar Players**")
            if similar_players.empty:
                st.info("No strong similarity matches found with enough shared combine metrics.")
            else:
                st.dataframe(similar_players, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Export")
output_columns = [
    "Player", "Position", "Year",
    "Height", "Weight",
    "40-yd Dash", "Vertical Jump", "Bench Press",
    "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
    "Athletic_Score", "Tier", "Position_Rank"
]

csv_data = scored_df[output_columns].to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download ranked combine players CSV",
    data=csv_data,
    file_name="ranked_combine_players.csv",
    mime="text/csv"
)
