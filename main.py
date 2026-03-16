import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/NFL_Combine_Since_2000.csv")
df.columns = df.columns.str.strip()

# Keep a copy of the original loaded data for name searching
raw_df = df.copy()

# Keep only relevant columns
columns = [
    "Player", "Position", "Height", "Weight",
    "40-yd Dash", "Vertical Jump", "Bench Press",
    "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
    "Year"
]
df = df[columns]

# Define combine metrics
metrics = [
    "40-yd Dash",
    "Vertical Jump",
    "Bench Press",
    "Broad Jump",
    "3-Cone Drill",
    "20-yd Shuttle"
]

# Require at least 4 available metrics
df["metrics_available"] = df[metrics].notna().sum(axis=1)
df = df[df["metrics_available"] >= 2].copy()

# Remove positions with very small sample sizes
position_counts = df["Position"].value_counts()
valid_positions = position_counts[position_counts >= 20].index
df = df[df["Position"].isin(valid_positions)].copy()

# Create z-scores within each position
for metric in metrics:
    df[f"{metric}_z"] = df.groupby("Position")[metric].transform(
        lambda x: (x - x.mean()) / x.std()
    )

# Reverse metrics where lower is better
reverse_metrics = ["40-yd Dash", "3-Cone Drill", "20-yd Shuttle"]
for metric in reverse_metrics:
    df[f"{metric}_z"] *= -1

# Create athletic score
z_columns = [f"{metric}_z" for metric in metrics]
df["Athletic_Score_Raw"] = df[z_columns].mean(axis=1, skipna=True)

min_score = df["Athletic_Score_Raw"].min()
max_score = df["Athletic_Score_Raw"].max()

df["Athletic_Score"] = (
    (df["Athletic_Score_Raw"] - min_score) / (max_score - min_score)
) * 100


# Sort best to worst
df = df.sort_values("Athletic_Score", ascending=False).copy()

# Create rank within each position
df["Position_Rank"] = (
    df.groupby("Position")["Athletic_Score"]
    .rank(method="min", ascending=False)
    .astype(int)
)

# Print top 15 overall
# print("\nTop 15 Overall Athletic Scores:\n")
top15 = df[["Player", "Position", "Year", "Athletic_Score"]].head(15).copy()
top15["Athletic_Score"] = top15["Athletic_Score"].round(2)

print("\nTop 15 Overall Athletic Scores:\n")
print(top15.to_string(index=False))

# Save to CSV
output_columns = [
    "Player", "Position", "Year",
    "Height", "Weight",
    "40-yd Dash", "Vertical Jump", "Bench Press",
    "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
    "Athletic_Score", "Position_Rank"
]

df[output_columns].to_csv("ranked_combine_players.csv", index=False)

# Plot top 10 overall
top10 = df.head(10)
top10["Athletic_Score"] = top10["Athletic_Score"].round(2)

plt.figure(figsize=(10, 6))
plt.barh(top10["Player"], top10["Athletic_Score"])
plt.gca().invert_yaxis()
plt.title("Top 10 Athletic Scores Since 2000")
plt.xlabel("Athletic Score")
plt.tight_layout()
plt.show()


def get_score_tier(score):
    if score >= 90:
        return "Elite"
    elif score >= 80:
        return "Great"
    elif score >= 70:
        return "Good"
    elif score >= 60:
        return "Solid"
    else:
        return "Average"
    

def find_similar_players(player, df, metrics, top_n=5, min_shared_metrics=3):
    position = player["Position"]

    # Only compare within same position
    position_df = df[df["Position"] == position].copy()

    # Remove the player themselves
    position_df = position_df[position_df["Player"] != player["Player"]].copy()

    z_columns = [f"{metric}_z" for metric in metrics]
    player_vector = player[z_columns]

    def calculate_distance(row):
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

    return similar_players[
        ["Player", "Position", "Year", "Athletic_Score", "Similarity_Distance", "Shared_Metrics"]
    ]
# -----------------------------
# Player lookup feature
# -----------------------------
player_name = input("\nEnter a player name to search: ").strip()

# First try exact match in scored dataframe
player_result = df[df["Player"].str.lower() == player_name.lower()]

# If no exact match, try partial match
if player_result.empty:
    player_result = df[df["Player"].str.lower().str.contains(player_name.lower(), na=False)]

if player_result.empty:
    print(f"\nNo scored player found with name: {player_name}")

    raw_result = raw_df[raw_df["Player"].str.lower().str.contains(player_name.lower(), na=False)]

    if raw_result.empty:
        print("That player does not appear to exist in this dataset.")
    else:
        print("\nPlayer exists in the dataset but was filtered out before scoring.")
        print(raw_result[["Player", "Position", "Year"]].drop_duplicates().head(10))

        for _, row in raw_result.head(1).iterrows():
            print("\nReason check:")
            player_row = row[metrics]
            available_metrics = player_row.notna().sum()
            print(f"Available combine metrics: {available_metrics}/{len(metrics)}")
            print("Recorded metrics:")
            print(player_row)
else:
    player = player_result.iloc[0]
    tier = get_score_tier(player["Athletic_Score"])

    print("\nPlayer Lookup Result:")
    print(f"Player: {player['Player']}")
    print(f"Position: {player['Position']}")
    print(f"Year: {int(player['Year'])}")
    print(f"Athletic Score: {player['Athletic_Score']:.2f}/100")
    print(f"Tier: {tier}")
    print(f"Position Rank: {player['Position_Rank']}")

    # Show available raw metrics
    print("\nCombine Metrics:")
    for metric in metrics:
        value = player[metric]
        display_value = "N/A" if pd.isna(value) else value
        print(f"{metric}: {display_value}")

    # Find strongest and weakest z-score metrics
    z_map = {metric: player[f"{metric}_z"] for metric in metrics if pd.notna(player[f"{metric}_z"])}
    if z_map:
        strongest_metric = max(z_map, key=z_map.get)
        weakest_metric = min(z_map, key=z_map.get)

        print("\nPerformance Summary:")
        print(f"Strongest metric: {strongest_metric} ({z_map[strongest_metric]:.2f} z-score)")
        print(f"Weakest metric: {weakest_metric} ({z_map[weakest_metric]:.2f} z-score)")
        
        
        similar_players = find_similar_players(player, df, metrics, top_n=5, min_shared_metrics=3)

        if not similar_players.empty:
            similar_players = similar_players.copy()
            similar_players["Athletic_Score"] = similar_players["Athletic_Score"].round(2)
            similar_players["Similarity_Distance"] = similar_players["Similarity_Distance"].round(3)
            similar_players["Shared_Metrics"] = similar_players["Shared_Metrics"].astype(int)

            print("\nMost Similar Players:")
            print(similar_players.to_string(index=False))
        else:
            print("\nNo strong similarity matches found with enough shared combine metrics.")