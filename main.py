import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/NFL_Combine_Since_2000.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop rows missing 40 time (core metric)
df = df.dropna(subset=["40-yd Dash"])

# Only keep relevant columns
columns = [
    "Player", "Position", "Height", "Weight",
    "40-yd Dash", "Vertical Jump", "Bench Press",
    "Broad Jump", "3-Cone Drill", "20-yd Shuttle",
    "Year"
]
df = df[columns]

# Function to calculate z-score within each position
def z_score(group, column):
    return (group[column] - group[column].mean()) / group[column].std()

metrics = [
    "40-yd Dash",
    "Vertical Jump",
    "Bench Press",
    "Broad Jump",
    "3-Cone Drill",
    "20-yd Shuttle"
]

# Create z-scores by position
for metric in metrics:
    df[metric + "_z"] = df.groupby("Position")[metric].transform(
        lambda x: (x - x.mean()) / x.std()
    )

# Reverse speed metrics (lower is better)
df["40-yd Dash_z"] *= -1
df["3-Cone Drill_z"] *= -1
df["20-yd Shuttle_z"] *= -1

# Create final athletic score
z_columns = [m + "_z" for m in metrics]
df["Athletic_Score"] = df[z_columns].mean(axis=1)

# Sort by best score
df = df.sort_values("Athletic_Score", ascending=False)

print(df[["Player", "Position", "Athletic_Score"]].head(15))



# Check column names
print(df.columns)



import matplotlib.pyplot as plt

top10 = df.head(10)

plt.barh(top10["Player"], top10["Athletic_Score"])
plt.gca().invert_yaxis()
plt.title("Top 10 Athletic Scores Since 2000")
plt.xlabel("Athletic Score")
plt.show()