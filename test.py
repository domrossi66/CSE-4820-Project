

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist, pdist, squareform


df = pd.read_csv(r"games_details_cleaned.csv")

print("Original dataset:")
print(df.head())

print("Dataset dimensions:", df.shape)


sum_columns = [
    "FGM", "FGA", "FG3M", "FG3A",
    "FTM", "FTA", "OREB", "DREB",
    "AST", "STL", "BLK", "TO",
    "PF", "PTS", "PLUS_MINUS"
]

player_stats = df.groupby(
    ["PLAYER_ID", "PLAYER_NAME"]
).agg(
    {
        **{col: "sum" for col in sum_columns},
        "MIN": "mean",
        "GAME_ID": "nunique"
    }
).reset_index()

player_stats = player_stats.rename(
    columns={"GAME_ID": "GAMES_PLAYED"}
)

# Convert counting statistics to per-game averages
per_game_columns = [
    "OREB", "DREB", "AST", "STL", "BLK",
    "TO", "PF", "PTS", "FG3A"
]

for col in per_game_columns:
    player_stats[col] /= player_stats["GAMES_PLAYED"]

# Recalculate shooting percentages from season totals
player_stats["FG_PCT"] = (
    player_stats["FGM"] /
    player_stats["FGA"].replace(0, np.nan)
)

player_stats["FG3_PCT"] = (
    player_stats["FG3M"] /
    player_stats["FG3A"].replace(0, np.nan)
)

player_stats["FT_PCT"] = (
    player_stats["FTM"] /
    player_stats["FTA"].replace(0, np.nan)
)

player_stats = player_stats.replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# Only include players who played at least 10 games
player_stats = player_stats[
    player_stats["GAMES_PLAYED"] >= 10
].copy()

print("\nPlayer-season records:", len(player_stats))


# 4. SELECT FEATURES

features = [
    "PTS", "OREB", "DREB",
    "AST", "STL", "BLK", "TO",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    "FG3A", "MIN"
]

X = player_stats[features].to_numpy(dtype=float)

# Handle missing or infinite values
X = np.nan_to_num(X)


# 5. STANDARDIZE DATA USING NUMPY

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)

# Prevent division by zero
stds[stds == 0] = 1

X_scaled = (X - means) / stds

print("\nData standardized successfully.")


# 6. IMPLEMENT K-MEANS

def kmeans_clustering(X, k, max_iterations=300, random_state=42):

    rng = np.random.default_rng(random_state)

    # Randomly select initial centroids from observations
    indices = rng.choice(len(X), size=k, replace=False)
    centroids = X[indices].copy()

    for iteration in range(max_iterations):

        # Calculate distance from each point to each centroid
        distances = cdist(X, centroids)

        # Assign each point to its closest centroid
        labels = np.argmin(distances, axis=1)

        # Calculate new centroids
        new_centroids = np.array([
            X[labels == cluster].mean(axis=0)
            if np.any(labels == cluster)
            else centroids[cluster]
            for cluster in range(k)
        ])

        # Stop if centroids no longer change significantly
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Calculate final labels
    distances = cdist(X, centroids)
    labels = np.argmin(distances, axis=1)

    # Calculate inertia (within-cluster sum of squared distances)
    inertia = np.sum(
        (X - centroids[labels]) ** 2
    )

    return labels, centroids, inertia


# 7. IMPLEMENT SILHOUETTE SCORE

def silhouette_calculation(X, labels):

    distances = squareform(pdist(X))

    unique_clusters = np.unique(labels)

    if len(unique_clusters) < 2:
        return np.nan

    scores = []

    for i in range(len(X)):

        own_cluster = labels[i]

        # Average distance to points in the same cluster
        same_cluster = np.where(labels == own_cluster)[0]
        same_cluster = same_cluster[same_cluster != i]

        if len(same_cluster) == 0:
            scores.append(0)
            continue

        a = np.mean(distances[i, same_cluster])

        # Find the closest alternative cluster
        b = np.inf

        for cluster in unique_clusters:

            if cluster == own_cluster:
                continue

            other_cluster = np.where(labels == cluster)[0]

            average_distance = np.mean(
                distances[i, other_cluster]
            )

            b = min(b, average_distance)

        # Calculate silhouette value
        score = (b - a) / max(a, b)
        scores.append(score)

    return np.mean(scores)


# 8. FIND OPTIMAL NUMBER OF CLUSTERS

k_values = range(2, 11)

inertia_values = []
silhouette_values = []

for k in k_values:

    labels, centroids, inertia = kmeans_clustering(
        X_scaled, k
    )

    score = silhouette_calculation(
        X_scaled, labels
    )

    inertia_values.append(inertia)
    silhouette_values.append(score)

    print(
        f"K = {k} | "
        f"Inertia = {inertia:.2f} | "
        f"Silhouette = {score:.4f}"
    )



optimal_k = 3

kmeans_labels, centroids, inertia = kmeans_clustering(
    X_scaled, optimal_k
)

player_stats["KMeans_Cluster"] = kmeans_labels

kmeans_score = silhouette_calculation(
    X_scaled, kmeans_labels
)

print("\nK-Means Silhouette Score:", kmeans_score)


hierarchical_linkage = linkage(
    X_scaled,
    method="ward"
)

hierarchical_labels = fcluster(
    hierarchical_linkage,
    t=optimal_k,
    criterion="maxclust"
)

# Convert cluster labels to start at zero
hierarchical_labels -= 1

player_stats["Hierarchical_Cluster"] = hierarchical_labels

hierarchical_score = silhouette_calculation(
    X_scaled, hierarchical_labels
)

print(
    "\nHierarchical Silhouette Score:",
    hierarchical_score
)


print("\nClustering Method Comparison")
print("-----------------------------")

print("K-Means:", kmeans_score)
print("Hierarchical:", hierarchical_score)


 
# Center the standardized data
X_centered = X_scaled - np.mean(X_scaled, axis=0)

# Singular Value Decomposition
U, S, Vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)

# Project data onto the first two principal components
X_pca = U[:, :2] * S[:2]

player_stats["PCA1"] = X_pca[:, 0]
player_stats["PCA2"] = X_pca[:, 1]

# Calculate explained variance
explained_variance = S ** 2 / (len(X_scaled) - 1)

explained_ratio = (
    explained_variance[:2].sum() /
    explained_variance.sum()
)

print("\nPCA explained variance (first 2 components):",
      explained_ratio)


cluster_profiles = player_stats.groupby(
    "KMeans_Cluster"
)[features].mean()

print("\nK-Means Cluster Profiles:")
print(cluster_profiles.round(2))


for cluster in sorted(
    player_stats["KMeans_Cluster"].unique()
):

    print(f"\nCLUSTER {cluster}")

    players = player_stats[
        player_stats["KMeans_Cluster"] == cluster
    ].sort_values(
        by="PTS",
        ascending=False
    )

    print(
    players[
        ["PLAYER_NAME"] + features
    ].head(15).to_string(index=False)
)


plt.figure(figsize=(10, 7))

for cluster in sorted(player_stats["KMeans_Cluster"].unique()):

    subset = player_stats[
        player_stats["KMeans_Cluster"] == cluster
    ]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        label=f"Cluster {cluster}",
        s=50
    )

plt.title("NBA Player Archetypes - K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(title="Cluster")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 7))

for cluster in sorted(
    player_stats["Hierarchical_Cluster"].unique()
):

    subset = player_stats[
        player_stats["Hierarchical_Cluster"] == cluster
    ]

    plt.scatter(
        subset["PCA1"],
        subset["PCA2"],
        label=f"Cluster {cluster}",
        s=50
    )

plt.title("NBA Player Archetypes - Hierarchical Clustering")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend(title="Cluster")
plt.grid(True)
plt.show()



player_stats.to_csv(
    "nba_player_clusters.csv",
    index=False
)

cluster_profiles.to_csv(
    "nba_cluster_profiles.csv"
)

print("\nResults saved successfully!")

print("1. nba_player_clusters.csv")
print("2. nba_cluster_profiles.csv")
