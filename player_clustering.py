"""Objective 2: exploratory K-means clustering of NBA player-seasons.
Run from the repository root: python player_clustering.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA = Path('data/games_details_cleaned.csv')
OUT = Path('clustering_results')
OUT.mkdir(exist_ok=True)

columns = ['PLAYER_ID', 'PLAYER_NAME', 'SEASON', 'PTS', 'AST', 'REB',
           'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA']
df = pd.read_csv(DATA, usecols=columns)
for col in columns[3:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# One row per player per season; count appearances and sum made/attempted shots.
group = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'SEASON'], as_index=False)
players = group.agg(games=('PTS', 'count'), pts=('PTS', 'mean'),
                    ast=('AST', 'mean'), reb=('REB', 'mean'),
                    fgm=('FGM', 'sum'), fga=('FGA', 'sum'),
                    fg3m=('FG3M', 'sum'), fg3a=('FG3A', 'sum'),
                    ftm=('FTM', 'sum'), fta=('FTA', 'sum'))
players = players[players['games'] >= 10].copy()
# Season-wide shooting accuracy, not the average of game percentages.
for made, attempted, result in [('fgm', 'fga', 'fg_pct'),
                                ('fg3m', 'fg3a', 'fg3_pct'),
                                ('ftm', 'fta', 'ft_pct')]:
    players[result] = (players[made] / players[attempted].replace(0, float('nan'))).fillna(0)

features = ['pts', 'ast', 'reb', 'fg_pct', 'fg3_pct', 'ft_pct']
players = players.dropna(subset=features).copy()
X = StandardScaler().fit_transform(players[features])

ks = range(2, 8)
inertias, silhouettes = [], []
for k in ks:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X, labels, sample_size=min(5000, len(X)), random_state=42))
    print(f'k={k}: inertia={model.inertia_:.1f}, silhouette={silhouettes[-1]:.3f}')

best_k = list(ks)[silhouettes.index(max(silhouettes))]
players['cluster'] = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(X)
players.to_csv(OUT / 'player_season_clusters.csv', index=False)
summary = players.groupby('cluster')[features + ['games']].mean().round(3)
summary.insert(0, 'player_seasons', players.groupby('cluster').size())
summary.to_csv(OUT / 'cluster_summary.csv')
print(f'\nSelected k={best_k} by sampled silhouette score; player-seasons={len(players)}')
print(summary.to_string())

plt.figure()
plt.plot(list(ks), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('K-means elbow curve')
plt.tight_layout()
plt.savefig(OUT / 'elbow_curve.png', dpi=160)
plt.close()

plt.figure()
plt.plot(list(ks), silhouettes, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette score')
plt.title('K-means silhouette scores')
plt.tight_layout()
plt.savefig(OUT / 'silhouette_scores.png', dpi=160)
plt.close()
