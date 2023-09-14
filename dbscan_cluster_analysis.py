import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load your reduced data (modify the path as needed)
data = pd.read_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/reduced_data.csv')

# Define a range of eps values to explore
eps_values = np.linspace(0.1, 2.0, 20)

# Initialize lists to store silhouette scores
silhouette_scores = []

# Initialize variables to track the best silhouette score and corresponding eps
best_silhouette_score = -1
best_eps = None

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(data)
    if len(set(clusters)) > 1:  # Check that there are at least 2 clusters
        silhouette = silhouette_score(data, clusters)
        silhouette_scores.append(silhouette)
        if silhouette > best_silhouette_score:
            best_silhouette_score = silhouette
            best_eps = eps

# Plot silhouette scores for different eps values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(eps_values, silhouette_scores, marker='o')
plt.xlabel('Eps (Neighbor Radius)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for DBSCAN')
plt.grid(True)

# Plot the number of clusters against eps (Elbow Method)
plt.subplot(1, 2, 2)
plt.plot(eps_values, [len(set(DBSCAN(eps=eps, min_samples=5).fit_predict(data))) for eps in eps_values], marker='o')
plt.xlabel('Eps (Neighbor Radius)')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters for DBSCAN')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print the best epsilon and its corresponding silhouette score
print(f"Best Eps: {best_eps}, Best Silhouette Score: {best_silhouette_score}")
