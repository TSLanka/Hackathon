import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Load your reduced data (modify the path as needed)
data = pd.read_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/reduced_data.csv')

# Define a range of eps values to explore
eps_values = np.linspace(0.1, 2.0, 20)

# Initialize a list to store silhouette scores
silhouette_scores = []

for eps in eps_values:
    dbscan = DBSCAN(eps=eps, min_samples=5)
    clusters = dbscan.fit_predict(data)
    num_unique_clusters = len(set(clusters))
    if num_unique_clusters > 1:  # Check that there are at least 2 clusters
        silhouette = silhouette_score(data, clusters)
        silhouette_scores.append(silhouette)

# Plot silhouette scores for different eps values
plt.plot(eps_values, silhouette_scores, marker='o')
plt.xlabel('Eps (Neighbor Radius)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for DBSCAN')
plt.grid(True)
plt.show()

# Find the index of the maximum silhouette score
best_eps_index = np.argmax(silhouette_scores)

# Print the best epsilon and its corresponding silhouette score
best_eps = eps_values[best_eps_index]
best_silhouette_score = silhouette_scores[best_eps_index]
print(f"Best Eps: {best_eps}, Best Silhouette Score: {best_silhouette_score}")
