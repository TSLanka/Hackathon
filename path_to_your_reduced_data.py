import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Use an absolute file path to specify the location of reduced_data.csv
data = pd.read_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/reduced_data.csv')

# Apply DBSCAN
# These are just example values for eps and min_samples. Adjust them based on your dataset and requirements.
dbscan = DBSCAN(eps=0.2, min_samples=5)
clusters = dbscan.fit_predict(data)

# Visualize the clusters
plt.scatter(data['PC1'], data['PC2'], c=clusters, cmap='rainbow')
plt.title('DBSCAN Clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid(True)
plt.show()

# Save cluster labels as a CSV file
cluster_labels = pd.DataFrame(data=clusters, columns=['Cluster'])
cluster_labels.to_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/cluster_labels.csv', index=False)
