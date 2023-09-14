import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Use an absolute file path to specify the location of reduced_data.csv
data = pd.read_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/reduced_data.csv')

# Apply K-Means clustering
# Replace `n_clusters` with the desired number of clusters
n_clusters = 10  # Adjust this as needed
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(data)

# Save the cluster assignments as an output file
output_df = data.copy()
output_df['Cluster'] = clusters
output_df.to_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/kmeans_clusters.csv', index=False)

# Visualize the clusters
plt.scatter(data['PC1'], data['PC2'], c=clusters, cmap='rainbow')
plt.title('K-Means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
