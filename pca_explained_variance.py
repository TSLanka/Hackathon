import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def plot_explained_variance(X):
    # Apply PCA without reducing the number of components first to get all explained variances
    pca_full = PCA()
    pca_full.fit(X)

    # Plotting the explained variance
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10,10))

    # Individual explained variance
    plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center', label='Individual explained variance')

    # Cumulative explained variance
    plt.step(range(len(cumulative_variance)), cumulative_variance, where='mid',label='Cumulative explained variance')

    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def perform_pca_and_save_reduced_data():
    # Load your TF-IDF features
    data = pd.read_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/tfidf_features.csv')

    # Assuming the last column might be labels or some identifier. If that's the case, drop it. If not, adjust accordingly.
    X = data.drop(columns=[data.columns[-1]])

    # Plot the explained variance to decide on the number of components
    plot_explained_variance(X)

    # Initialize PCA with the desired number of components
    n_components = 2  # Replace with the number of components you chose
    pca = PCA(n_components=n_components)

    # Apply PCA to the data
    reduced_data = pca.fit_transform(X)

    # Create a DataFrame for the reduced data
    columns = [f'PC{i+1}' for i in range(n_components)]  # Column names for the principal components
    reduced_df = pd.DataFrame(data=reduced_data, columns=columns)

    # Save the reduced data to a new CSV file
    reduced_df.to_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/reduced_data.csv', index=False)

if __name__ == "__main__":
    perform_pca_and_save_reduced_data()
