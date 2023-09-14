import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load spaCy English Model and NLTK Stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords', quiet=True)

# Data Preprocessing Functions
def clean_text(text):
    # Remove punctuations and numbers, and lowercase the text
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    return text.lower()

def preprocess_data(data):
    # Apply text cleaning, lowercasing, stopword removal, stemming, and lemmatization
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    data['cleaned_text'] = data['Description'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    data['tokens'] = data['cleaned_text'].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])
    
    return data

# Feature Extraction Function
def extract_features(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_data = pd.DataFrame(X.toarray(), columns=feature_names)
    
    return tfidf_data

# PCA Function
def apply_pca(tfidf_data):
    pca = PCA()
    reduced_data = pca.fit_transform(tfidf_data)
    explained_variance = pca.explained_variance_ratio_
    
    return reduced_data, explained_variance

# K-Means Clustering Function
def kmeans_clustering(data, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clusters = kmeans.fit_predict(data)
    return clusters

# Unified Algorithm
def enhanced_unified_algorithm(data_path):
    # Data Collection
    data = pd.read_csv(data_path)
    
    # Data Preprocessing
    preprocessed_data = preprocess_data(data)
    
    # Feature Extraction
    tfidf_data = extract_features(preprocessed_data)
    
    # Dimensionality Reduction
    reduced_data, explained_variance = apply_pca(tfidf_data)
    
    # Plotting the explained variance for PCA
    cumulative_variance = np.cumsum(explained_variance)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(len(cumulative_variance)), cumulative_variance, where='mid', label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # K-Means Clustering
    clusters = kmeans_clustering(reduced_data)
    
    # Saving and Displaying Results
    output_df = pd.DataFrame(data=reduced_data[:, :2], columns=['PC1', 'PC2'])

    output_df['Cluster'] = clusters
    output_df.to_csv('kmeans_clusters_output.csv', index=False)
    
    # Displaying the number of data points in each cluster
    cluster_counts = output_df['Cluster'].value_counts()
    print("Number of data points in each cluster:")
    print(cluster_counts)
    
    plt.scatter(output_df['PC1'], output_df['PC2'], c=clusters, cmap='rainbow')
    plt.title('K-Means Clustering')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.show()
    
    return output_df

# Placeholder for execution on local machine
# Uncomment the line below and provide the path to your dataset to run the algorithm
output_dataframe = enhanced_unified_algorithm(r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions_data.csv")

