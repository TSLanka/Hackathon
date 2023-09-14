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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load spaCy English Model and NLTK Stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords', quiet=True)

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

def extract_features(data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['cleaned_text'])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_data = pd.DataFrame(X.toarray(), columns=feature_names)
    
    return tfidf_data

def apply_pca(tfidf_data):
    pca = PCA()
    reduced_data = pca.fit_transform(tfidf_data)
    explained_variance = pca.explained_variance_ratio_
    
    return reduced_data, explained_variance

def kmeans_clustering(data, n_clusters=10):
    # [The entire kmeans_clustering function logic]

def enhanced_unified_algorithm(input_path):
    # [The entire enhanced_unified_algorithm function]

# Load the labeled and unlabeled data
labeled_data_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv"
unlabeled_data_path = r"C:/Users/TSLanka/Documents/GitHub/Hackathon/merged_data.csv"

labeled_data = pd.read_csv(labeled_data_path)
unlabeled_data = pd.read_csv(unlabeled_data_path)

# Preprocess the labeled and unlabeled data
labeled_data = preprocess_data(labeled_data)
unlabeled_data = preprocess_data(unlabeled_data)

# Apply K-Means clustering and SVM training on labeled data
output_dataframe = enhanced_unified_algorithm(labeled_data_path)

# Extract features for the unlabeled data
tfidf_data_unlabeled = extract_features(unlabeled_data)

# Transform the new unlabeled data using the same feature representation as labeled data
X_unlabeled = tfidf_data_unlabeled

# Train SVM on labeled data and predict labels for unlabeled data
X = tfidf_data
y = output_dataframe['Cluster']

clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, y)

predicted_labels = clf.predict(X_unlabeled)
unlabeled_data['Predicted_Label'] = predicted_labels

# Save the labeled data
unlabeled_data.to_csv('labeled_data.csv', index=False)
