import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  # Import K-Means clustering
from sklearn.svm import LinearSVC

import torch
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(texts):
    # Convert texts to BERT embeddings
    embeddings = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())
    return embeddings



# Load spaCy English Model and NLTK Stopwords
nlp = spacy.load("en_core_web_sm")
nltk.download('stopwords', quiet=True)

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ''.join([char for char in text if not char.isdigit()])
    return text.lower()

def preprocess_data(data):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    data['cleaned_text'] = data['Description'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
    data['tokens'] = data['cleaned_text'].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])
    
    return data

# Load the labeled data (merged with K-Means pseudo-labels) and preprocess it
labeled_data_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\labeled_data.csv"
labeled_data = pd.read_csv(labeled_data_path)

# Add the Cluster column from kmeans_clusters_output.csv
kmeans_clusters_data = pd.read_csv(r"C:/Users/TSLanka/Documents/GitHub/Hackathon/kmeans_clusters_output.csv")
labeled_data['Cluster'] = kmeans_clusters_data['Cluster']

labeled_data = preprocess_data(labeled_data)


# Get BERT embeddings for labeled data
X_labeled_embeddings = get_bert_embeddings(labeled_data['cleaned_text'].tolist())

# Create a K-Means clustering model with explicit n_init value
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
kmeans.fit(X_labeled_embeddings)


# Load and preprocess the unlabeled data
unlabeled_data_path = r"C:/Users/TSLanka/Documents/GitHub/Hackathon/merged_data.csv"
unlabeled_data = pd.read_csv(unlabeled_data_path)
unlabeled_data = preprocess_data(unlabeled_data)


# Get BERT embeddings for unlabeled data
X_unlabeled_embeddings = get_bert_embeddings(unlabeled_data['cleaned_text'].tolist())

# Cluster the unlabeled data using K-Means
cluster_labels = kmeans.predict(X_unlabeled_embeddings)

vectorizer = TfidfVectorizer()  
X_labeled = vectorizer.fit_transform(labeled_data['cleaned_text'])
y_labeled = labeled_data['Cluster']

# Create a K-Means clustering model with explicit n_init value
kmeans = KMeans(n_clusters=10, random_state=0, n_init=20)  # Change the number of clusters and n_init value here
kmeans.fit(X_labeled)


# Load and preprocess the unlabeled data
unlabeled_data_path = r"C:/Users/TSLanka/Documents/GitHub/Hackathon/merged_data.csv"
unlabeled_data = pd.read_csv(unlabeled_data_path)
unlabeled_data = preprocess_data(unlabeled_data)

# Transform the unlabeled data using the vectorizer fitted on labeled data
X_unlabeled = vectorizer.transform(unlabeled_data['cleaned_text'])

# Cluster the unlabeled data using K-Means
cluster_labels = kmeans.predict(X_unlabeled)
unlabeled_data['Cluster_Label'] = cluster_labels

# Save the clustered data
unlabeled_data.to_csv('C:/Users/TSLanka/Documents/GitHub/Hackathon/clustered_data.csv', index=False)

