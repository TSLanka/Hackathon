import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
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

# Preprocess labeled data
labeled_data = preprocess_data(labeled_data)

# Get BERT embeddings for labeled data
X_labeled_embeddings = get_bert_embeddings(labeled_data['cleaned_text'].tolist())

# Calculate perplexity value based on the number of data points
perplexity_value = min(30, labeled_data.shape[0] - 1)

# Applying t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity_value)
X_labeled_embeddings_tsne = tsne.fit_transform(X_labeled_embeddings)

# Using t-SNE transformed embeddings for DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=10)  # Adjust these values based on your data
clusters_labeled = dbscan.fit_predict(X_labeled_embeddings_tsne)

# Save the clustered data
labeled_data['Cluster_Label'] = clusters_labeled
labeled_data.to_csv(r"C:\Users\TSLanka\Documents\GitHub\Hackathon\cluster_data.csv", index=False)
