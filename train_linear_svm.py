import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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
labeled_data_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv"
labeled_data = pd.read_csv(labeled_data_path)

# Add the Cluster column from kmeans_clusters_output.csv
kmeans_clusters_data = pd.read_csv(r"C:/Users/TSLanka/Documents/GitHub/Hackathon/kmeans_clusters_output.csv")
labeled_data['Cluster'] = kmeans_clusters_data['Cluster']

labeled_data = preprocess_data(labeled_data)

# Apply TF-IDF Vectorization on labeled data
vectorizer = TfidfVectorizer()  
X_labeled = vectorizer.fit_transform(labeled_data['cleaned_text'])
y_labeled = labeled_data['Cluster']

# Train the SVM on labeled data, explicitly setting dual to 'auto'
clf = LinearSVC(random_state=0, max_iter=10000, dual='auto')
clf.fit(X_labeled, y_labeled)

# Load and preprocess the unlabeled data
unlabeled_data_path = r"C:/Users/TSLanka/Documents/GitHub/Hackathon/merged_data.csv"
unlabeled_data = pd.read_csv(unlabeled_data_path)
unlabeled_data = preprocess_data(unlabeled_data)

# Transform the unlabeled data using the vectorizer fitted on labeled data
X_unlabeled = vectorizer.transform(unlabeled_data['cleaned_text'])

# Predict labels for the unlabeled data
predicted_labels = clf.predict(X_unlabeled)
unlabeled_data['Predicted_Label'] = predicted_labels

# Save the labeled data
unlabeled_data.to_csv('labeled_data.csv', index=False)
