
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import spacy  # Import spaCy library
from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF Vectorizer

# Ensure you've downloaded the necessary NLTK data
nltk.download('stopwords')

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
data_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv"  # Replace with the path to your file
data = pd.read_csv(data_path)

# Text Cleaning Function
def clean_text(text):
    # Remove punctuations
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    # Convert to lowercase
    text = text.lower()
    return text

# Apply Text Cleaning
data['cleaned_text'] = data['Description'].apply(lambda x: clean_text(x))

# Stopword Removal
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Tokenization and Lemmatization using spaCy
data['tokens'] = data['cleaned_text'].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['cleaned_text'])
feature_names = vectorizer.get_feature_names_out()
tfidf_data = pd.DataFrame(X.toarray(), columns=feature_names)

# Save the TF-IDF features to a new CSV file
tfidf_data.to_csv("tfidf_features.csv", index=False)

# Save the processed data to another CSV file
data.to_csv("processed_data_improved.csv", index=False)

print("Data preprocessing and TF-IDF feature extraction completed. Features saved to 'tfidf_features.csv'")
