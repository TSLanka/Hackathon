import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import spacy  # Import spaCy library

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
    return text

# Apply Text Cleaning
data['cleaned_text'] = data['Description'].apply(lambda x: clean_text(x))

# Tokenization using spaCy
data['tokens'] = data['cleaned_text'].apply(lambda x: [token.text for token in nlp(x)])

# Lowercasing
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: x.lower())

# Stopword Removal
stop_words = set(stopwords.words('english'))
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# Stemming
stemmer = PorterStemmer()
data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

# Save the processed data to a new CSV file
data.to_csv("processed_dataT.csv", index=False)

print("Data preprocessing completed and saved to 'processed_dataT.csv'")
