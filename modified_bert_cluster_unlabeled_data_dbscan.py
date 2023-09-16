import pandas as pd
import nltk
import string
import spacy
from transformers import BertTokenizer, BertModel
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Lazy Load BERT model and tokenizer
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def get_bert_embeddings(texts, tokenizer, model, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.extend(outputs.last_hidden_state.mean(dim=1).squeeze().tolist())
    return embeddings

# Lazy load spaCy model and NLTK stopwords
def load_spacy_nltk():
    nlp = spacy.load("en_core_web_sm")
    nltk.download('stopwords', quiet=True)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    return nlp, stop_words

def clean_text(text):
    text = ''.join([char for char in text if char not in string.punctuation and not char.isdigit()])
    return text.lower()

def preprocess_data(data, nlp, stop_words):
    data['cleaned_text'] = data['Description'].apply(clean_text)
    data['cleaned_text'] = data['cleaned_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    data['tokens'] = data['cleaned_text'].apply(lambda x: [token.lemma_ for token in nlp(x) if not token.is_stop])
    return data

def load_data(file_path):
    return pd.read_csv(file_path)

def cluster_embeddings(embeddings, eps=0.5, min_samples=5, metric='euclidean'):
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(embeddings)
    return clustering.labels_

def visualize_clusters(embeddings, labels):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=300, random_state=23)
    new_values = tsne_model.fit_transform(embeddings)
    x = [value[0] for value in new_values]
    y = [value[1] for value in new_values]
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i], color=plt.cm.rainbow(labels[i] / len(set(labels))))
        plt.annotate(str(labels[i]), xy=(x[i], y[i]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()

def save_clustered_data(data, save_path="clustered_data.csv"):
    data.to_csv(save_path, index=False)
    print(f"Clustered data saved to {save_path}")

def main(file_path=r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv", save_path="clustered_data.csv"):
    data = load_data(r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv")
    nlp, stop_words = load_spacy_nltk()
    data = preprocess_data(data, nlp, stop_words)
    tokenizer, model = load_bert()
    embeddings = get_bert_embeddings(data['cleaned_text'].tolist(), tokenizer, model)
    labels = cluster_embeddings(embeddings)
    data['cluster'] = labels
    visualize_clusters(embeddings, labels)
    save_clustered_data(data, save_path)
    return data
