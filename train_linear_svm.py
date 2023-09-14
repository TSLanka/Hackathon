import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the original data and the cluster labels
data_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\projectdescriptions.csv"
cluster_labels_path = r"C:\Users\TSLanka\Documents\GitHub\Hackathon\kmeans_clusters_output.csv"

data = pd.read_csv(data_path)
cluster_labels = pd.read_csv(cluster_labels_path)

# Create the feature matrix and labels
X = data['Description']
y = cluster_labels['Cluster']

# Initialize and fit the TF-IDF vectorizer using the original dataset
vectorizer = TfidfVectorizer()  # Use the same parameters as before
X = vectorizer.fit_transform(X)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X_train, y_train)

# Evaluate the SVM
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Load the new unlabeled data
unlabeled_data_path = r"C:/Users/TSLanka/Documents/GitHub/Hackathon/merged_data.csv"
unlabeled_data = pd.read_csv(unlabeled_data_path)

# Transform the new data using the already fitted vectorizer
X_unlabeled = vectorizer.transform(unlabeled_data['Description'])

# Predict the labels for the unlabeled data
predicted_labels = clf.predict(X_unlabeled)

# Add the predicted labels to your dataframe
unlabeled_data['Predicted_Label'] = predicted_labels

# Save the labeled data
unlabeled_data.to_csv('labeled_data.csv', index=False)
