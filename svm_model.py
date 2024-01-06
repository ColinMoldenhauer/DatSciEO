import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from utils.dataset import TreeClassifPreprocessedDataset

# Seed for reproducibility
seed = 42
np.random.seed(seed)

# Load data
data = []
labels = []

# Specify data folder direction
data_dir = r'C:\Users\User\desktop\DatSciEO-class\data\1123_delete_nan_samples_nanmean_B2'
ds = TreeClassifPreprocessedDataset(data_dir)

for data_, label_ in ds:
    data.append(data_)
    labels.append(label_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)

# Create a pipeline with SVM and StandardScaler
model = make_pipeline(StandardScaler(), SVC(random_state=seed))

# Train the classifier
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
conf_mat = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
#print("Confusion Matrix:\n", conf_mat)