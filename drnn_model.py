import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, confusion_matrix
from utils.dataset import TreeClassifPreprocessedDataset

# Seed for reproducibility
seed = 42
torch.manual_seed(seed)
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

# Convert data and labels to PyTorch tensors
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)

# Convert list of numpy arrays to a single numpy array
X_train = np.stack(X_train)
X_test = np.stack(X_test)

# Convert data to PyTorch tensors
X_train, X_test, y_train, y_test = (
    torch.tensor(X_train).float(),
    torch.tensor(X_test).float(),
    torch.tensor(y_train),
    torch.tensor(y_test),
)

# Define the deep recursive neural network model
class RecursiveNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RecursiveNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Add an additional dimension for the sequence length
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use the output of the last time step
        return out
    
# Initialize the model, loss function, and optimizer
input_size = X_train.shape[1]  # Input size is the depth of your data
hidden_size = 64
output_size = len(np.unique(labels))  # Number of unique classes
num_layers = 2

model = RecursiveNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
batch_size = 16

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs, 1)

# Convert predicted labels to numpy array
y_pred = predicted.numpy()

# Calculate various metrics
accuracy = accuracy_score(y_test.numpy(), y_pred)
kappa = cohen_kappa_score(y_test.numpy(), y_pred)
precision = precision_score(y_test.numpy(), y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test.numpy(), y_pred, average='weighted')
conf_mat = confusion_matrix(y_test.numpy(), y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Cohen's Kappa: {kappa:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
#print("Confusion Matrix:\n", conf_mat)