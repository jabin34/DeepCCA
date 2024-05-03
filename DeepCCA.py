import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
# Load pathway data from CSV file
pathways_data = pd.read_csv("kegg_legacy_ensembl.csv")

# Extract gene symbols and ensembl IDs
gene_symbols = []
ensembl_ids = []
for row in pathways_data.itertuples():
    gene_symbols.extend(eval(row.geneSymbols))
    ensembl_ids.extend(eval(row.ensembl))

# Create a set of unique gene symbols and ensembl IDs
unique_gene_symbols = list(set(gene_symbols))
unique_ensembl_ids = list(set(ensembl_ids))

# Create a dictionary to map gene symbols and ensembl IDs to indices
gene_symbol_to_index = {gene_symbol: idx for idx, gene_symbol in enumerate(unique_gene_symbols)}
ensembl_id_to_index = {ensembl_id: idx for idx, ensembl_id in enumerate(unique_ensembl_ids)}

# Create a feature matrix where each row represents a pathway and each column represents a gene symbol or ensembl ID
num_pathways = len(pathways_data)
num_gene_symbols = len(unique_gene_symbols)
num_ensembl_ids = len(unique_ensembl_ids)
feature_matrix = torch.zeros(num_pathways, num_gene_symbols + num_ensembl_ids)

for i, row in enumerate(pathways_data.itertuples()):
    for gene_symbol in eval(row.geneSymbols):
        feature_matrix[i, gene_symbol_to_index[gene_symbol]] = 1
    for ensembl_id in eval(row.ensembl):
        feature_matrix[i, num_gene_symbols + ensembl_id_to_index[ensembl_id]] = 1

# Standardize features
scaler = StandardScaler()
pathways_tensor = scaler.fit_transform(feature_matrix)

# Convert scaled features to PyTorch tensor
pathways_tensor = torch.FloatTensor(pathways_tensor)

# Split data into training and validation sets
X_train, X_val = train_test_split(pathways_tensor, test_size=0.2, random_state=42)

# Define deep neural network architecture for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(FeatureExtractor, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

# Define DCCA model
class DCCA(nn.Module):
    def __init__(self, input_size1, input_size2, hidden_size):
        super(DCCA, self).__init__()
        self.feature_extractor1 = FeatureExtractor(input_size1, hidden_size)
        self.feature_extractor2 = FeatureExtractor(input_size2, hidden_size)
        self.correlation_layer = nn.Linear(hidden_size, 1)

    def forward(self, x1, x2):
        features1 = self.feature_extractor1(x1)
        features2 = self.feature_extractor2(x2)
        correlation = torch.sigmoid(self.correlation_layer(features1 * features2))
        return correlation

# Initialize DCCA model
input_size = pathways_tensor.shape[1]
hidden_size = 64
model = DCCA(input_size, input_size, hidden_size)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert validation data to DataLoader
val_dataset = TensorDataset(X_val, X_val)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Convert training data to DataLoader
train_dataset = TensorDataset(X_train, X_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training loop with early stopping
num_epochs = 100
best_val_loss = float('inf')
patience = 5  # Number of epochs to wait if validation loss doesn't decrease
early_stopping_counter = 0
for epoch in range(num_epochs):
    model.train()
    for data1, data2 in train_dataloader:
        # Forward pass
        correlation = model(data1, data2)
        # Compute loss (minimize negative correlation)
        loss = -torch.mean(correlation)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data1, data2 in val_dataloader:  # Change train_dataloader to val_dataloader
            correlation = model(data1, data2)
            val_loss += -torch.mean(correlation).item()
        val_loss /= len(val_dataloader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping!")
            break


# Split data into training and test sets
X_train, X_test = train_test_split(pathways_tensor, test_size=0.2, random_state=42)



# Convert test data to DataLoader
test_dataset = TensorDataset(X_test, X_test)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Evaluate the model on the test set
model.eval()
test_loss = 0.0
with torch.no_grad():
    for data1, data2 in test_dataloader:
        correlation = model(data1, data2)
        test_loss += -torch.mean(correlation).item()
    test_loss /= len(test_dataloader)

print(f"Test Loss: {test_loss}")



