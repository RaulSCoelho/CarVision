import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.cars_dataset import CarsDataset
from models.car_classifier import CarClassifier
from utils.data import mat_to_list

# Define paths and parameters
cars_train_dir = 'src/data/cars_train'
cars_train_csv_dir = 'src/data/car_dataset_train.csv'
cars_mat_file_dir = 'src/data/cars_annos.mat'
batch_size = 32

classes_df = mat_to_list(cars_mat_file_dir, 'class_names')
num_classes=len(classes_df)

annotations_df = pd.read_csv(cars_train_csv_dir)
annotations_df['class'] -= 1
annotations_df['image_path'] = annotations_df['image_path'].apply(lambda x: os.path.join(cars_train_dir, x))

# Shuffle the indices of the dataset
indices = np.arange(len(annotations_df))
np.random.shuffle(indices)

# Define the split ratio
split_ratio = 0.8  # 80% for training, 20% for validation
# Calculate the split index
split_index = int(len(annotations_df) * split_ratio)
# Split the indices into training and validation sets
train_indices = indices[:split_index]
val_indices = indices[split_index:]

# Create training and validation DataFrames
train_df = annotations_df.iloc[train_indices]
val_df = annotations_df.iloc[val_indices]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Create training dataset and DataLoader
train_dataset = CarsDataset(train_df, transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Create validation dataset and DataLoader
val_dataset = CarsDataset(val_df, transform)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Create model instance
car_classifier = CarClassifier(num_classes)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(car_classifier.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    car_classifier.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = car_classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# Evaluation loop
car_classifier.eval()
with torch.no_grad():
    total_correct = 0
    total_samples = 0
    for images, labels in val_dataloader:
        outputs = car_classifier(images)
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Validation Accuracy: {accuracy}')
