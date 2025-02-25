import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Define transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load dataset
dataset = datasets.ImageFolder(root='HW5\AA_JHU_3D_V2_28x28-20250225T022212Z-001\AA_JHU_3D_V2_28x28', transform=transform)

# Split into training and test sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

labels_list = [label for _, label in test_dataset]
print("Class distribution in test set:", Counter(labels_list))

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the CNN model
class AminoAcidCNN(nn.Module):
    def __init__(self):
        super(AminoAcidCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)  # Input: 1 channel, output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Input: 32 channels, output: 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)  # Input: 64 channels, output: 128 channels
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 3 * 3, 512)  # Input: Flattened output of the conv layers, output: 512
        self.fc2 = nn.Linear(512, 20)  # Output layer: 20 classes (one for each amino acid)
        
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 128 * 3 * 3)  # Flatten the tensor before feeding it to fully connected layers
        
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate model, loss function, and optimizer
model = AminoAcidCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Train and evaluate
train_model(model, train_loader, criterion, optimizer, epochs=10)
evaluate_model(model, test_loader)

def get_misclassified_images(model, test_loader, dataset):
    model.eval()
    correct_counts = np.zeros(20)
    total_counts = np.zeros(20)
    misclassified_images = []
    correctly_classified_images = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(labels)):
                total_counts[labels[i].item()] += 1
                if predicted[i] == labels[i]:
                    correct_counts[labels[i].item()] += 1
                    correctly_classified_images.append((images[i], labels[i]))
                else:
                    misclassified_images.append((images[i], labels[i], predicted[i]))

    # Prevent division by zero
    accuracy_per_class = np.divide(correct_counts, total_counts, where=total_counts != 0)
    accuracy_per_class = np.nan_to_num(accuracy_per_class, nan=0.0)  # Convert NaNs to 0

    best_class = np.argmax(accuracy_per_class)
    worst_class = np.argmin(accuracy_per_class)

    print(f"Most correctly classified amino acid: {dataset.classes[best_class]}")
    print(f"Least correctly classified amino acid: {dataset.classes[worst_class]}")

    return correctly_classified_images, misclassified_images, best_class, worst_class

correct_images, misclassified_images, best_class, worst_class = get_misclassified_images(model, test_loader, dataset)

# Show one correctly classified and one misclassified image
if misclassified_images:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(correct_images[0][0].squeeze(), cmap='gray')
    ax[0].set_title(f"Correctly classified: {dataset.classes[correct_images[0][1]]}")
    ax[1].imshow(misclassified_images[0][0].squeeze(), cmap='gray')
    ax[1].set_title(f"Misclassified as: {dataset.classes[misclassified_images[0][2]]}")
    plt.show()
else:
    print("No misclassified images found.")
