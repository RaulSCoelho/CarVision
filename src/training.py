import torch
import torchvision
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Define paths and parameters
train_cars_dataset_path = 'src/data/cars_train/train'
test_cars_dataset_path = 'src/data/cars_train/test'
mean = [0.4708, 0.4602, 0.4550]
std = [0.2593, 0.2584, 0.2634]

train_cars_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

test_cars_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

# Load the dataset
train_cars_dataset = ImageFolder(root=train_cars_dataset_path, transform=train_cars_transforms)
test_cars_dataset = ImageFolder(root=test_cars_dataset_path, transform=test_cars_transforms)

def show_transformed_images(dataset: ImageFolder, num_images: int = 6):
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)
    batch = next(iter(loader))
    images, labels = batch

    print(f"Labels: {labels}")
    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()

show_transformed_images(train_cars_dataset)

# Load the dataset
train_cars_loader = DataLoader(train_cars_dataset, batch_size=32, shuffle=True)
test_cars_loader = DataLoader(test_cars_dataset, batch_size=32, shuffle=False)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_checkpoint(model: torch.nn.Module, epoch: int, optimizer: torch.optim.Optimizer, best_acc: float):
    state = {
        'model': model.state_dict(),
        'epoch': epoch,
        'optimizer': optimizer.state_dict(),
        'best_acc': best_acc
    }
    torch.save(state, 'best_cars_model_checkpoint.pth.tar')

def train_cars_nn(model: torch.nn.Module, train_loader: DataLoader, test_loader: DataLoader, criterion: torch.nn.CrossEntropyLoss, optimizer: torch.optim.Optimizer, n_epochs: int):
    device = get_device()
    best_acc = 0

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * running_correct / total

        print(f"    Training: Predicted {running_correct} of {total} images correctly (Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.2f}%)")

        eval_acc = evaluate_model(model, test_loader)

        if eval_acc > best_acc:
            best_acc = eval_acc
            save_checkpoint(model, epoch + 1, optimizer, best_acc)

    return model

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader):
    model.eval()
    device = get_device()
    predicted_correctly = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            total += labels.size(0)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_correctly += (predicted == labels).sum().item()

    accuracy = 100 * predicted_correctly / total
    print(f"    Testing: Predicted {predicted_correctly} of {total} images correctly (Acc: {accuracy:.2f}%)")

    return accuracy

resnet18_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
n_features = resnet18_model.fc.in_features
n_classes = len(train_cars_dataset.classes)
resnet18_model.fc = torch.nn.Linear(n_features, n_classes)
device = get_device()
resnet18_model = resnet18_model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

train_cars_nn(resnet18_model, train_cars_loader, test_cars_loader, loss_fn, optimizer, n_epochs=20)

checkpoint = torch.load('best_cars_model_checkpoint.pth.tar')

resnet18_model = torchvision.models.resnet18()
n_features = resnet18_model.fc.in_features
n_classes = len(train_cars_dataset.classes)
resnet18_model.fc = torch.nn.Linear(n_features, n_classes)
resnet18_model.load_state_dict(checkpoint['model'])

torch.save(resnet18_model, 'best_cars_model.pth')
