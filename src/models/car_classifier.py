import torch.nn as nn

class CarClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super(CarClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 128 * 128)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
