from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

class CarsDataset(Dataset):
    def __init__(self, cars_data_dir, transform=None):
        self.data = ImageFolder(cars_data_dir, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes
