import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CarsDataset(Dataset):
    def __init__(self, annotations_df: pd.DataFrame, transform=None):
        self.annotations_df = annotations_df
        self.transform = transform

    def __len__(self):
        return len(self.annotations_df)

    def __getitem__(self, idx: int):
        image_info = self.annotations_df.iloc[idx]
        image_path = image_info['image_path']
        image = Image.open(image_path).convert('RGB')

        # Extract bounding box coordinates
        bbox = [
            image_info['x_min'],
            image_info['y_min'],
            image_info['x_max'],
            image_info['y_max']
        ]

        # Extract class label
        class_label = image_info['class']

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)

        return image, bbox, class_label
