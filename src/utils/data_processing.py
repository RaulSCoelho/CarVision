import torch
from torchvision import datasets, transforms

def load_data(data_dir):
    """
    Load data from the specified directory using torchvision's ImageFolder.

    Args:
    - data_dir (str): Path to the directory containing image data.

    Returns:
    - dataset (torch.utils.data.Dataset): Image dataset loaded using ImageFolder.
    """
    dataset = datasets.ImageFolder(root=data_dir)
    return dataset

def preprocess_data(dataset, image_size, batch_size, shuffle=True):
    """
    Pre-process the image dataset.

    Args:
    - dataset (torch.utils.data.Dataset): Image dataset.
    - image_size (tuple): Desired image size (width, height).
    - batch_size (int): Batch size for data loader.
    - shuffle (bool, optional): Whether to shuffle the data. Default is True.

    Returns:
    - data_loader (torch.utils.data.DataLoader): DataLoader object for pre-processed data.
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Apply transformations to the dataset
    dataset = dataset.transform(transform)

    # Create DataLoader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True  # If using CUDA, pin memory for faster data transfer
    )

    return data_loader
