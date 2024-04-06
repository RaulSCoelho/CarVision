import os
import random
from typing import List
import torch
from torchvision import transforms
from matplotlib import pyplot as plt
from utils.data import mat_to_list
from PIL import Image

cars_model = torch.load('best_cars_model.pth')
classes_df = mat_to_list('src/data/cars_annos.mat', 'class_names')

mean = [0.4708, 0.4602, 0.4550]
std = [0.2593, 0.2584, 0.2634]

cars_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model: torch.nn.Module, image_transforms: transforms.Compose, image_paths: str, classes: List[str]):
    model = model.eval()

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    images = {image_path: Image.open(image_path) for image_path in image_paths}

    num_images = len(images)
    cols = min(3, num_images)  # Limit to at most 3 columns
    rows = (num_images + cols - 1) // cols  # Calculate required rows

    total_width = sum(image.width for image in images.values())
    total_height = max(image.height for image in images.values())

    _, axes = plt.subplots(rows, cols, figsize=(total_width // 100, total_height // 100))

    for i, (image_path, image) in enumerate(images.items()):
        total_width += image.width
        total_height += image.height
        image_transformed = image_transforms(image).float()
        image_transformed = image_transformed.unsqueeze(0)

        output = model(image_transformed)
        _, predicted = torch.max(output.data, 1)

        predicted_class = classes[predicted.item()]
        print(f"{image_path}: {predicted_class}")

        if num_images == 1:
            axes.imshow(image)  # Use flattened axes for efficient indexing
            axes.set_title(predicted_class)
            axes.axis('off')
        else:
            axes.flat[i].imshow(image)  # Use flattened axes for efficient indexing
            axes.flat[i].set_title(predicted_class)
            axes.flat[i].axis('off')

    plt.tight_layout()
    plt.show()

# Get a random image path from the cars_test directory
cars_test_dir = 'src/data/cars_test'
image_dirs = os.listdir(cars_test_dir)
image_paths = [os.path.join(cars_test_dir, dir) for dir in random.sample(image_dirs, 6)]

classify(cars_model, cars_transforms, image_paths, classes_df)
