import torch
from torchvision import transforms
from utils.data import mat_to_list
import PIL.Image as Image

cars_model = torch.load('best_cars_model.pth')
classes_df = mat_to_list('src/data/cars_annos.mat', 'class_names')

mean = [0.4708, 0.4602, 0.4550]
std = [0.2593, 0.2584, 0.2634]

cars_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model: torch.nn.Module, image_transforms: transforms.Compose, image_path: str, classes: list[str]):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)

    print(f"Predicted class: {predicted.item()}")
