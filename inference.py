import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt

class GrayscaleToRGB(object):
    def __call__(self, image):
        return image.convert('RGB')

# Define the transform for test data
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((20, 20)),  # Resize to the input size expected by mobileNet
    GrayscaleToRGB(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
])


# Create a custom dataset for your test data
class CustomTestDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.image_files = [os.path.join(root, img) for img in os.listdir(root)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        image = self.transform(image)
        return image


# Load the six models
model_paths = [
    'models/mobilenet_model_class_1.pth',
    'models/mobilenet_model_class_2.pth',
    'models/mobilenet_model_class_3.pth',
    'models/mobilenet_model_class_4.pth',
    'models/mobilenet_model_class_5.pth',
    'models/mobilenet_model_class_6.pth'
]

num_classes = 6
models_dict = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
for i, model_path in enumerate(model_paths):
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    models_dict[i + 1] = model  # Store the models with the corresponding label

# Load the test dataset
test_data = CustomTestDataset(root='data/seperate_data/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

# Make predictions for each image
for i, image in enumerate(test_loader, 1):
    image = image.to(device)  # Move image to the device (CPU/GPU)
    predictions = {}

    for label, model in models_dict.items():
        with torch.no_grad():
            output = model(image)
            softmax = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_label = torch.max(softmax, 0)
            predictions[label] = (predicted_label.item(), confidence.item())

    # Print or store the predictions for this image
    print(f"Predictions for image {i}: {predictions}")
