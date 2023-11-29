import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm

num_classes = 6  # Number of classes

class GrayscaleToRGB(object):
    def __call__(self, image):
        return image.convert('RGB')

for class_num in range(1, num_classes + 1):
    print(f'Training class {class_num}')
    # Load the pre-trained ResNet-18 models
    model = models.mobilenet_v2(pretrained=True)

    # Modify the models's final fully connected layer to match your number of classes
    model.classifier[1] = nn.Linear(model.last_channel, 2)

    # Set up data transforms for training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((20, 20)),  # Resize to the input size expected by mobileNet
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale images
    ])

    # Load the data for the current class
    data_path = f'data/seperate_data/train/{class_num}'
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    # Define a DataLoader for training
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Training loop for the current class
    num_epochs = 10  # You can adjust this
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")
    # Save or export the trained models for this class
    torch.save(model.state_dict(), f'models/mobilenet_model_class_{class_num}.pth')
