import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# 1. Setup paths
# Replace input dir with the data stemming from Zenodo: https://zenodo.org/records/18241541 
DATA_DIR = 'data/chest_xray'  # This is where the Zenodo connector should mount

  
# Replace output dir with your output data folder with read/write access
OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_pneumonia_model():
    # 2. Image transformations for PNG Chest X-rays
    # Pneumonia images are often grayscale; we convert to RGB for standard ResNet compatibility
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3), 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Data loading
    # Expected structure: data/chest_xray/train/NORMAL and data/chest_xray/train/PNEUMONIA
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} not found. Ensure Zenodo connector is active.")
        return

    train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

      # 4. Model definition (transfer learning for speed)
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) # Binary: Normal vs Pneumonia
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 5. Training snippet
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on {len(train_dataset)} PNG images...")
    model.train()
    
    # We run only 1 epoch for the workshop demonstration to save time
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        break # We break early just to prove the pipeline works

    # 6. Save the outcome
    model_path = os.path.join(OUTPUT_DIR, 'pneumonia_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_pneumonia_model()
