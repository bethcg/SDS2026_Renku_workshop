import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Path to the unzipped ASCE folder
DATA_DIR = '/home/renku/work/data/ASCE_Enhanced_Renamed'
OUTPUT_DIR = '/home/renku/work/data/outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_asce_model():
    # Note: We keep 3 channels (RGB) because colors represent anatomical segmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset (ImageFolder will automatically pick up Normal, Virus, Bacteria)
    if not os.path.exists(DATA_DIR):
        print(f"Error: Path {DATA_DIR} not found.")
        return

    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    
    # Simple split for the workshop (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_ds, _ = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Use ResNet18 - Adjusting for 3 classes
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) # Normal, Viral, Bacterial
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on ASCE Enhanced Data: {full_dataset.classes}")
    model.train()
    
    # Workshop Demo: Run for a few batches to prove completion
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i >= 5: break # End early for demo purposes

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/asce_classifier.pth")
    print("Model saved to outputs/models/asce_classifier.pth")

if __name__ == "__main__":
    train_asce_model()
