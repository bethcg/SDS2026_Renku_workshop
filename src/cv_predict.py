import torch
from PIL import Image

# Load the trained weights
model.load_state_dict(torch.load('outputs/models/asce_classifier.pth'))
model.eval()

# Predict a new image
def predict_xray(image_path):
    img = transform(Image.open(image_path))
    output = model(img.unsqueeze(0))
    _, predicted = torch.max(output, 1)
    classes = ['Bacteria', 'Normal', 'Virus']
    return classes[predicted.item()]
