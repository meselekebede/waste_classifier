import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os


# === CONFIGURATION ===
MODEL_PATH = "best_resnet_model.pth"
IMAGE_SIZE = 224
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# Define preprocessing pipeline
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def build_resnet_model(num_classes=len(CLASS_NAMES)) -> models.ResNet:
    """
    Builds and returns a ResNet-34 model with custom final layer.
    """
    model = models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(model_path: str = MODEL_PATH) -> torch.nn.Module:
    """
    Loads the trained model weights and sets the model to evaluation mode.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    return model.to(device)


def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocesses the input image and returns a batched tensor.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

    image = Image.open(image_path).convert("RGB")
    return TRANSFORMS(image).unsqueeze(0)  # Add batch dimension


def predict_with_confidence(model: torch.nn.Module, image_tensor: torch.Tensor) -> tuple[str, dict]:
    """
    Predicts class and returns confidence scores for each waste category.
    """
    device = next(model.parameters()).device  # Get model's device
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_idx = np.argmax(probs)
    prediction = CLASS_NAMES[predicted_idx]
    confidence_dict = {cls.capitalize(): float(probs[i]) for i, cls in enumerate(CLASS_NAMES)}
    return prediction, confidence_dict