import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models

BASE_OUTPUT_DIR = Path("d:/ML Lab/report/Output")

FRUITS_13 = [
    'banana', 'custard_apple', 'dragonfruit', 'guava', 'jackfruit',
    'jujube', 'lemon', 'lychee', 'mango', 'papaya',
    'pineapple', 'sapodilla', 'star_fruit'
]

QUALITY_CLASSES = ['bad', 'good', 'medium']

FRUIT_DIR_MAP = {
    'banana': 'banana/fruit_quality_outputs_banana',
    'custard_apple': 'custard_apple/fruit_quality_outputs_custard_apple',
    'dragonfruit': 'dragonfruit/fruit_quality_outputs_dragonfruit',
    'guava': 'guava/fruit_quality_outputs_guava',
    'jackfruit': 'jackfruit/fruit_quality_outputs_jackfruit',
    'jujube': 'jujube/fruit_quality_outputs_jujube',
    'lemon': 'lemon/fruit_quality_outputs_lemon (1)',
    'lychee': 'lychee/fruit_quality_outputs_lychee',
    'mango': 'mango/fruit_quality_outputs_mango',
    'papaya': 'papaya/fruit_quality_outputs_papaya',
    'pineapple': 'pineapple/fruit_quality_outputs_pineapple',
    'sapodilla': 'sapodilia/fruit_quality_outputs_sapodilla',
    'star_fruit': 'starfruit/fruit_quality_outputs_star_fruit',
}

BEST_MODEL_CONFIG = {
    'banana': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (100.0%)'),
    'custard_apple': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (99.69%)'),
    'dragonfruit': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (100.0%)'),
    'guava': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (99.38%)'),
    'jackfruit': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (99.67%)'),
    'jujube': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (98.81%)'),
    'lemon': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (95.95%)'),
    'lychee': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (100.0%)'),
    'mango': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (94.38%)'),
    'papaya': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (85.59%)'),
    'pineapple': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (97.00%)'),
    'sapodilla': ('efficientnet_b0', 'best_efficientnet_b0.pth', 'EfficientNet-B0 (100.0%)'),
    'star_fruit': ('mobilenetv3_large', 'best_mobilenetv3_large.pth', 'MobileNetV3-Large (92.19%)'),
}

# --- Neural Network Architectures ---

class CustomCNN13(nn.Module):
    def __init__(self, num_classes=13, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.global_pool(self.features(x)))


class CustomCNNQuality(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.global_pool(self.features(x)))


def build_resnet50(num_classes=13):
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_mobilenet_v3(num_classes=3):
    model = models.mobilenet_v3_large(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def build_efficientnet_b0(num_classes=3):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# --- Loader Helpers ---

def load_fruit_classifier(mode="best", device="cpu"):
    """Load Stage 1: 13-Fruit Species Classifier."""
    if mode == "custom_cnn":
        model = CustomCNN13(num_classes=13)
        ckpt_path = BASE_OUTPUT_DIR / "13-fruits/fruit_classification_outputs/models/best_custom_cnn.pth"
    else:
        model = build_resnet50(num_classes=13)
        ckpt_path = BASE_OUTPUT_DIR / "13-fruits/fruit_classification_outputs/models/best_resnet_50.pth"

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_quality_classifier(fruit_name: str, mode="best", device="cpu"):
    """Load Stage 2: 3-Class Quality Classifier for the specified fruit."""
    rel_dir = FRUIT_DIR_MAP.get(fruit_name)
    if not rel_dir:
        raise ValueError(f"Unknown fruit: {fruit_name}")

    models_dir = BASE_OUTPUT_DIR / rel_dir / "models"

    if mode == "custom_cnn":
        model = CustomCNNQuality(num_classes=3)
        ckpt_path = models_dir / "best_custom_cnn.pth"
        model_name = "Custom CNN Quality Baseline"
    else:
        arch, filename, model_name = BEST_MODEL_CONFIG[fruit_name]
        ckpt_path = models_dir / filename
        if arch == "mobilenetv3_large":
            model = build_mobilenet_v3(num_classes=3)
        elif arch == "efficientnet_b0":
            model = build_efficientnet_b0(num_classes=3)
        elif arch == "resnet_50":
            model = build_resnet50(num_classes=3)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, model_name
