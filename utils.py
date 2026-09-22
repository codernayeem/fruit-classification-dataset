import random
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

# Normalization constants (ImageNet)
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

IMG_SIZE = 224

TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE
])

FRUIT_METADATA = {
    'banana': {'display': 'Banana', 'local_name': 'Kola'},
    'custard_apple': {'display': 'Custard Apple', 'local_name': 'Ata / Sharifa'},
    'dragonfruit': {'display': 'Dragon Fruit', 'local_name': 'Dragon Fruit'},
    'guava': {'display': 'Guava', 'local_name': 'Peyara'},
    'jackfruit': {'display': 'Jackfruit', 'local_name': 'Kathal'},
    'jujube': {'display': 'Jujube', 'local_name': 'Boroi / Kul'},
    'lemon': {'display': 'Lemon', 'local_name': 'Lebu'},
    'lychee': {'display': 'Lychee', 'local_name': 'Lichu'},
    'mango': {'display': 'Mango', 'local_name': 'Aam'},
    'papaya': {'display': 'Papaya', 'local_name': 'Pepe'},
    'pineapple': {'display': 'Pineapple', 'local_name': 'Anarosh'},
    'sapodilla': {'display': 'Sapodilla', 'local_name': 'Sofeda'},
    'star_fruit': {'display': 'Star Fruit', 'local_name': 'Kamranga'},
}

QUALITY_METADATA = {
    'good': {
        'display': 'Good (Prime / Fresh)',
        'color': '#10B981',
        'badge': 'Prime Quality',
        'desc': 'Fresh, optimal ripeness, free of rot or fungal defects. Suitable for premium distribution.'
    },
    'medium': {
        'display': 'Medium (Partially Ripe / Minor Blemish)',
        'color': '#F59E0B',
        'badge': 'Medium Grade',
        'desc': 'Early stage decay, minor surface blemish, or partial ripeness. Suitable for immediate processing.'
    },
    'bad': {
        'display': 'Bad (Rotten / Defective)',
        'color': '#EF4444',
        'badge': 'Defective / Discard',
        'desc': 'Extensive fungal growth, severe rot, or decay. Must be discarded.'
    }
}

MANIFEST_PATH = Path("d:/ML Lab/fruit_dataset_processed/split_manifest.csv")
TEST_DIR = Path("d:/ML Lab/fruit_dataset_processed/test")

_manifest_df = None

def get_test_manifest():
    global _manifest_df
    if _manifest_df is None:
        if MANIFEST_PATH.exists():
            df = pd.read_csv(MANIFEST_PATH)
            _manifest_df = df[df['split'] == 'test'].copy()
        else:
            records = []
            if TEST_DIR.exists():
                for f_dir in TEST_DIR.iterdir():
                    if f_dir.is_dir():
                        for q_dir in f_dir.iterdir():
                            if q_dir.is_dir():
                                for img in q_dir.glob("*.*"):
                                    records.append({
                                        'split': 'test',
                                        'fruit': f_dir.name,
                                        'quality': q_dir.name,
                                        'file_name': img.name,
                                        'rel_path': f"test/{f_dir.name}/{q_dir.name}/{img.name}"
                                    })
            _manifest_df = pd.DataFrame(records)
    return _manifest_df

def get_random_test_sample(fruit_filter=None, quality_filter=None):
    """Retrieve a random image metadata and path from the test partition."""
    df = get_test_manifest()
    if df.empty:
        return None

    filtered = df
    if fruit_filter and fruit_filter != "All Fruits":
        filtered = filtered[filtered['fruit'] == fruit_filter]
    if quality_filter and quality_filter != "All Qualities":
        filtered = filtered[filtered['quality'] == quality_filter]

    if filtered.empty:
        return None

    sampled = filtered.sample(n=1).iloc[0]
    full_path = Path("d:/ML Lab/fruit_dataset_processed") / sampled['rel_path']
    return {
        'fruit': sampled['fruit'],
        'quality': sampled['quality'],
        'file_name': sampled['file_name'],
        'full_path': full_path
    }

def preprocess_image(pil_img: Image.Image):
    """Preprocess PIL Image into PyTorch input tensor."""
    rgb_img = pil_img.convert("RGB")
    tensor = TRANSFORMS(rgb_img).unsqueeze(0)
    return tensor

def apply_clahe(pil_img: Image.Image):
    """Apply Contrast Limited Adaptive Histogram Equalization for diagnostic view."""
    img_np = np.array(pil_img.convert("RGB"))
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced_rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(enhanced_rgb)

def plot_fruit_probabilities(labels, values, title="Top-5 Fruit Species Confidence (%)"):
    """Generate a horizontal bar plot with crisp white background for fruit variety."""
    rev_labels = labels[::-1]
    rev_vals = values[::-1]
    y_pos = np.arange(len(labels))
    
    fig, ax = plt.subplots(figsize=(6.5, 2.2), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    # Highlight highest confidence
    colors = ['#CBD5E1' if i < len(labels)-1 else '#2563EB' for i in range(len(labels))]
    bars = ax.barh(y_pos, rev_vals, color=colors, height=0.55, edgecolor='none')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(rev_labels, fontsize=9.5, fontweight='600', color='#0F172A')
    ax.set_xlim(0, 118)
    ax.set_xlabel('Confidence (%)', fontsize=8.5, fontweight='600', color='#475569')
    ax.tick_params(colors='#475569', labelsize=8.5)
    
    for bar, val in zip(bars, rev_vals):
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', ha='left', fontsize=8.5, fontweight='700',
                color='#1E293B' if val > 0 else '#94A3B8')
                
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.grid(axis='x', linestyle='--', color='#E2E8F0', alpha=0.8)
    if title:
        ax.set_title(title, fontsize=10, fontweight='700', color='#0F172A', pad=8, loc='left')
    fig.tight_layout(pad=0.6)
    return fig


def plot_quality_probabilities(labels, values, title="Quality Grade Distribution (%)"):
    """Generate a horizontal bar plot with crisp white background for quality inspection."""
    rev_labels = labels[::-1]
    rev_vals = values[::-1]
    y_pos = np.arange(len(labels))
    
    color_map = {
        'good': '#10B981',    # Emerald Green
        'medium': '#F59E0B',  # Amber
        'bad': '#EF4444'      # Crimson
    }
    bar_colors = [color_map.get(lbl.lower(), '#3B82F6') for lbl in rev_labels]
    
    fig, ax = plt.subplots(figsize=(6.5, 1.8), dpi=140)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    
    bars = ax.barh(y_pos, rev_vals, color=bar_colors, height=0.52, edgecolor='none')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.upper() for l in rev_labels], fontsize=9.5, fontweight='600', color='#0F172A')
    ax.set_xlim(0, 118)
    ax.set_xlabel('Confidence (%)', fontsize=8.5, fontweight='600', color='#475569')
    ax.tick_params(colors='#475569', labelsize=8.5)
    
    for bar, val in zip(bars, rev_vals):
        w = bar.get_width()
        ax.text(w + 1.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}%',
                va='center', ha='left', fontsize=8.5, fontweight='700',
                color='#1E293B' if val > 0.1 else '#94A3B8')
                
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CBD5E1')
    ax.spines['bottom'].set_color('#CBD5E1')
    ax.grid(axis='x', linestyle='--', color='#E2E8F0', alpha=0.8)
    if title:
        ax.set_title(title, fontsize=10, fontweight='700', color='#0F172A', pad=8, loc='left')
    fig.tight_layout(pad=0.6)
    return fig

