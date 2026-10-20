import sys
import time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(".").resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from model_loader import (
    load_fruit_classifier,
    load_quality_classifier,
    FRUITS_13,
    QUALITY_CLASSES,
    BEST_MODEL_CONFIG,
    BASE_DIR
)
from utils import NORMALIZE, IMG_SIZE

import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    cpu_cores = os.cpu_count() or 8
    torch.set_num_threads(cpu_cores)
    print(f"Using device: CPU with {cpu_cores} threads")
else:
    print(f"Using device: {device}")

# Preprocessing transform
TRANSFORMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    NORMALIZE
])

class FruitTestDataset(Dataset):
    def __init__(self, manifest_df, base_img_dir, transform=None):
        self.df = manifest_df.reset_index(drop=True)
        self.base_img_dir = Path(base_img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_img_dir / row['rel_path']
        raw_img = Image.open(img_path)
        img_rgb = ImageOps.exif_transpose(raw_img).convert("RGB")
        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)
        return tensor, row['fruit'], row['quality'], str(row['file_name'])

manifest_path = BASE_DIR / "fruit_dataset_processed" / "split_manifest.csv"
df = pd.read_csv(manifest_path)
test_df = df[df['split'] == 'test'].copy().reset_index(drop=True)
print(f"Total test samples to evaluate: {len(test_df)}")

dataset = FruitTestDataset(test_df, BASE_DIR / "fruit_dataset_processed", transform=TRANSFORMS)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

# 1. Load Stage 1 Model
print("\n--- Loading Stage 1 Fruit Classifier (ResNet-50) ---")
s1_model = load_fruit_classifier(mode="best", device=device)
s1_model.eval()

# 2. Stage 1 Predictions
all_s1_preds = []
all_s1_confs = []
all_gt_fruits = []
all_gt_qualities = []
all_filenames = []
all_tensors = []

print("Running Stage 1 Inference...")
t0 = time.time()
with torch.no_grad():
    for tensors, fruits, qualities, fnames in dataloader:
        tensors = tensors.to(device)
        logits = s1_model(tensors)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        
        all_s1_preds.extend([FRUITS_13[p] for p in preds])
        all_s1_confs.extend(confs)
        all_gt_fruits.extend(fruits)
        all_gt_qualities.extend(qualities)
        all_filenames.extend(fnames)

t_s1 = time.time() - t0
print(f"Stage 1 completed in {t_s1:.2f}s (Avg {t_s1/len(test_df)*1000:.2f} ms/image)")

# Stage 1 Accuracy
s1_acc = accuracy_score(all_gt_fruits, all_s1_preds)
print(f"Stage 1 (13-Fruit Species) Accuracy: {s1_acc*100:.2f}%")

# 3. Stage 2 Predictions
print("\n--- Running Stage 2 Quality Grading for Predicted Fruits ---")
# Preload all 13 Stage 2 models to avoid repeated disk reads
s2_models = {}
for fruit in FRUITS_13:
    model, arch_name = load_quality_classifier(fruit, mode="best", device=device)
    model.eval()
    s2_models[fruit] = model

# Group dataset indices by predicted fruit for batched Stage 2 inference
test_df['pred_fruit'] = all_s1_preds
test_df['pred_fruit_conf'] = all_s1_confs

all_s2_preds = [None] * len(test_df)
all_s2_confs = [0.0] * len(test_df)

t1 = time.time()
with torch.no_grad():
    for pred_f in FRUITS_13:
        idxs = test_df.index[test_df['pred_fruit'] == pred_f].tolist()
        if not idxs:
            continue
        sub_subset = torch.utils.data.Subset(dataset, idxs)
        sub_loader = DataLoader(sub_subset, batch_size=64, shuffle=False, num_workers=0)
        
        q_model = s2_models[pred_f]
        sub_preds = []
        sub_confs = []
        for tensors, _, _, _ in sub_loader:
            tensors = tensors.to(device)
            logits = q_model(tensors)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)
            sub_preds.extend([QUALITY_CLASSES[p] for p in preds])
            sub_confs.extend(confs)
            
        for orig_idx, pred_q, conf_q in zip(idxs, sub_preds, sub_confs):
            all_s2_preds[orig_idx] = pred_q
            all_s2_confs[orig_idx] = conf_q

t_s2 = time.time() - t1
print(f"Stage 2 completed in {t_s2:.2f}s (Avg {t_s2/len(test_df)*1000:.2f} ms/image)")
print(f"Total Pipeline Time: {t_s1 + t_s2:.2f}s")

# 4. Construct 39-Class Labels
# Ordered class names: 13 fruits x 3 qualities
class_39_names = []
for fruit in sorted(FRUITS_13):
    for q in QUALITY_CLASSES:
        class_39_names.append(f"{fruit}_{q}")

print(f"\nConstructed {len(class_39_names)} Fine-Grained Classes.")

test_df['pred_quality'] = all_s2_preds
test_df['pred_quality_conf'] = all_s2_confs

test_df['gt_39'] = test_df['fruit'] + "_" + test_df['quality']
test_df['pred_39'] = test_df['pred_fruit'] + "_" + test_df['pred_quality']

overall_acc = accuracy_score(test_df['gt_39'], test_df['pred_39'])
macro_f1 = f1_score(test_df['gt_39'], test_df['pred_39'], average='macro')
weighted_f1 = f1_score(test_df['gt_39'], test_df['pred_39'], average='weighted')

print(f"\n=======================================================")
print(f"=== 39-CLASS END-TO-END HIERARCHICAL BENCHMARK METRICS ===")
print(f"  * Overall Top-1 Accuracy:  {overall_acc*100:.2f}% ({sum(test_df['gt_39'] == test_df['pred_39'])} / {len(test_df)})")
print(f"  * Macro-Averaged F1-Score: {macro_f1*100:.2f}%")
print(f"  * Weighted F1-Score:       {weighted_f1*100:.2f}%")
print(f"=======================================================\n")

# Confusion Matrix
cm_39 = confusion_matrix(test_df['gt_39'], test_df['pred_39'], labels=class_39_names)
cm_df = pd.DataFrame(cm_39, index=class_39_names, columns=class_39_names)

# Save Raw and Normalized Confusion Matrices to CSV
out_dir = BASE_DIR / "Results" / "39-classes_benchmark"
out_dir.mkdir(parents=True, exist_ok=True)

cm_df.to_csv(out_dir / "confusion_matrix_39_raw.csv")

# Normalized (Row-wise percentage)
cm_norm = cm_39.astype('float') / cm_39.sum(axis=1)[:, np.newaxis] * 100
cm_norm_df = pd.DataFrame(cm_norm, index=class_39_names, columns=class_39_names)
cm_norm_df.to_csv(out_dir / "confusion_matrix_39_normalized.csv")

# Classification Report
clf_report = classification_report(test_df['gt_39'], test_df['pred_39'], labels=class_39_names, target_names=class_39_names, digits=4)
with open(out_dir / "classification_report_39.txt", "w") as f:
    f.write(clf_report)

# Save full predictions table
test_df.to_csv(out_dir / "test_predictions_39.csv", index=False)
print(f"Saved artifacts to {out_dir}")

# 5. Plot Confusion Matrix
plt.figure(figsize=(22, 18), dpi=200)
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".1f",
    cmap="Blues",
    xticklabels=[c.replace("_", "\n") for c in class_39_names],
    yticklabels=class_39_names,
    cbar_kws={'label': 'Recall / True Class Accuracy (%)'}
)
plt.title(f"39-Class End-to-End Hierarchical Inspection Confusion Matrix\nOverall Accuracy: {overall_acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Predicted Class (Fruit + Quality Tier)", fontsize=13, fontweight='bold', labelpad=12)
plt.ylabel("Ground Truth Class", fontsize=13, fontweight='bold', labelpad=12)
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8, rotation=0)
plt.tight_layout()
plot_path = out_dir / "confusion_matrix_39.png"
plt.savefig(plot_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved confusion matrix heatmap to {plot_path}")
