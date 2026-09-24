# A Multi-Fruit Classification and Quality Inspection Dataset

A comprehensive deep learning benchmark and two-stage inspection system for 13 commercial and domestic fruit varieties and their 3-tier quality grades (Good, Medium, Bad), featuring confidence-gated routing, state-of-the-art transfer learning architectures, lightweight custom CNN baselines, and an interactive Streamlit application.

---

## 1. Dataset Overview

- **Total Images:** 20,823 curated, authentic RGB optical captures (~1.0 GB)
- **Fruit Varieties (13 species):**
  1. Banana (*Musa acuminata*)
  2. Custard Apple (*Annona squamosa*)
  3. Dragon Fruit (*Selenicereus undatus*)
  4. Guava (*Psidium guajava*)
  5. Jackfruit (*Artocarpus heterophyllus*)
  6. Jujube (*Ziziphus mauritiana*)
  7. Lemon (*Citrus limon*)
  8. Lychee (*Litchi chinensis*)
  9. Mango (*Mangifera indica*)
  10. Papaya (*Carica papaya*)
  11. Pineapple (*Ananas comosus*)
  12. Sapodilla / Sofeda (*Manilkara zapota*)
  13. Star Fruit (*Averrhoa carambola*)
- **Quality Tiers (3 grades per fruit):**
  - **Good:** Firm physical structure, clear peel pigmentation, free of pathological blemishes or mechanical bruises.
  - **Medium:** Minor cosmetic abrasions, partial skin discoloration, or early ripening stage; commercially viable for immediate consumption or processing.
  - **Bad:** Advanced fungal infection, deep tissue decay, open epidermal lacerations, or microbial breakdown; unmarketable.
- **Data Partitions:** 80% Training (16,658 images) and 20% Testing (4,165 images), deterministically stratified across both species and quality grades using random seed `42`.

---

## 2. Directory and Repository Layout

```text
root
├── fruit_dataset/                # Raw unpartitioned photographic collection (<fruit>/<quality>/)
├── fruit_dataset_processed/      # Stratified 80/20 train/test image splits (224x224, CLAHE-enhanced)
│   ├── train/                    # 16,658 training images
│   ├── test/                     # 4,165 testing images
│   └── split_manifest.csv        # Partition mapping per image
├── metadata/                     # Machine-readable schemas and dataset manifests
│   ├── data_dictionary.csv       # Attribute schemas, data types, and constraints
│   ├── class_labels.json         # Botanical taxonomy, local names, and index mappings
│   ├── dataset_statistics.csv    # Class-by-class image counts per split
│   └── split_manifest.csv        # Master split manifest
├── Codebase/                     # Standalone, reproducible training & evaluation notebooks (.ipynb)
│   ├── 13-fruits/                # Task 1: 13-Fruit Species Classification
│   │   └── Fruit_Classification.ipynb
│   ├── banana/                   # Task 2: Intra-fruit quality notebooks
│   │   └── Fruit_quality_classification_banana.ipynb
│   ├── custard_apple/
│   │   └── Fruit_quality_classification_custard_apple.ipynb
│   ├── dragonfruit/
│   │   └── Fruit_quality_classification_dragonfruit.ipynb
│   ├── guava/
│   │   └── Fruit_quality_classification_guava.ipynb
│   ├── jackfruit/
│   │   └── Fruit_quality_classification_jackfruit.ipynb
│   ├── jujube/
│   │   └── Fruit_quality_classification_jujube.ipynb
│   ├── lemon/
│   │   └── Fruit_quality_classification_lemon.ipynb
│   ├── lychee/
│   │   └── Fruit_quality_classification_lychee.ipynb
│   ├── mango/
│   │   └── Fruit_quality_classification_mango.ipynb
│   ├── papaya/
│   │   └── Fruit_quality_classification_papaya.ipynb
│   ├── pineapple/
│   │   └── Fruit_quality_classification_pineapple.ipynb
│   ├── sapodilia/
│   │   └── Fruit_quality_classification_sofeda.ipynb
│   └── starfruit/
│       └── Fruit_quality_classification_starfruit.ipynb
├── Result/                       # Benchmark outputs, trained weights, and evaluation artifacts
│   ├── 13-fruits/
│   │   └── fruit_classification_outputs/
│   │       ├── final_benchmark_summary.csv
│   │       ├── models/           # Best PyTorch model checkpoints (.pth)
│   │       ├── plots/            # Confusion matrices, ROC curves, learning curves, prediction grids
│   │       └── reports/          # Per-class classification reports (.txt)
│   └── <fruit_name>/
│       └── fruit_quality_outputs_<fruit>/
│           ├── final_quality_benchmark_summary.csv
│           ├── models/           # Checkpoints: best_custom_cnn, best_mobilenetv3, best_resnet50, etc.
│           ├── plots/            # Diagnostic curves and normalized confusion matrices
│           └── reports/          # Precision, Recall, Macro-F1 logs
├── app.py                        # Interactive Streamlit application (Live 2-stage grading)
└── README.md                     # Project documentation and reproduction guide
```

---

## 3. Two-Stage Inspection Architecture

```text
                       [ Input Fruit Image (224x224 RGB) ]
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Stage 1: Fruit Variety Recognition    │
                   │        (ResNet-50 / Custom CNN)         │
                   └─────────────────────────────────────────┘
                                        │
                         Confidence Score (P_max)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
             P_max >= 0.85                         P_max < 0.85
                   │                                       │
                   ▼                                       ▼
    ┌─────────────────────────────┐         ┌─────────────────────────────┐
    │ Stage 2: Quality Inspection │         │  Stage 2 Inspection Bypassed│
    │ (Fruit-Specific Top Model)  │         │ (Prevents Error Propagation)│
    └─────────────────────────────┘         └─────────────────────────────┘
                   │
                   ▼
         [ Good / Medium / Bad ]
```

- **Stage 1 (Macro Fruit-Type Recognition):** Evaluates all 13 fruit species simultaneously. ResNet-50 and YOLO26n-cls achieve **99.93%** test accuracy (Macro F1 = 0.9993).
- **Confidence Gate ($\tau = 0.85$):** Samples with species confidence below 85% bypass Stage 2 to prevent cascading classification errors.
- **Stage 2 (Intra-Fruit Quality Inspection):** Evaluates 3-tier freshness (Good, Medium, Bad). Pre-trained backbones achieve **96.99%** average accuracy across all 13 fruits (MobileNetV3-Large, EfficientNet-B0, ResNet-50).

---

## 4. Benchmark Performance Summary

| Fruit Species | Custom CNN (%) | MobileNetV3-Large (%) | YOLO26n-cls (%) | EfficientNet-B0 (%) | ResNet-50 (%) | Top Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Papaya** | 76.18 | **85.59** | 83.82 | 84.41 | 85.29 | MobileNetV3-Large |
| **Lemon** | 84.10 | 96.53 | 96.24 | 95.95 | **96.82** | ResNet-50 |
| **Lychee** | 99.06 | **100.00** | **100.00** | **100.00** | **100.00** | EfficientNet-B0 / MobileNetV3 |
| **Jackfruit** | 76.49 | **99.67** | 97.68 | **99.67** | 99.34 | MobileNetV3-Large |
| **Mango** | 85.31 | **94.38** | **94.38** | 92.81 | 92.50 | MobileNetV3-Large |
| **Star Fruit** | 80.00 | **92.19** | 89.06 | 90.00 | 90.00 | MobileNetV3-Large |
| **Guava** | 97.21 | 99.07 | **99.38** | **99.38** | 99.07 | EfficientNet-B0 |
| **Jujube** | 89.32 | **98.81** | 97.63 | **98.81** | 97.63 | MobileNetV3-Large |
| **Banana** | 97.45 | 99.68 | **100.00** | **100.00** | 99.68 | EfficientNet-B0 |
| **Dragon Fruit**| 99.70 | **100.00** | **100.00** | **100.00** | **100.00** | EfficientNet-B0 |
| **Pineapple** | 92.67 | 96.33 | 96.33 | **97.00** | 96.00 | EfficientNet-B0 |
| **Sapodilla** | 89.62 | 98.96 | **100.00** | **100.00** | 99.65 | EfficientNet-B0 |
| **Custard Apple**| 72.00 | **99.69** | **99.69** | 99.08 | **99.69** | MobileNetV3-Large |
| **Average** | **87.62** | **96.99** | **96.48** | **96.70** | **96.59** | **MobileNetV3-Large (96.99%)** |

---

## 5. Quickstart & Execution

### Prerequisites
- Python 3.10+
- PyTorch >= 2.0.0, Torchvision >= 0.15.0
- Streamlit, OpenCV-Python, Matplotlib, Seaborn, Pandas, Scikit-learn, Pillow

### 1. Launching the Interactive Web Application
```bash
streamlit run app.py
```
**Application Features:**
- Upload image or pick test samples from any of the 13 fruits and 3 quality tiers.
- Real-time Stage 1 (Species Recognition) and Stage 2 (Freshness / Quality Grading).
- Live latency benchmarking, probability distributions, and confidence gating controls.
- Side-by-side original and CLAHE contrast-enhanced visualizations.

### 2. Loading Pretrained Weights from Result/
All trained PyTorch checkpoints (`.pth`) reside under `Result/`:
```python
import torch
from torchvision import models

# Load ResNet-50 13-Fruit Species Classifier
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 13)
state_dict = torch.load("Result/13-fruits/fruit_classification_outputs/models/best_resnet_50.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
```

---

## 6. Citation and License

- **Dataset & Code License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- Open for academic, research, educational, and commercial applications with proper attribution.
