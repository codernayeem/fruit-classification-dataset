# Multi-Fruit Variety & Quality Inspection System

A deep learning benchmark and two-stage inspection pipeline for 13 agricultural fruit varieties and their 3-tier quality grades (Good, Medium, Bad), featuring confidence gating, transfer learning champions, and lightweight custom CNN architectures.

---

## 1. Dataset Overview

- **Total Images:** 20,823 curated RGB photographs (~1.0 GB)
- **Fruit Varieties (13 classes):** Banana (*Musa acuminata*), Custard Apple (*Annona squamosa*), Dragonfruit (*Selenicereus undatus*), Guava (*Psidium guajava*), Jackfruit (*Artocarpus heterophyllus*), Jujube (*Ziziphus mauritiana*), Lemon (*Citrus limon*), Lychee (*Litchi chinensis*), Mango (*Mangifera indica*), Papaya (*Carica papaya*), Pineapple (*Ananas comosus*), Sapodilla (*Manilkara zapota*), Star Fruit (*Averrhoa carambola*).
- **Quality Tiers (3 grades):**
  - **Good:** Firm physical structure, clear skin, free of disease or severe mechanical damage.
  - **Medium:** Minor cosmetic blemishes, partial skin discoloration, or early ripeness; suitable for immediate processing.
  - **Bad:** Severe fungal lesion, deep rot, mechanical rupture, or advanced microbial decay; unmarketable.
- **Data Partitions:** 80% Training (16,658 images) and 20% Testing (4,165 images), deterministically stratified using random seed `42`.

---

## 2. Repository Structure

```
d:/ML Lab/
├── fruit_dataset/                # Raw unpartitioned image collection (<fruit>/<quality>/)
├── fruit_dataset_processed/      # Stratified 80/20 train/test image splits
│   ├── train/                    # 16,658 training images
│   ├── test/                     # 4,165 testing images
│   └── split_manifest.csv        # Partition mapping per image
├── metadata/                     # Machine-readable schemas and manifests
│   ├── data_dictionary.csv       # Attribute schemas, data types, and constraints
│   ├── class_labels.json         # Taxonomy, local names, and numeric index mappings
│   ├── dataset_statistics.csv    # Class-by-class image counts per split
│   └── split_manifest.csv        # Master split manifest
├── Output/                       # Model checkpoints, evaluation metrics, and notebooks
│   ├── 13-fruits/                # Stage 1 macro classification (models, plots, summaries, notebook)
│   ├── <fruit_name>/             # Stage 2 intra-fruit quality models and benchmark logs
│   │   └── fruit_quality_outputs_<fruit>/
│   │       ├── models/           # Checkpoints (.pth): best_custom_cnn, best_resnet_50, etc.
│   │       ├── plots/            # Confusion matrices, ROC curves, learning curves
│   │       └── final_quality_benchmark_summary.csv
│   └── ...
├── base_notebooks/               # Clean, unexecuted baseline Jupyter notebooks
│   ├── Fruit_classification.ipynb
│   └── Fruit_quality_classification.ipynb
├── report/                       # Academic LaTeX manuscript and cropped publication figures
│   ├── main.tex                  # Primary LaTeX research paper
│   ├── figures/                  # Publication-ready figure assets
│   ├── Output/                   # Cleaned output folder retaining plots & summaries
│   └── references.bib            # BibTeX bibliography
├── app.py                        # Streamlit web application (Live 2-stage inspection)
├── model_loader.py               # PyTorch model definitions (CustomCNN13, CustomCNNQuality)
├── utils.py                      # Preprocessing, CLAHE equalization, Matplotlib visualizers
├── split_dataset.py              # Reproducible stratified train/test split script
└── README.md                     # Documentation and reproduction guide
```

---

## 3. Two-Stage Inspection Architecture

```
                       [ Input Fruit Image (224x224 RGB) ]
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Stage 1: Fruit Variety Recognition    │
                   │      (ResNet-50 / Custom CNN 13)        │
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

- **Stage 1 (Fruit Variety):** 13-class classification. ResNet-50 achieves **99.93%** test accuracy; Custom CNN achieves **96.37%** test accuracy.
- **Confidence Gate:** Default threshold $0.85$ (85%) ensures downstream quality models only receive correctly identified fruit varieties.
- **Stage 2 (Quality Inspection):** Fruit-specific 3-class classifier. Transfer learning champions (MobileNetV3-Large, EfficientNet-B0, ResNet-50) achieve **96.99%** mean accuracy across all 13 fruits.

---

## 4. Quickstart Guide

### Prerequisites
- Python 3.10+
- PyTorch with CUDA support (or CPU)
- Streamlit, Matplotlib, OpenCV, Pillow, Pandas, Scikit-learn

### Running the Streamlit Application
Launch the interactive web interface:
```bash
streamlit run app.py
```
Key features:
- Side-by-side **Image Upload** and **Random Test Image** selector.
- Real-time Stage 1 & Stage 2 inference with model latency display.
- High-contrast, publication-grade Matplotlib horizontal probability charts.
- Confidence gating toggle with adjustable threshold slider.
- Diagnostic CLAHE contrast-enhancement view.

### Reproducing the Train/Test Split
To regenerate the deterministic 80/20 train/test dataset partition:
```bash
python split_dataset.py
```

---

## 5. Model Weights & Artifacts

Pre-trained PyTorch weights (`.pth`) for all models are located in `Output/`:
- `Output/13-fruits/fruit_classification_outputs/models/best_resnet_50.pth`
- `Output/<fruit>/fruit_quality_outputs_<fruit>/models/best_<model_name>.pth`

Models can be loaded programmatically via `model_loader.py`:
```python
from model_loader import load_fruit_classifier, load_quality_classifier

# Load Stage 1 classifier
fruit_model = load_fruit_classifier(mode='best', device='cuda')

# Load Stage 2 classifier for a specific fruit
quality_model, model_name = load_quality_classifier('sapodilla', mode='best', device='cuda')
```
