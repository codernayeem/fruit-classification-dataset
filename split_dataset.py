import os
import shutil
import math
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(SEED)

SRC_DIR = Path("d:/ML Lab/fruit_dataset")
DST_DIR = Path("d:/ML Lab/fruit_dataset_processed")

EXPECTED_FRUITS = [
    'banana', 'custard_apple', 'dragonfruit', 'guava', 'jackfruit',
    'jujube', 'lemon', 'lychee', 'mango', 'papaya',
    'pineapple', 'sapodilla', 'star_fruit'
]

EXPECTED_QUALITIES = ['good', 'medium', 'bad']
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

def scan_dataset(base_dir: Path):
    if not base_dir.exists():
        raise FileNotFoundError(f"Source directory '{base_dir}' not found.")

    records = []
    for fruit in sorted(EXPECTED_FRUITS):
        fruit_dir = base_dir / fruit
        if not fruit_dir.exists():
            continue

        for quality in EXPECTED_QUALITIES:
            quality_dir = fruit_dir / quality
            if quality_dir.exists():
                images = sorted([f for f in quality_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS and f.is_file()])
                for img_path in images:
                    records.append({
                        "fruit": fruit,
                        "quality": quality,
                        "strat_label": f"{fruit}_{quality}",
                        "file_name": img_path.name,
                        "path": str(img_path)
                    })
    return pd.DataFrame(records)

def main():
    print(f"Scanning dataset from {SRC_DIR}...")
    df = scan_dataset(SRC_DIR)
    total_images = len(df)
    print(f"Total samples found: {total_images}")

    # Stratified 80:20 train/test split preserving both fruit and quality proportions
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df['strat_label'],
        random_state=SEED
    )

    print(f"Train split: {len(train_df)} samples ({len(train_df)/total_images*100:.1f}%)")
    print(f"Test split:  {len(test_df)} samples ({len(test_df)/total_images*100:.1f}%)")

    splits = {'train': train_df, 'test': test_df}

    # Prepare directories
    for split_name in ['train', 'test']:
        for fruit in EXPECTED_FRUITS:
            for quality in EXPECTED_QUALITIES:
                target_folder = DST_DIR / split_name / fruit / quality
                target_folder.mkdir(parents=True, exist_ok=True)

    # Copy files
    manifest_records = []
    for split_name, split_data in splits.items():
        print(f"Copying files for {split_name} split...")
        for _, row in tqdm(split_data.iterrows(), total=len(split_data), desc=f"Writing {split_name}"):
            src_path = Path(row['path'])
            fruit = row['fruit']
            quality = row['quality']
            dst_path = DST_DIR / split_name / fruit / quality / src_path.name

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            manifest_records.append({
                "split": split_name,
                "fruit": fruit,
                "quality": quality,
                "file_name": src_path.name,
                "rel_path": f"{split_name}/{fruit}/{quality}/{src_path.name}"
            })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_df.to_csv(DST_DIR / "split_manifest.csv", index=False)
    print("Dataset splitting and copying complete!")
    print(f"Manifest saved to {DST_DIR / 'split_manifest.csv'}")

if __name__ == "__main__":
    main()
