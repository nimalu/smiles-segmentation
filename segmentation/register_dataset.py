import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def _load_smiles_dataset(dataset_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load the SMILES dataset from the specified directory.

    :param dataset_dir: Path to the dataset directory
    :type dataset_dir: str | Path
    :return: List of dataset dictionaries in Detectron2 format
    :rtype: List[Dict[str, Any]]
    """
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")

    dataset_dicts = []

    # Get all JSON annotation files
    json_files = sorted(labels_dir.glob("*.json"))

    for idx, json_file in enumerate(json_files):
        with open(json_file, "r") as f:
            annotation_data = json.load(f)

        # Construct the full path to the image
        image_path = dataset_dir / annotation_data["file_name"]

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        record = {
            "file_name": str(image_path),
            "image_id": annotation_data["image_id"],
            "height": annotation_data["height"],
            "width": annotation_data["width"],
        }

        # Process annotations
        objs = []
        for ann in annotation_data["annotations"]:
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann["segmentation"],
                "category_id": ann["category_id"],
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def _split_dataset(
    dataset_dicts: List[Dict[str, Any]], train_ratio: float = 0.8, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train and validation sets.

    Args:
        dataset_dicts: Full dataset
        train_ratio: Ratio of training data (default 0.8 = 80% train, 20% val)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    random.seed(seed)
    indices = list(range(len(dataset_dicts)))
    random.shuffle(indices)

    split_idx = int(len(dataset_dicts) * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_dataset = [dataset_dicts[i] for i in train_indices]
    val_dataset = [dataset_dicts[i] for i in val_indices]

    return train_dataset, val_dataset


def register_smiles_dataset_splits(
    dataset_dir: str | Path, train_ratio: float = 0.8, seed: int = 42
):
    """
    Register the SMILES dataset with train/validation splits in Detectron2's DatasetCatalog.
    Datasets will be registered under the names "smiles_train" and "smiles_val".

    :param dataset_dir: Path to the dataset directory
    :type dataset_dir: str | Path
    :param train_ratio: Ratio of training data (default 0.8 = 80% train, 20% val)
    :type train_ratio: float
    :param seed: Random seed for reproducibility
    :type seed: int
    """
    # Load category names
    classes_file = Path(dataset_dir) / "classes.json"
    if not classes_file.exists():
        raise ValueError(f"Classes file not found: {classes_file}")
    with open(classes_file, "r") as f:
        class_to_id = json.load(f)

    # Load full dataset and split
    full_dataset = _load_smiles_dataset(dataset_dir)
    train_dataset, val_dataset = _split_dataset(full_dataset, train_ratio, seed)

    # Register train dataset
    DatasetCatalog.register("smiles_train", lambda: train_dataset)
    MetadataCatalog.get("smiles_train").set(
        thing_classes=list(class_to_id.keys()),
        evaluator_type="coco",
    )

    # Register validation dataset
    DatasetCatalog.register("smiles_val", lambda: val_dataset)
    MetadataCatalog.get("smiles_val").set(
        thing_classes=list(class_to_id.keys()),
        evaluator_type="coco",
    )

    print("✓ Registered SMILES dataset splits:")
    print(f"  Train: {len(train_dataset)} images ({train_ratio * 100:.0f}%)")
    print(f"  Val: {len(val_dataset)} images ({(1 - train_ratio) * 100:.0f}%)")
    print(f"  Categories: {list(class_to_id.keys())}")
