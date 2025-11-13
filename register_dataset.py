import json
from pathlib import Path
from typing import List, Dict, Any

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def load_smiles_dataset(dataset_dir: str | Path) -> List[Dict[str, Any]]:
    """
    Load the SMILES dataset from individual JSON annotation files.

    Args:
        dataset_dir: Path to the dataset directory containing 'images' and 'labels' folders

    Returns:
        List of dataset dictionaries in Detectron2 format
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
            # Convert bbox from [x, y, w, h] to [x1, y1, x2, y2]
            bbox = ann["bbox"]
            x, y, w, h = bbox

            obj = {
                "bbox": [x, y, x + w, y + h],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": ann["segmentation"],
                "category_id": ann["category_id"],
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts


def register_smiles_dataset(dataset_name: str, dataset_dir: str):
    """
    Register the SMILES dataset with Detectron2.

    Args:
        dataset_name: Name to register the dataset under (e.g., "smiles_train")
        dataset_dir: Path to the dataset directory
    """
    # load category names
    classes_file = Path(dataset_dir) / "classes.json"
    if not classes_file.exists():
        raise ValueError(f"Classes file not found: {classes_file}")
    with open(classes_file, "r") as f:
        class_to_id = json.load(f)

    # Register the dataset
    DatasetCatalog.register(dataset_name, lambda d=dataset_dir: load_smiles_dataset(d))

    # Set metadata
    MetadataCatalog.get(dataset_name).set(
        thing_classes=list(class_to_id.keys()),
        evaluator_type="coco",
    )

    print(f"✓ Registered dataset '{dataset_name}' with {len(class_to_id)} categories")
    print(f"  Categories: {list(class_to_id.keys())}")


def get_dataset_stats(dataset_name: str):
    """
    Print statistics about the registered dataset.

    Args:
        dataset_name: Name of the registered dataset
    """
    dataset_dicts = DatasetCatalog.get(dataset_name)

    print(f"\nDataset Statistics for '{dataset_name}':")
    print(f"  Total images: {len(dataset_dicts)}")

    total_annotations = sum(len(d["annotations"]) for d in dataset_dicts)
    print(f"  Total annotations: {total_annotations}")

    # Count annotations per category
    category_counts = {}
    for d in dataset_dicts:
        for ann in d["annotations"]:
            cat_id = ann["category_id"]
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

    metadata = MetadataCatalog.get(dataset_name)
    thing_classes = metadata.thing_classes

    print("  Annotations per category:")
    for cat_id, count in sorted(category_counts.items()):
        cat_name = (
            thing_classes[cat_id]
            if cat_id < len(thing_classes)
            else f"Unknown_{cat_id}"
        )
        print(f"    {cat_name} (id={cat_id}): {count}")


if __name__ == "__main__":
    # Path to the dataset
    DATASET_DIR = "../smiles-rendering/dataset"

    # Register the dataset
    register_smiles_dataset(
        dataset_name="smiles_dataset",
        dataset_dir=DATASET_DIR,
    )

    # Print dataset statistics
    get_dataset_stats("smiles_dataset")

    # Example: Access the dataset
    print("\n" + "=" * 60)
    print("Dataset registered successfully!")
    print("=" * 60)
    print("\nTo use this dataset in your code:")
    print("  from register_dataset import register_smiles_dataset")
    print("  register_smiles_dataset('smiles_dataset', '../smiles-rendering/dataset')")
    print("\nOr if you want to split into train/val:")
    print(
        "  register_smiles_dataset('smiles_train', '../smiles-rendering/dataset/train')"
    )
    print("  register_smiles_dataset('smiles_val', '../smiles-rendering/dataset/val')")
