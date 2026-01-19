"""
Script to run inference on the validation set of the SMILES dataset using a trained Detectron2 model
"""

import os
from pathlib import Path

import cv2
from config import get_config
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from register_dataset import register_smiles_dataset_splits

# Define the training run name to load the correct model weights
TRAINING_NAME = "20260119_145453"

# Register dataset with train/val split - same as training
register_smiles_dataset_splits(
    dataset_dir=Path(__file__).parent.parent / "dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes

# Configure model matching training script
cfg = get_config(n_classes=len(thing_classes))
cfg.MODEL.WEIGHTS = os.path.join(Path(__file__).parent, "output", TRAINING_NAME, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)


val_dataset = DatasetCatalog.get("smiles_val")
os.makedirs("debug", exist_ok=True)

for d in val_dataset[:20]:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    v = Visualizer(
        im[:, :, ::-1],
        scale=1.0,
        metadata=smiles_metadata,
        instance_mode=ColorMode.IMAGE,
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_path = os.path.join("debug", os.path.basename(d["file_name"]))
    output_image = v.get_image()[:, :, ::-1]
    cv2.imwrite(output_path, output_image)
    print(f"Saved prediction visualization to {output_path}")
