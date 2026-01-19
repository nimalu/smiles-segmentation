import os
import random
from pathlib import Path

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from register_dataset import load_smiles_dataset, register_smiles_dataset_splits

# Register dataset with train/val split (80/20) - same as training
register_smiles_dataset_splits(
    dataset_dir=Path(__file__).parent.parent / "dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes

# Configure with simple Mask R-CNN (matching 02b_train.py)
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
)

# Use same resolution as training
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1024
cfg.MODEL.WEIGHTS = os.path.join(
    Path(__file__).parent, "output", "20260119_125015", "model_final.pth"
)  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
predictor = DefaultPredictor(cfg)


dataset = load_smiles_dataset(Path(__file__).parent.parent / "dataset")

os.makedirs("debug", exist_ok=True)

for d in random.sample(dataset, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    if len(outputs["instances"]) == 0:
        print("No instances detected.")
        continue
    else:
        print(f"Detected {len(outputs['instances'])} instances.")

    v = Visualizer(
        im[:, :, ::-1],
        scale=1.0,
        metadata=smiles_metadata,
        instance_mode=ColorMode.IMAGE,  # Show original image colors
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    output_path = os.path.join("debug", os.path.basename(d["file_name"]))
    output_image = v.get_image()[:, :, ::-1]
    cv2.imwrite(output_path, output_image)
    print(f"Saved prediction visualization to {output_path}")
