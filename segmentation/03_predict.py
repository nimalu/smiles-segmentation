import os
import random

import cv2
import matplotlib.pyplot as plt
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.visualizer import ColorMode, Visualizer
from PIL import Image
from register_dataset import load_smiles_dataset, register_smiles_dataset_splits

# Register dataset with train/val split (80/20) - same as training
register_smiles_dataset_splits(
    dataset_dir="../smiles-dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes

cfg = get_cfg()
add_pointrend_config(cfg)
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.merge_from_file("configs/Base-PointRend-RCNN-FPN.yaml")

cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(thing_classes)

# Use native resolution for testing
cfg.INPUT.MIN_SIZE_TEST = 1024
cfg.INPUT.MAX_SIZE_TEST = 1400
cfg.MODEL.WEIGHTS = os.path.join(
    "output", "20251221_191719", "model_0000499.pth"
)  # Path to the trained model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
predictor = DefaultPredictor(cfg)


dataset = load_smiles_dataset("../smiles-dataset")

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
        instance_mode=ColorMode.IMAGE_BW,
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    fig, ax = plt.subplots()
    im = Image.fromarray(v.get_image()[:, :, ::-1])
    ax.imshow(im)
    ax.axis("off")
    output_path = os.path.join("debug", os.path.basename(d["file_name"]))
    os.makedirs("debug", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=300)
    print(f"Saved prediction visualization to {output_path}")
