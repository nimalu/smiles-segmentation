import os
import random

import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from PIL import Image

from register_dataset import register_smiles_dataset, load_smiles_dataset

register_smiles_dataset(
    dataset_name="smiles_dataset",
    dataset_dir="../smiles-rendering/dataset",
)
smiles_metadata = MetadataCatalog.get("smiles_dataset")
thing_classes = smiles_metadata.thing_classes

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("output", "model_0000499.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
predictor = DefaultPredictor(cfg)


dataset = load_smiles_dataset('../smiles-rendering/dataset')

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
        instance_mode=ColorMode.IMAGE_BW
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    fig, ax = plt.subplots()
    im = Image.fromarray(v.get_image()[:, :, ::-1])
    ax.imshow(im)
    ax.axis("off")
    output_path = os.path.join("output", os.path.basename(d["file_name"]))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    print(f"Saved prediction visualization to {output_path}")