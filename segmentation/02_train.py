import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# ruff: noqa: E402
import os
from datetime import datetime
from pathlib import Path

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from register_dataset import register_smiles_dataset_splits

# Register dataset with train/val split (80/20)
register_smiles_dataset_splits(
    dataset_dir=Path(__file__).parent.parent / "dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes


# Configure Detectron2 with simple Mask R-CNN for training
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml")
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
)

cfg.DATASETS.TRAIN = ("smiles_train",)
cfg.DATASETS.TEST = ("smiles_val",)

cfg.INPUT.MIN_SIZE_TRAIN = (800,)
cfg.INPUT.MAX_SIZE_TRAIN = 1024

cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)

cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.01
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.STEPS = (200, 500, 750)  # Steps to reduce LR
cfg.SOLVER.GAMMA = 0.5  # Multiply LR by 0.5 at each step
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.CHECKPOINT_PERIOD = 100

# Set output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.OUTPUT_DIR = str(Path(__file__).parent / "output" / timestamp)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
