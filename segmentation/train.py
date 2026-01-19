"""
Script to train a segmentation model on the SMILES dataset using Detectron2.
It registers the dataset, configures the model, and starts training.
Outputs are saved in a timestamped directory.
"""

import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# ruff: noqa: E402
import os
from datetime import datetime
from pathlib import Path

from config import get_config
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from register_dataset import register_smiles_dataset_splits

# Register smiles dataset
register_smiles_dataset_splits(
    dataset_dir=Path(__file__).parent.parent / "dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes


cfg = get_config(n_classes=len(thing_classes))

# Set output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.OUTPUT_DIR = str(Path(__file__).parent / "output" / timestamp)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(cfg)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
