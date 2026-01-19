import warnings

warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*", category=FutureWarning)

# ruff: noqa: E402
import os
from datetime import datetime
from pathlib import Path

import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.utils.events import get_event_storage
from register_dataset import register_smiles_dataset_splits

# Register dataset with train/val split (80/20)
register_smiles_dataset_splits(
    dataset_dir=Path(__file__).parent.parent / "dataset",
    train_ratio=0.8,
    seed=42,
)
smiles_metadata = MetadataCatalog.get("smiles_train")
thing_classes = smiles_metadata.thing_classes


# Configure Detectron2 with PointRend for training
cfg = get_cfg()
add_pointrend_config(cfg)
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.merge_from_file(str(Path(__file__).parent / "configs" / "Base-PointRend-RCNN-FPN.yaml"))
# Load Pre-trained PointRend Weights
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"


cfg.MODEL.POINT_HEAD.NUM_CLASSES = len(thing_classes)
cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14

cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS = 2048
cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS = 5
cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS = 8192
cfg.MODEL.POINT_HEAD.IN_TEST = True


# Multi-scale training
cfg.INPUT.MIN_SIZE_TRAIN = (960, 1024, 1100)
cfg.INPUT.MAX_SIZE_TRAIN = 1400
cfg.DATASETS.TRAIN = ("smiles_train",)
cfg.DATASETS.TEST = ("smiles_val",)
cfg.DATALOADER.NUM_WORKERS = 4


# Optimizer settings
cfg.SOLVER.IMS_PER_BATCH = 8  # PointRend uses more memory, keep conservative
cfg.SOLVER.BASE_LR = 0.002
cfg.SOLVER.MAX_ITER = 2000  # PointRend converges slightly slower
cfg.SOLVER.STEPS = (1000, 1800)  # Learning rate decay steps
cfg.SOLVER.GAMMA = 0.1  # Learning rate decay factor
cfg.SOLVER.WARMUP_ITERS = 250  # Gradual warmup
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.CHECKPOINT_PERIOD = 250
# cfg.SOLVER.AMP.ENABLED = True  # Enable mixed precision training
cfg.MODEL.DEVICE = "cpu"

# Model settings
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(thing_classes)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # ROIs per image
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256  # Anchors for RPN

# Data augmentation
cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.8, 1.0]


# Validation loss evaluation period
cfg.TEST.EVAL_PERIOD = 100  # Check validation loss every 100 iterations

# Set output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
cfg.OUTPUT_DIR = str(Path(__file__).parent / timestamp)


class ValidationLossHook(HookBase):
    """Hook to compute validation loss and implement early stopping."""

    def __init__(self, cfg, patience=5, min_delta=0.001):
        super().__init__()
        self._cfg = cfg.clone()
        self._loader = None
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.should_stop = False

    def before_train(self):
        # Build validation data loader
        self._loader = iter(build_detection_test_loader(self._cfg, self._cfg.DATASETS.TEST[0]))

    def after_step(self):
        # Only evaluate at specified intervals
        if (self.trainer.iter + 1) % self._cfg.TEST.EVAL_PERIOD != 0:
            return

        # Compute validation loss
        val_loss = self._compute_val_loss()

        # Log validation loss
        storage = get_event_storage()
        storage.put_scalar("validation_loss", val_loss)

        # Early stopping logic
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            print(f"Iter {self.trainer.iter}: New best validation loss: {val_loss:.4f}")
        else:
            self.patience_counter += 1
            print(
                f"Iter {self.trainer.iter}: Val loss: {val_loss:.4f} (no improvement, patience: {self.patience_counter}/{self.patience})"
            )

            if self.patience_counter >= self.patience:
                print(f"Early stopping triggered after {self.trainer.iter} iterations")
                self.should_stop = True

    def _compute_val_loss(self):
        """Compute average validation loss over a subset of validation data."""
        # Keep model in training mode to get loss values
        was_training = self.trainer.model.training
        self.trainer.model.train()

        total_loss = 0.0
        num_batches = min(10, len(self._cfg.DATASETS.TEST))  # Sample 10 batches

        with torch.no_grad():
            for i in range(num_batches):
                try:
                    data = next(self._loader)
                except StopIteration:
                    # Reload iterator if exhausted
                    self._loader = iter(
                        build_detection_test_loader(self._cfg, self._cfg.DATASETS.TEST[0])
                    )
                    data = next(self._loader)

                # Model returns loss dict when in training mode
                loss_dict = self.trainer.model(data)
                losses = sum(loss_dict.values())
                total_loss += losses.item()

        # Restore original training state
        if not was_training:
            self.trainer.model.eval()

        return total_loss / num_batches


class MyTrainer(DefaultTrainer):
    def __init__(self, cfg, patience=5, min_delta=0.001):
        self._patience = patience
        self._min_delta = min_delta
        super().__init__(cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        # Add validation loss hook
        self.val_hook = ValidationLossHook(self.cfg, self._patience, self._min_delta)
        hooks.insert(-1, self.val_hook)  # Insert before PeriodicWriter
        return hooks

    def train(self):
        """Override train to check for early stopping."""
        super().train()
        if self.val_hook.should_stop:
            print("Training stopped early due to no improvement in validation loss.")


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

print(cfg)
# Initialize trainer with early stopping parameters
trainer = MyTrainer(cfg, patience=5, min_delta=0.001)
trainer.resume_or_load(resume=False)
trainer.train()
