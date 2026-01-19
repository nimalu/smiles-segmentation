from pathlib import Path

from detectron2.config import CfgNode, get_cfg
from detectron2.projects import point_rend


def get_config(n_classes: int) -> CfgNode:
    """
    Get Detectron2 config for PointRend Mask R-CNN model.

    :param n_classes: Number of classes for the model
    :type n_classes: int
    :return: Config object for the model
    :rtype: CfgNode
    """
    cfg = get_cfg()
    point_rend.add_pointrend_config(cfg)
    cfg.merge_from_file(f"{Path(__file__).parent}/configs/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

    cfg.DATASETS.TRAIN = ("smiles_train",)
    cfg.DATASETS.TEST = ("smiles_val",)

    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1024

    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1024

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = n_classes
    cfg.MODEL.POINT_HEAD.NUM_CLASSES = n_classes

    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.01
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.STEPS = (200, 500, 750)  # Steps to reduce LR
    cfg.SOLVER.GAMMA = 0.5  # Multiply LR by 0.5 at each step
    cfg.SOLVER.MAX_ITER = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    return cfg
