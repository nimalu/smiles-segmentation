# SMILES Instance Segmentation

This project trains a deep learning model to perform instance segmentation on molecular structure diagrams, detecting and segmenting individual chemical bonds and atoms from rendered SMILES notation.

## Overview

The project consists of two main components:

1. **Synthetic Dataset Generation** - Automatically generates a labeled dataset of molecular structure diagrams from SMILES notation with various rendering parameters (rotation, size, font, bond widths)
2. **Model Training and Inference** - Trains a Mask R-CNN model using Detectron2 to perform instance segmentation on molecular structures

## Examples

### Input Molecular Structures
![Sample molecule 1](dataset/images/000000.png)

### Segmentation Results

![Prediction 1](debug/002089.png)

## Installation & Usage

Install
```bash
uv sync
```

Generate a synthetic dataset:
```bash
uv run python rendering/generate_dataset.py
```

Train the segmentation model:
```bash
uv run python segmentation/02_train.py
```

Run predictions on test images:
```bash
uv run python segmentation/03_predict.py
```

## Dataset Generation Pipeline

- Using RDKit, a SMILES string is rendered to SVG
- A few postprocessing steps are applied to extract instance segmentation masks for each object
- The SVG is converted to PNG


## Model Training Pipeline

- Uses Detectron2's Mask R-CNN architecture 
- performs fine-tuning on a pretrained COCO model

## Known limitations

- Because of the way the renderer converts SVG paths into polygons, it creates artifacts for donut-like shapes.