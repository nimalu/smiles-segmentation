import json
import random
from pathlib import Path

import tqdm
from molecules import COMMON_MOLECULES
from smiles_segmentation import Options
from smiles_segmentation.coco import extract_coco_label
from smiles_segmentation.images import svg_to_pil
from smiles_segmentation.postprocessing import (
    annotate_svg_with_instances,
    annotate_svg_with_smiles,
    crop_scale_svg,
    extract_classes_from_svg,
    flatten_paths_to_polygons,
)
from smiles_segmentation.renderer import create_mol, create_svg

# Dataset configuration constants
OUTPUT_DIR = Path(__file__).parent.parent / "dataset"
NUM_SAMPLES = 5_000

# Randomization ranges
ROTATION_RANGE = (0, 360)
WIDTH_RANGE = (400, 700)
HEIGHT_RANGE = (400, 700)
BASE_FONT_SIZE_RANGE = (3.0, 5.0)
ATOM_LABEL_PADDING_RANGE = (0.05, 0.15)
BOND_LINE_WIDTH_RANGE = (3.0, 5.5)
MULTIPLE_BOND_OFFSET_RANGE = (0.1, 0.2)

# Random seed for reproducibility (None for random)
RANDOM_SEED = 42


def sample_options(
    rotation_range: tuple[int, int] = (0, 360),
    width_range: tuple[int, int] = (512, 768),
    height_range: tuple[int, int] = (512, 768),
    base_font_size_range: tuple[float, float] = (3, 5),
    atom_label_padding_range: tuple[float, float] = (0.05, 0.15),
    bond_line_width_range: tuple[float, float] = (1.5, 2.5),
    multiple_bond_offset_range: tuple[float, float] = (0.1, 0.2),
):
    rotation = random.randint(*rotation_range)
    width = random.randint(*width_range)
    height = random.randint(*height_range)
    base_font_size = random.uniform(*base_font_size_range)
    atom_label_padding = random.uniform(*atom_label_padding_range)
    bond_line_width = random.uniform(*bond_line_width_range)
    multiple_bond_offset = random.uniform(*multiple_bond_offset_range)

    # Create options with randomized parameters
    return Options(
        width=width,
        height=height,
        rotate=rotation,
        baseFontSize=base_font_size,
        additionalAtomLabelPadding=atom_label_padding,
        bondLineWidth=bond_line_width,
        multipleBondOffset=multiple_bond_offset,
        backgroundColour=(1, 1, 1),  # White background
    )


def render_base_svg(
    smiles: str,
    options: Options,
):
    mol = create_mol(smiles)
    svg = create_svg(mol, options)
    svg = annotate_svg_with_instances(svg, mol, options)
    svg = annotate_svg_with_smiles(svg, smiles)
    return svg


def generate_dataset(
    output_base: Path,
    num_samples: int = 100,
    rotation_range: tuple[int, int] = (0, 360),
    width_range: tuple[int, int] = (512, 768),
    height_range: tuple[int, int] = (512, 768),
    base_font_size_range: tuple[float, float] = (0.5, 0.7),
    atom_label_padding_range: tuple[float, float] = (0.05, 0.15),
    bond_line_width_range: tuple[float, float] = (1.5, 2.5),
    multiple_bond_offset_range: tuple[float, float] = (0.1, 0.2),
    seed: int | None = None,
):
    if seed is not None:
        random.seed(seed)

    # Create directory structure
    images_dir = output_base / "images"
    labels_dir = output_base / "labels"
    for dir_path in [images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm.tqdm(range(num_samples), desc="Generating base SVGs"):
        options = sample_options(
            rotation_range,
            width_range,
            height_range,
            base_font_size_range,
            atom_label_padding_range,
            bond_line_width_range,
            multiple_bond_offset_range,
        )
        smiles, name = random.choice(COMMON_MOLECULES)
        svg = render_base_svg(smiles, options)

        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "w") as f:
            f.write(svg)

    for i in tqdm.tqdm(range(num_samples), desc="Flattening SVG paths"):
        imgae_path = images_dir / f"{i:06d}.svg"
        with open(imgae_path, "r") as f:
            svg = f.read()
        svg = flatten_paths_to_polygons(svg)
        with open(imgae_path, "w") as f:
            f.write(svg)

    for i in tqdm.tqdm(range(num_samples), desc="Cropping and scaling SVGs"):
        imgae_path = images_dir / f"{i:06d}.svg"
        with open(imgae_path, "r") as f:
            svg = f.read()
        svg = crop_scale_svg(svg, scale=3.0)
        with open(imgae_path, "w") as f:
            f.write(svg)

    classes = set()
    for i in tqdm.tqdm(range(num_samples), desc="Preparing class mapping"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        classes.update(extract_classes_from_svg(svg))

    class_to_id = {cls: idx for idx, cls in enumerate(sorted(classes))}
    with open(output_base / "classes.json", "w") as f:
        json.dump(class_to_id, f, indent=4)

    for i in tqdm.tqdm(range(num_samples), desc="Generating labels"):
        svg_path = images_dir / f"{i:06d}.svg"
        sample = extract_coco_label(svg_path, f"images/{i:06d}.png", f"{i:06d}", class_to_id)

        label_path = labels_dir / f"{i:06d}.json"
        with open(label_path, "w") as f:
            json.dump(sample, f, indent=4)

    for i in tqdm.tqdm(range(num_samples), desc="Generating PNGs"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        pil_image = svg_to_pil(svg)
        png_path = images_dir / f"{i:06d}.png"
        pil_image.save(png_path)

    for i in tqdm.tqdm(range(num_samples), desc="Cleaning up SVGs"):
        image_path = images_dir / f"{i:06d}.svg"
        image_path.unlink()  # Remove SVG file


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output directory {OUTPUT_DIR} already exists. Please remove it before running."
        )

    generate_dataset(
        output_base=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        rotation_range=ROTATION_RANGE,
        width_range=WIDTH_RANGE,
        height_range=HEIGHT_RANGE,
        base_font_size_range=BASE_FONT_SIZE_RANGE,
        atom_label_padding_range=ATOM_LABEL_PADDING_RANGE,
        bond_line_width_range=BOND_LINE_WIDTH_RANGE,
        multiple_bond_offset_range=MULTIPLE_BOND_OFFSET_RANGE,
        seed=RANDOM_SEED,
    )


if __name__ == "__main__":
    main()
