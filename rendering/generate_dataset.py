import json
import random
from pathlib import Path

import tqdm
from molecules import COMMON_MOLECULES
from smiles.coco import extract_coco_label
from smiles.images import svg_to_pil
from smiles.options import Options
from smiles.postprocessing import (
    annotate_svg_with_instances,
    annotate_svg_with_smiles,
    crop_scale_svg,
    extract_classes_from_svg,
    flatten_paths_to_polygons,
)
from smiles.renderer import create_mol, create_svg

OUTPUT_DIR = Path(__file__).parent.parent / "dataset"
NUM_SAMPLES = 5_000
RANDOM_SEED = 42
RANDOMIZATION_RANGES = {
    "rotation": (0, 360),
    "width": (400, 700),
    "height": (400, 700),
    "base_font_size": (3.0, 5.0),
    "atom_label_padding": (0.05, 0.15),
    "bond_line_width": (3.0, 5.5),
    "multiple_bond_offset": (0.1, 0.2),
}


def sample_options(randomization_ranges: dict) -> Options:
    """
    Samples rendering options within the specified randomization ranges.

    :param randomization_ranges: Dictionary containing ranges for randomization parameters
    :type randomization_ranges: dict
    :return: An Options object with randomized parameters
    :rtype: Options
    """
    rotation = random.randint(*randomization_ranges["rotation"])
    width = random.randint(*randomization_ranges["width"])
    height = random.randint(*randomization_ranges["height"])
    base_font_size = random.uniform(*randomization_ranges["base_font_size"])
    atom_label_padding = random.uniform(*randomization_ranges["atom_label_padding"])
    bond_line_width = random.uniform(*randomization_ranges["bond_line_width"])
    multiple_bond_offset = random.uniform(*randomization_ranges["multiple_bond_offset"])

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
    """
    Renders the base SVG for a given SMILES string and rendering options.
    The SVG is annotated with instance and SMILES information.

    :param smiles: SMILES string representing the molecule
    :type smiles: str
    :param options: Rendering options
    :type options: Options
    """
    mol = create_mol(smiles)
    svg = create_svg(mol, options)
    svg = annotate_svg_with_instances(svg, mol, options)
    svg = annotate_svg_with_smiles(svg, smiles)
    return svg


def generate_dataset(
    output_base: Path,
    num_samples: int,
    randomization_ranges: dict,
    seed: int | None = None,
    clean_up=False,
):
    ###### Preparation ######
    if seed is not None:
        random.seed(seed)

    # Create directory structure
    images_dir = output_base / "images"
    labels_dir = output_base / "labels"
    for dir_path in [images_dir, labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    ###### Generate base SVGs ######
    for i in tqdm.tqdm(range(num_samples), desc="Generating base SVGs"):
        options = sample_options(randomization_ranges)
        smiles, _ = random.choice(COMMON_MOLECULES)
        svg = render_base_svg(smiles, options)

        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "w") as f:
            f.write(svg)

    ###### Flatten SVG paths ######
    for i in tqdm.tqdm(range(num_samples), desc="Flattening SVG paths"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        svg = flatten_paths_to_polygons(svg)
        with open(image_path, "w") as f:
            f.write(svg)

    ###### Crop and scale SVGs ######
    for i in tqdm.tqdm(range(num_samples), desc="Cropping and scaling SVGs"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        svg = crop_scale_svg(svg, scale=3.0)
        with open(image_path, "w") as f:
            f.write(svg)

    ###### Collect classes ######
    classes = set()
    for i in tqdm.tqdm(range(num_samples), desc="Preparing class mapping"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        classes.update(extract_classes_from_svg(svg))

    class_to_id = {cls: idx for idx, cls in enumerate(sorted(classes))}
    with open(output_base / "classes.json", "w") as f:
        json.dump(class_to_id, f, indent=4)

    ###### Extract labels ######
    for i in tqdm.tqdm(range(num_samples), desc="Generating labels"):
        svg_path = images_dir / f"{i:06d}.svg"
        sample = extract_coco_label(svg_path, f"images/{i:06d}.png", f"{i:06d}", class_to_id)

        label_path = labels_dir / f"{i:06d}.json"
        with open(label_path, "w") as f:
            json.dump(sample, f, indent=4)

    ###### Generate PNGS ######
    for i in tqdm.tqdm(range(num_samples), desc="Generating PNGs"):
        image_path = images_dir / f"{i:06d}.svg"
        with open(image_path, "r") as f:
            svg = f.read()
        pil_image = svg_to_pil(svg)
        png_path = images_dir / f"{i:06d}.png"
        pil_image.save(png_path)

    ###### Cleanup SVGs ######
    if not clean_up:
        return
    for i in tqdm.tqdm(range(num_samples), desc="Cleaning up SVGs"):
        image_path = images_dir / f"{i:06d}.svg"
        image_path.unlink()  # Remove SVG file


if __name__ == "__main__":
    generate_dataset(
        output_base=OUTPUT_DIR,
        num_samples=NUM_SAMPLES,
        randomization_ranges=RANDOMIZATION_RANGES,
        seed=RANDOM_SEED,
    )
