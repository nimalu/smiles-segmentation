import re
from xml.etree import ElementTree

import numpy as np
import svgpathtools
from rdkit import Chem
from svgpathtools import parse_path

from . import renderer


def extract_classes_from_svg(svg: str) -> set[str]:
    classes = set()
    svg_root = ElementTree.fromstring(svg)
    for element in list(svg_root.iter()):
        instance_class = element.get("instance-class")
        if instance_class is not None:
            classes.add(instance_class)
    return classes


def annotate_svg_with_smiles(svg: str, smiles: str) -> str:
    """
    Add SMILES string as metadata to SVG molecular diagrams.

    Inserts a <metadata> element containing the SMILES string at the
    beginning of the SVG content.

    Args:
        svg (str): Raw SVG content from RDKit
        smiles (str): SMILES string representing the molecule

    Returns:
        str: Enhanced SVG with SMILES metadata
    """
    svg_root = ElementTree.fromstring(svg)
    metadata_elem = ElementTree.Element("metadata")
    smiles_elem = ElementTree.SubElement(metadata_elem, "smiles")
    smiles_elem.text = smiles
    svg_root.insert(0, metadata_elem)
    return ElementTree.tostring(svg_root, encoding="unicode")


def crop_scale_svg(svg: str, padding=5, scale=1.0) -> str:
    """
    Crop the SVG content to the bounding box of the drawn elements (paths).
    Determines width and height based on the extents of all path elements.
    Then, adjusts the elements' coordinates
    """
    pattern = r"(\d+\.?\d*),(\d+\.?\d*)"
    matches = re.findall(pattern, svg)
    x_coords = [float(m[0]) for m in matches]
    y_coords = [float(m[1]) for m in matches]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    width = round((max_x - min_x + 2 * padding) * scale)
    height = round((max_y - min_y + 2 * padding) * scale)

    # Adjust the SVG elements' coordinates
    def adjust_coords(match):
        x = (float(match.group(1)) - min_x + padding) * scale
        y = (float(match.group(2)) - min_y + padding) * scale
        return f"{x},{y}"

    adjusted_svg = re.sub(pattern, adjust_coords, svg)

    # Update the SVG root element's width, height, and viewBox
    svg_root = ElementTree.fromstring(adjusted_svg)
    svg_root.set("width", str(width))
    svg_root.set("height", str(height))
    svg_root.set("viewBox", f"0 0 {width} {height}")

    return ElementTree.tostring(svg_root, encoding="unicode")


def annotate_svg_with_instances(svg: str, mol: Chem.Mol, options: renderer.Options) -> str:
    """
    Add semantic information to SVG molecular diagrams.

    Post-processes RDKit-generated SVG by adding instance IDs and classes to
    identify individual atoms and bonds. This enables subsequent coloring and
    analysis of molecular fragments.

    Args:
        svg (str): Raw SVG content from RDKit
        mol (Chem.Mol): RDKit molecule object used to generate the SVG
        options (renderer.Options): Rendering options that affect post-processing

    Returns:
        str: Enhanced SVG with instance-id, instance-class, and bond-atoms attributes

    Note:
        This function removes non-molecular SVG paths if remove_non_molecular_paths
        is True in the options.
    """
    """
    Post-process the SVG output by adding instance IDs and classes.
    Each atom and bond is assigned a unique instance ID and a class based on
    its type (element symbol for atoms, bond type for bonds).

    Non-atom/bond paths are removed from the SVG.
    """
    bond_types = [str(bond.GetBondType()) for bond in mol.GetBonds()]
    bond_dirs = [str(bond.GetBondDir()) for bond in mol.GetBonds()]
    bond_atoms = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    instances = []

    svg_root = ElementTree.fromstring(svg)
    for element in list(svg_root.iter()):
        # only process path elements
        if element.tag != r"{http://www.w3.org/2000/svg}path":
            continue

        bond_index = _get_bond_index(element)
        if bond_index is not None:
            instance_name = f"bond-{bond_index}"
            if instance_name not in instances:
                instances.append(instance_name)
            instance_id = instances.index(instance_name)

            attrib = {
                "instance-class": f"{bond_types[bond_index]}-{bond_dirs[bond_index]}",
                "instance-id": str(instance_id),
                "bond-atoms": f"{bond_atoms[bond_index][0]}-{bond_atoms[bond_index][1]}",
            }
            attrib.update(element.attrib)
            element.attrib = attrib
            continue

        atom_index = _get_atom_index(element)
        if atom_index is not None:
            instance_name = f"atom-{atom_index}"
            if instance_name not in instances:
                instances.append(instance_name)
            instance_id = instances.index(instance_name)

            attrib = {
                "instance-class": "ATOM",
                "instance-id": str(instance_id),
            }
            attrib.update(element.attrib)
            element.attrib = attrib
            continue

        # remove non-atom/bond paths
        if options.remove_non_molecular_paths:
            svg_root.remove(element)

    # remove class attributes
    for element in list(svg_root.iter()):
        if "class" in element.attrib:
            del element.attrib["class"]

    return ElementTree.tostring(svg_root, encoding="unicode")


def flatten_paths_to_polygons(svg: str, n_per_curve=5) -> str:
    """
    Convert all SVG path elements to polygon elements with stroke width applied.

    This function flattens complex path definitions into simple polygons,
    expanding stroked paths into filled polygons that represent the same visual area.
    Paths with stroke width are converted to outline polygons; filled paths are
    converted directly.

    Args:
        svg (str): Original SVG content
        n_per_curve (int): Number of points to sample per curved segment
    Returns:
        str: Modified SVG content with paths converted to polygons
    """
    svg_root = ElementTree.fromstring(svg)
    for element in list(svg_root.iter()):
        if element.tag != "{http://www.w3.org/2000/svg}path":
            continue

        path_data = element.get("d", "")
        path = parse_path(path_data)

        # Extract stroke width from style or stroke-width attribute
        stroke_width = 0
        style = element.get("style", "")
        style_dict = dict(s.split(":") for s in style.split(";") if s and ":" in s)

        if "stroke-width" in style_dict:
            stroke_width = float(style_dict["stroke-width"].replace("px", ""))
        elif "stroke-width" in element.attrib:
            stroke_width = float(element.attrib["stroke-width"].replace("px", ""))

        # extract fill
        fill = style_dict.get("fill", element.get("fill", "none"))

        # Sample points along the path / or take line segments directly
        points = []
        for segment in path:
            if isinstance(segment, svgpathtools.path.Line):
                points.append((segment.start.real, segment.start.imag))
            else:
                for t in np.linspace(0, 1, n_per_curve):
                    pt = segment.point(t)
                    points.append((pt.real, pt.imag))

        # Add final point
        final_pt = path[-1].end
        points.append((final_pt.real, final_pt.imag))

        # Remove duplicate consecutive points
        if points:
            new_points = [points[0]]
            for pt in points[1:]:
                if pt != new_points[-1]:
                    new_points.append(pt)
            points = new_points

        # If stroke width exists, create outline polygon
        # Goes around the path and offsets points perpendicularly
        if stroke_width > 0 and fill == "none":
            # Calculate perpendicular offsets for each point
            outline_points = []
            half_width = stroke_width / 2

            for i in range(len(points)):
                x, y = points[i]

                # Determine tangent direction
                if i == 0:
                    dx = points[i + 1][0] - x
                    dy = points[i + 1][1] - y
                elif i == len(points) - 1:
                    dx = x - points[i - 1][0]
                    dy = y - points[i - 1][1]
                else:
                    dx = points[i + 1][0] - points[i - 1][0]
                    dy = points[i + 1][1] - points[i - 1][1]

                # Normalize and get perpendicular
                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx /= length
                    dy /= length

                # Perpendicular vector
                perp_x = -dy
                perp_y = dx

                # Add offset points
                outline_points.append((x + perp_x * half_width, y + perp_y * half_width))

            # Add reverse side
            for i in range(len(points) - 1, -1, -1):
                x, y = points[i]

                if i == 0:
                    dx = points[i + 1][0] - x
                    dy = points[i + 1][1] - y
                elif i == len(points) - 1:
                    dx = x - points[i - 1][0]
                    dy = y - points[i - 1][1]
                else:
                    dx = points[i + 1][0] - points[i - 1][0]
                    dy = points[i + 1][1] - points[i - 1][1]

                length = np.sqrt(dx * dx + dy * dy)
                if length > 0:
                    dx /= length
                    dy /= length

                perp_x = -dy
                perp_y = dx

                outline_points.append((x - perp_x * half_width, y - perp_y * half_width))

            points = outline_points

        polygon_points = " ".join(f"{x},{y}" for x, y in points)
        element.set("points", polygon_points)
        element.tag = "{http://www.w3.org/2000/svg}polygon"

        # Remove stroke from style and set fill
        if "stroke" in style_dict:
            style_dict["fill"] = style_dict.get("stroke", "#000000")
            del style_dict["stroke"]
        if "stroke-width" in style_dict:
            del style_dict["stroke-width"]

        element.set("style", ";".join(f"{k}:{v}" for k, v in style_dict.items()))

        if "d" in element.attrib:
            del element.attrib["d"]
        if "stroke-width" in element.attrib:
            del element.attrib["stroke-width"]

    return ElementTree.tostring(svg_root, encoding="unicode")


def color_instances(svg: str):
    """
    Color instances in the SVG output. Each unique instance-id is assigned a unique color.
    """
    matches = re.finditer(r'instance-id="(\d+)"', svg)
    instance_ids = {int(m.group(1)) for m in matches}
    svg_root = ElementTree.fromstring(svg)
    for color, instance_id in zip(_unique_colors(), sorted(instance_ids)):
        # find all elements with the current instance-id
        instance_elements = []
        for element in list(svg_root.iter()):
            if element.get("instance-id") == str(instance_id):
                instance_elements.append(element)

        # set the color of these elements
        for element in instance_elements:
            # adjust style
            style = element.get("style", "")
            style_dict = dict(s.split(":") for s in style.split(";") if s)
            if "stroke" in style_dict:
                style_dict["stroke"] = color

            if "fill" in style_dict and style_dict["fill"] != "none":
                style_dict["fill"] = color
            style = ";".join(f"{k}:{v}" for k, v in style_dict.items())
            element.set("style", style)

            # adjust fill
            if "fill" in element.attrib:
                element.set("fill", color)

    return ElementTree.tostring(svg_root, encoding="unicode")


def _get_atom_index(element: ElementTree.Element):
    """
    Extract the atom index from the class attribute of an SVG element.

    Parses RDKit-generated class attributes to identify which atom
    a particular SVG path element represents.

    Args:
        element (ElementTree.Element): SVG element to examine

    Returns:
        int | None: Atom index if found, None otherwise
    """
    """
    Extract the atom index from the class attribute of an SVG element.
    """
    class_attr = element.get("class", "")
    match = re.search(r"atom-(\d+)", class_attr)
    if match:
        return int(match.group(1))
    return None


def _get_bond_index(element: ElementTree.Element):
    """
    Extract the bond index from the class attribute of an SVG element.

    Parses RDKit-generated class attributes to identify which bond
    a particular SVG path element represents.

    Args:
        element (ElementTree.Element): SVG element to examine

    Returns:
        int | None: Bond index if found, None otherwise
    """
    """
    Extract the bond index from the class attribute of an SVG element.
    """
    class_attr = element.get("class", "")
    match = re.search(r"bond-(\d+)", class_attr)
    if match:
        return int(match.group(1))
    return None


def _unique_colors():
    """
    Generate a sequence of unique, visually distinct colors.

    Uses the HSV color space with golden ratio-based spacing to generate
    a theoretically infinite sequence of unique colors. The algorithm
    ensures good visual separation between consecutive colors.

    Yields:
        str: Hexadecimal color codes (e.g., "#ff5733")

    Note:
        Generates at least 2500 unique colors before any potential repetition,
        which is sufficient for most molecular visualization needs.
    """
    """
    Generate a sequence of unique colors (at least 2500 unique colors).
    """

    def hsv_to_rgb(h, s, v) -> tuple[int, int, int]:
        """
        Convert HSV color values to RGB.

        Args:
            h (float): Hue (0-1)
            s (float): Saturation (0-1)
            v (float): Value/brightness (0-1)

        Returns:
            tuple[int, int, int]: RGB values (0-1 range)
        """
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)

        i %= 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        else:
            return (v, p, q)

    def generate_colors():
        """
        Generate an infinite sequence of visually distinct colors.

        Uses golden ratio spacing in HSV space to maximize visual separation
        between consecutive colors.

        Yields:
            str: Hexadecimal color codes
        """
        sat_offset = 0.0
        hue = 0
        val_offset = 0.0
        golden_ratio = 0.618033988749895
        while True:
            hue = (hue + golden_ratio) % 1
            rgb = hsv_to_rgb(hue, 0.5 + sat_offset * 0.5, 0.5 * val_offset + 0.5)
            yield f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

            sat_offset = (sat_offset + golden_ratio * 3) % 1
            rgb = hsv_to_rgb(hue, 0.5 + sat_offset * 0.5, 0.5 * val_offset + 0.5)
            yield f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

            val_offset = (val_offset + golden_ratio * 11) % 1
            rgb = hsv_to_rgb(hue, 0.5 + sat_offset * 0.5, 0.5 * val_offset + 0.5)
            yield f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"

    return generate_colors()
