from pathlib import Path
from xml.etree import ElementTree


def extract_coco_label(
    svg_path: Path, image_path: str, image_id: str, class_to_id: dict[str, int]
) -> dict:
    """
    Extract COCO-style label dictionary from an SVG file.
    Note: Assumes that the SVG has been annotated with instance and class information.

    :param svg_path: Path to the SVG file
    :type svg_path: Path
    :param image_path: Path to the image file
    :type image_path: str
    :param image_id: Unique identifier for the image
    :type image_id: str
    :param class_to_id: Mapping from instance class names to category IDs
    :type class_to_id: dict[str, int]
    :return: COCO-style label dictionary
    :rtype: dict
    """
    with open(svg_path, "r") as f:
        svg = f.read()

    # get width and height from svg root
    svg_root = ElementTree.fromstring(svg)
    width = int(svg_root.get("width", "0").replace("px", ""))
    height = int(svg_root.get("height", "0").replace("px", ""))

    return {
        "file_name": image_path,
        "width": width,
        "height": height,
        "image_id": image_id,
        "annotations": _extract_coco_annotations_from_svg(svg, class_to_id),
    }


def _extract_coco_annotations_from_svg(svg: str, class_to_id: dict[str, int]) -> list[dict]:
    annotations = []
    svg_root = ElementTree.fromstring(svg)
    instance_ids = set()

    for element in list(svg_root.iter()):
        instance_id = element.get("instance-id")
        if instance_id is not None:
            instance_ids.add(instance_id)

    for instance_id in instance_ids:
        # Find all elements with this instance-id
        elements = [el for el in list(svg_root.iter()) if el.get("instance-id") == instance_id]
        if not elements:
            continue

        # Assume all elements with the same instance-id have the same class
        instance_class = elements[0].get("instance-class")
        if instance_class is None or instance_class not in class_to_id:
            raise ValueError(f"Unknown class '{instance_class}' for instance-id '{instance_id}'")

        # Collect all path data for this instance
        paths = []
        for el in elements:
            if el.tag.endswith("polygon"):
                points = el.get("points")
                if not points:
                    raise ValueError(
                        f"Polygon element with instance-id '{instance_id}' has no points attribute"
                    )
                points = points.strip().split(" ")
                points = [tuple(map(float, p.split(","))) for p in points]
                points = [(round(x), round(y)) for x, y in points]
                paths.append(points)
            else:
                raise ValueError(
                    f"Unsupported SVG element '{el.tag}' for instance-id '{instance_id}'"
                )

        # build bounding box
        min_x = min(point[0] for path in paths for point in path)
        max_x = max(point[0] for path in paths for point in path)
        min_y = min(point[1] for path in paths for point in path)
        max_y = max(point[1] for path in paths for point in path)
        bbox = [min_x, min_y, max_x, max_y]

        # flatten paths for segmentation
        segmentation = []
        for path in paths:
            flat_path = []
            for point in path:
                flat_path.extend(point)
            segmentation.append(flat_path)

        annotation = {
            "bbox": bbox,
            "category_id": class_to_id[instance_class],
            "segmentation": segmentation,
        }
        annotations.append(annotation)

    return annotations
