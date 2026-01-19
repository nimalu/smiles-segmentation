from io import BytesIO

from cairosvg import svg2png
from PIL import Image


def svg_to_pil(svg: str):
    """
    Convert an SVG string to a PIL Image object.

    Uses CairoSVG to convert the SVG to PNG data, then loads it as a PIL Image.
    This is useful for generating raster image outputs from vector molecular diagrams.

    Args:
        svg (str): SVG content as a string

    Returns:
        PIL.Image: PIL Image object containing the rasterized molecular diagram

    Raises:
        ValueError: If SVG conversion fails
        ImportError: If required dependencies (cairosvg, PIL) are not available
    """

    png_data = svg2png(bytestring=svg.encode("utf-8"))
    if png_data is None:
        raise ValueError("Failed to convert SVG to PNG")
    return Image.open(BytesIO(png_data))
