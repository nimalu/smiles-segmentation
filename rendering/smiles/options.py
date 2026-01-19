from dataclasses import dataclass


@dataclass
class Options:
    """
    Configuration options for molecular rendering.

    This class contains configurable parameters for generating molecular diagrams,
    including dimensions, colors, fonts, and various display options.
    """

    width: int = 500
    height: int = 500
    backgroundColour: tuple = (1, 1, 1)
    dummiesAreAttachments: bool = False
    dummyIsotopeLabels: bool = False
    noAtomLabels: bool = False
    addBondIndices: bool = False
    addAtomIndices: bool = False
    unspecifiedStereoIsUnknown: bool = False
    useMolBlockWedging: bool = True
    baseFontSize: float = 0.6
    fixedFontSize: int = 50  # Absolute font size in pixels (overrides baseFontSize)
    fontFile: str | None = None
    additionalAtomLabelPadding: float = 0.1
    bondLineWidth: float = 2
    multipleBondOffset: float = 0.15
    fixedScale: float = 50.0  # Fixed scale for consistent sizing (pixels per angstrom)
    rotate: int = 30
    remove_non_molecular_paths: bool = True
