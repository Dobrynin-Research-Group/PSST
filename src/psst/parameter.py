from enum import Enum

__all__ = ["Parameter"]


class Parameter(Enum):
    """Represents either the good solvent parameter (``bg``) or the thermal blob
    parameter (``bth``).
    """

    bg = "bg"
    bth = "bth"
