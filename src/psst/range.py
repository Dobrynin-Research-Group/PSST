from functools import singledispatch
from typing import Optional

import attrs
import attrs.converters as conv
import attrs.validators as valid


__all__ = ["convert_to_range", "Range"]


@attrs.frozen(eq=True)
class Range:
    """Specifies a range of values between ``min_value`` and ``max_value``, specifying
    ``shape`` for the shape of the desired tesnor and ``log_scale`` for logarithmic
    spacing. For use in ``psst.NormedTensor`` and ``psst.GridTensor``.

    Usage:
        >>> r = Range(1.0, 1e6, shape = 100, log_scale = True)
        >>> t = GridTensor(r)

    Args:
        min_value (float): Minimum value of the range.
        max_value (float): Maximum value of the range.
        shape (tuple[int, ...]): Desired shape of the tensor. Can be a single int for a
          1D tensor of that length.
        log_scale (bool): If False, the range is taken on a linear scale between
          ``min_value`` and ``max_value``. If True, the values are spaced
          geometrically. Default is False.
    """

    @staticmethod
    def _check_max(instance: "Range", _, max_value: float):
        if instance.min_value >= max_value:
            raise ValueError("Cannot have `max_value <= min_value`")

    @staticmethod
    def _check_log_scale(instance: "Range", _, log_scale_val: bool):
        if log_scale_val and instance.min_value <= 0.0:
            raise ValueError(
                f"Cannot have negative values of Range with `log_scale=True`"
            )

    min_value: float = attrs.field(converter=float)
    max_value: float = attrs.field(converter=float, validator=_check_max)
    shape: Optional[int] = attrs.field(
        default=None,
        converter=conv.optional(int),
        validator=valid.optional(valid.gt(0)),
    )
    log_scale: bool = attrs.field(
        default=False, converter=bool, validator=_check_log_scale
    )


@singledispatch
def convert_to_range(x) -> Range:
    raise TypeError("Input type must be tuple, list, or dict")


@convert_to_range.register
def _(x: dict) -> Range:
    return Range(**x)


@convert_to_range.register(tuple)
@convert_to_range.register(list)
def _(x) -> Range:
    return Range(*x)
