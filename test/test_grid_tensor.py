import pytest

import psst


def test_raw_creation():
    ...


@pytest.mark.parametrize("min_value", [1e-6, 0.1])
@pytest.mark.parametrize("max_value", [0.2, 0.9])
@pytest.mark.parametrize("log_scale", [True, False])
@pytest.mark.parametrize("shape", [1, 64, (2, 2), (64, 1, 1)])
def test_from_range(
    min_value: float, max_value: float, log_scale: bool, shape: int | tuple
):
    ...


def test_raw_errors():
    ...


def test_from_range_errors():
    ...
