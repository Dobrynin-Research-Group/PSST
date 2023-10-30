import pytest

import psst


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_normalize():
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_unnormalize():
    ...


def test_repeated_norm_unnorm():
    ...


@pytest.mark.parametrize("log_scale", [True, False])
def test_generate():
    ...


def test_create_raw():
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_create_from_numpy():
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_create_from_range():
    ...
