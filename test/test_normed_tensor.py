import pytest

import psst


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_normalize(is_normalized: bool, log_scale: bool):
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_unnormalize(is_normalized: bool, log_scale: bool):
    ...


def test_repeated_norm_unnorm():
    ...


@pytest.mark.parametrize("log_scale", [True, False])
def test_generate(log_scale: bool):
    ...


def test_create_raw():
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_create_from_numpy(is_normalized: bool, log_scale: bool):
    ...


@pytest.mark.parametrize("is_normalized", [True, False])
@pytest.mark.parametrize("log_scale", [True, False])
def test_create_from_range(is_normalized: bool, log_scale: bool):
    ...
