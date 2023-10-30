from math import nan, inf
import numpy as np
import pytest
import torch

import psst


@pytest.fixture
def gen_config():
    return psst.GeneratorConfig(
        parameter="Bg",
        phi_range=psst.Range(1e-5, 0.1, 64, True),
        nw_range=psst.Range(100, 1e5, 64, True),
        visc_range=psst.Range(1, 1e6, 2, True),
        bg_range=psst.Range(0.5, 1.5, 2),
        bth_range=psst.Range(0.2, 0.95, 2),
        pe_range=psst.Range(3, 25, 2),
        batch_size=2,
    )


@pytest.mark.parametrize("phi_size", [2, 29, 100])
@pytest.mark.parametrize("nw_size", [2, 29, 100])
@pytest.mark.parametrize("batch_size", [2, 29, 100])
def test_sample_shapes(phi_size, nw_size, batch_size, gen_config: psst.GeneratorConfig):
    gen_config.phi_range = psst.Range(
        gen_config.phi_range.min_value, gen_config.phi_range.max_value, phi_size, True
    )
    gen_config.nw_range = psst.Range(
        gen_config.nw_range.min_value, gen_config.nw_range.max_value, nw_size, True
    )
    gen_config.batch_size = batch_size
    gen_samples = psst.SampleGenerator(**gen_config)
    value = next(gen_samples(1))

    assert len(value) == 4

    visc = value[0]
    Bg = value[1]
    Bth = value[2]
    Pe = value[3]

    assert visc.shape == (batch_size, phi_size, nw_size)
    assert Bg.shape == (batch_size,)
    assert Bth.shape == (batch_size,)
    assert Pe.shape == (batch_size,)


def test_parameter_value(gen_config: psst.GeneratorConfig):
    for p in (1, ("Bg",)):
        gen_config.parameter = p
        with pytest.raises(TypeError, match=r"must be a string"):
            _ = psst.SampleGenerator(**gen_config)

    for p in ("", "bg", b"bth"):
        gen_config.parameter = p
        with pytest.raises(ValueError, match=r"must be either 'Bg' or 'Bth'"):
            _ = psst.SampleGenerator(**gen_config)

    for p in ("Bg", "Bth"):
        gen_config.parameter = p
        gen_samples = psst.SampleGenerator(**gen_config)
        assert gen_samples.parameter == p


def test_batch_size(gen_config: psst.GeneratorConfig):
    for b in ("ab", (1,), nan):
        gen_config.batch_size = b  # type: ignore
        with pytest.raises(TypeError):
            _ = psst.SampleGenerator(**gen_config)

    gen_config.batch_size = inf
    with pytest.raises(OverflowError):
        _ = psst.SampleGenerator(**gen_config)

    for b in (0.2, -50):
        gen_config.batch_size = b
        with pytest.raises(ValueError):
            _ = psst.SampleGenerator(**gen_config)

    gen_config.batch_size = 82
    gen_samples = psst.SampleGenerator(**gen_config)
    assert gen_samples.batch_size == 82
    for _ in gen_samples(4):
        pass
    assert gen_samples.batch_size == 82


def _do_range_errors_grid(key: str, config: psst.GeneratorConfig):
    values: list[tuple[float, float, int]] = [
        (-3, 1, 244),
        (0, 1, 244),
        (1, 0.9, 244),
        (1, 3, 1),
        (1, 3, 0),
    ]
    ranges = [psst.Range(*val, True) for val in values]
    for r in ranges:
        setattr(config, key, r)
        with pytest.raises(ValueError):
            _ = psst.SampleGenerator(**config)

    setattr(config, key, (1e-6, 0.1, 244, True))
    with pytest.raises(AttributeError):
        _ = psst.SampleGenerator(**config)

    class Range:
        min = 1e-6
        max = 0.1
        num = 244

    setattr(config, key, Range())
    with pytest.raises(AttributeError):
        _ = psst.SampleGenerator(**config)


def _do_range_errors_other(key: str, config: psst.GeneratorConfig):
    values: list[tuple[float, float, int]] = [(-3, 1, 244), (0, 1, 244), (1, 0.9, 244)]
    ranges = [psst.Range(*abc, True) for abc in values]
    for r in ranges:
        setattr(config, key, r)
        with pytest.raises(ValueError):
            _ = psst.SampleGenerator(**config)

    setattr(config, key, (1e-6, 0.1, 0, True))
    with pytest.raises(AttributeError):
        _ = psst.SampleGenerator(**config)

    class Range:
        min = 1e-6
        max = 0.1
        num = 0

    setattr(config, key, Range())
    with pytest.raises(AttributeError):
        _ = psst.SampleGenerator(**config)
