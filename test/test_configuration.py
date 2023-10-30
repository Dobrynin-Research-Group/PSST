from pathlib import Path

import pytest

import psst


@pytest.fixture
def yaml_config() -> str:
    return """---
run:
  num_epochs: 3
  num_samples_train: 512000
  num_samples_test: 219500

generator:
  parameter: Bg
  batch_size: 64
  phi_range:
    min_value: 1e-2
    max_value: 1e3
    shape: 224
    log_scale: True
  nw_range:
    min_value: 1e-2
    max_value: 1e3
    shape: 224
    log_scale: True
  visc_range:
    min_value: 1e-2
    max_value: 1e3
    shape: (64, 244, 244)
    log_scale: True
  bg_range:
    min_value: 1e-2
    max_value: 1e3
    shape: (64, 1, 1)
  bth_range:
    min_value: 1e-2
    max_value: 1e3
    shape: (64, 1, 1)
  pe_range:
    min_value: 1e-2
    max_value: 1e3
    shape: (64, 1, 1)

adam:
  lr: 1e-3
  betas: [0.7, 0.9]
  eps: 1e-9
  weight_decay: 0.0
...
"""


@pytest.fixture
def json_config() -> str:
    return """{
    "run": {
        "num_epochs": 3,
        "num_samples_train": 512000,
        "num_samples_test": 219500
    },
    "generator": {
        "parameter": "Bg",
        "batch_size": 64,
        "phi_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "shape": 224,
            "log_scale": true
        },
        "nw_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "shape": 224,
            "log_scale": true
        },
        "visc_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "log_scale": true,
            "shape": (64, 244, 244)
        },
        "bg_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "shape": (64, 1, 1)
        },
        "bth_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "shape": (64, 1, 1)
        },
        "pe_range": {
            "min_value": 0.01,
            "max_value": 1000.0,
            "shape": (64, 1, 1)
        }
    },
    "adam": {
        "lr": 0.001,
        "betas": [
            0.7,
            0.9
        ],
        "eps": 1e-09,
        "weight_decay": 0.0
    }
}
"""


def _do_run_config(run_config: psst.RunConfig):
    assert run_config.num_epochs == 3
    assert run_config.num_samples_train == 512000
    assert run_config.num_samples_test == 219500
    assert run_config.checkpoint_filename == "chk.pt"
    assert run_config.checkpoint_frequency == 0


def _do_adam_config(adam_config: psst.AdamConfig):
    assert adam_config.lr == 1e-3
    assert len(adam_config.betas) == 2
    assert adam_config.betas[0] == 0.7
    assert adam_config.betas[1] == 0.9
    assert adam_config.eps == 1e-9
    assert adam_config.weight_decay == 0


def _do_gen_config(gen_config: psst.GeneratorConfig):
    assert gen_config.parameter == "Bg"
    assert gen_config.batch_size == 64

    ranges = [
        gen_config.phi_range,
        gen_config.nw_range,
        gen_config.visc_range,
        gen_config.bg_range,
        gen_config.bth_range,
        gen_config.pe_range,
    ]
    assert all(r.min_value == 0.01 for r in ranges)
    assert all(r.max_value == 1000 for r in ranges)

    assert all(r.log_scale for r in ranges[:3])
    assert all(not r.log_scale for r in ranges[3:])

    assert all(r.shape[0] == 224 for r in ranges[:2])
    assert all(r.shape[0] == 64 for r in ranges[2:])


def test_yaml_config(tmp_path: Path, yaml_config: str):
    filepath = tmp_path / "test_config.yaml"
    filepath.write_text(yaml_config)

    run_config, adam_config, gen_config = psst.load_config(filepath)

    _do_run_config(run_config)
    _do_adam_config(adam_config)
    _do_gen_config(gen_config)


def test_json_config(tmp_path: Path, json_config: str):
    filepath = tmp_path / "test_config.yaml"
    filepath.write_text(json_config)

    run_config, adam_config, gen_config = psst.load_config(filepath)

    _do_run_config(run_config)
    _do_adam_config(adam_config)
    _do_gen_config(gen_config)
