from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import curve_fit

import psst
from .structures import *
from .config import EvaluationConfig


__all__ = [
    "AVOGADRO_CONSTANT",
    "GOOD_EXP",
    "transform_data",
    "fit_func",
    "evaluate_dataset",
]

AVOGADRO_CONSTANT = 6.0221408e23
GOOD_EXP = 0.588


def reduce_data(
    conc: np.ndarray,
    mol_weight: np.ndarray,
    repeat_unit: RepeatUnit,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce the concentration and molecular weight measurements from concentration
    in g/L to :math:`(c*l^3)` and weight-average molecular weight in kg/mol to weight-
    average degree of polymerization (number of repeat units per chain).

    Args:
        conc (np.ndarray): Concentration in g/L.
        mol_weight (np.ndarray): Weight-average molecular weight in kg/mol.
        repeat_unit (RepeatUnit): The mass in g/mol and length in nm of a repeat unit.

    Returns:
        tuple[np.ndarray, np.ndarray]: The reduced concentration :math:`cl^3` and
          degree of polymerization :math:`N_w`.
    """

    reduced_conc = AVOGADRO_CONSTANT * conc * (repeat_unit.length / 1e8) ** 3
    degree_polym = mol_weight / repeat_unit.mass * 1e3

    return reduced_conc, degree_polym


def process_data_to_grid(
    phi_data: np.ndarray,
    nw_data: np.ndarray,
    visc_data: np.ndarray,
    phi_range: psst.Range,
    nw_range: psst.Range,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform a set of data ``(phi, nw, visc)`` into an "image", where each "pixel"
    along axis 0 represents a bin of values in the log-scale of ``phi_range`` and those
    along axis 1 represent bins of values in the log-scale of ``nw_range``.

    Args:
        phi (np.ndarray): Experimental concentration data in reduced form :math:`cl^3`.
        nw (np.ndarray): Experimental data for weight-average degree of polymerization.
        visc (np.ndarray): Experimental specific viscosity data. Data at index ``i``
          should correspond to a solution state with reduced concentration ``phi[i]``
          and weight-average DP of polymer chains ``nw[i]``.
        phi_range (psst.Range): The minimum, maximum, and number of values in the range
          of reduced concentration values (``phi_range.log_scale`` should be True).
        nw_range (psst.Range): The minimum, maximum, and number of values in the range
          of weight-average DP values (``nw_range.log_scale`` should be True).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Three arrays representing the
          concentration value of each "pixel" along axis 0 (1D), the DP value of each
          "pixel" along axis 1 (1D), and the average values of specific viscosity at
          the given concentrations and DPs (2D). Simply put, the value of viscosity at
          index ``(i, j)`` approximately corresponds to the reduced concentration at
          index ``i`` and the DP at index ``j``.
    """
    assert phi_range.shape is not None and nw_range.shape is not None
    shape = (phi_range.shape, nw_range.shape)
    visc_out = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.uint32)

    log_phi_bins: np.ndarray = np.linspace(
        np.log10(phi_range.min_value),
        np.log10(phi_range.max_value),
        shape[0],
        endpoint=True,
    )
    phi_bin_edges = np.zeros(log_phi_bins.shape[0] + 1)
    phi_bin_edges[(0, -1)] = 10 ** log_phi_bins[(0, -1)]
    phi_bin_edges[1:-1] = 10 ** ((log_phi_bins[1:] + log_phi_bins[:-1]) / 2)
    phi_indices = np.digitize(phi_data, phi_bin_edges)

    log_nw_bins: np.ndarray = np.linspace(
        np.log10(nw_range.min_value),
        np.log10(nw_range.max_value),
        shape[1],
        endpoint=True,
    )
    nw_bin_edges = np.zeros(log_nw_bins.shape[0] + 1)
    nw_bin_edges[(0, -1)] = 10 ** log_nw_bins[(0, -1)]
    nw_bin_edges[1:-1] = 10 ** ((log_nw_bins[1:] + log_nw_bins[:-1]) / 2)
    nw_indices = np.digitize(nw_data, nw_bin_edges)

    data = np.stack((phi_indices, nw_indices, visc_data), axis=1)
    for p, n, v in data:
        visc_out[p, n] += v
        counts[p, n] += 1

    counts = np.maximum(counts, np.ones_like(counts))
    visc_out /= counts

    return 10**log_phi_bins, 10**log_nw_bins, visc_out


def transform_data(
    reduced_conc: np.ndarray,
    degree_polym: np.ndarray,
    spec_visc: np.ndarray,
    phi_range: psst.Range,
    nw_range: psst.Range,
    visc_range: psst.Range,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Transform the raw, reduced data into two 2D tensors of reduced, normalized
    viscosity data ready for use in a neural network.

    Args:
        reduced_conc (np.ndarray): Reduced concentration data :math:`\varphi=cl^3`.
        degree_polym (np.ndarray): Weight-average DP data :math:`N_w`.
        spec_visc (np.ndarray): Specific viscosity raw data
        phi_range (psst.Range): Range of values specifying the value of reduced
          concentration for each grid index along axis 0.
        nw_range (psst.Range): Range of values specifying the value of weight-average
          DP for each grid index along axis 1.
        visc_range (psst.Range): Range of values specifying the maximum and minimum
          values of specific viscosity.

    Returns:
        tuple[psst.NormedTensor, psst.NormedTensor]: The reduced specific viscosities
          :math:`\eta_{sp}/N_w \phi^{1.31}` and :math:`\eta_{sp}/N_w \phi^2`.
    """
    phi_arr, nw_arr, visc_arr = process_data_to_grid(
        reduced_conc,
        degree_polym,
        spec_visc,
        phi_range,
        nw_range,
    )

    bg_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** (
        1 / (3 * GOOD_EXP - 1)
    )
    bth_denom = nw_arr.reshape(1, -1) * phi_arr.reshape(-1, 1) ** 2

    visc_normed_bg = (
        psst.NormedTensor.create_from_numpy(
            visc_arr / bg_denom,
            min_value=visc_range.min_value / bg_denom.max(),
            max_value=visc_range.max_value / bg_denom.min(),
            log_scale=visc_range.log_scale,
        )
        .normalize()
        .clamp_(0, 1)
    )
    visc_normed_bth = (
        psst.NormedTensor.create_from_numpy(
            visc_arr / bth_denom,
            min_value=visc_range.min_value / bth_denom.max(),
            max_value=visc_range.max_value / bth_denom.min(),
            log_scale=visc_range.log_scale,
        )
        .normalize()
        .clamp_(0, 1)
    )

    return visc_normed_bg, visc_normed_bth


def inference_models(
    bg_model: torch.nn.Module,
    bth_model: torch.nn.Module,
    visc_normed_bg: torch.Tensor,
    visc_normed_bth: torch.Tensor,
    bg_range: psst.Range,
    bth_range: psst.Range,
) -> tuple[float, float]:
    """Run the processed viscosity data through the models to gain the inference
    results for :math:`B_g` and :math:`B_{th}`.

    Args:
        model_type (str): One of the pre-trained models from the keys of MODELS.
        bg_state_dict (dict): The state dictionary of the bg model.
        bth_state_dict (dict): The state dictionary of the bth model.
        visc_normed_bg (psst.NormedTensor): The reduced and normalized viscosity data
          for the bg model.
        visc_normed_bth (psst.NormedTensor): The reduced and normalized viscosity data
          for the bth model.
        bg_range (psst.Range): The minimum and maximum of the bg parameter (used for
          normalization and unnormalization).
        bth_range (psst.Range): The minimum and maximum of the bg parameter (used for
          normalization and unnormalization).

    Returns:
        tuple[float, float]: The inferred values of :math:`B_g` and :math:`B_{th}`,
          respectively.
    """
    bg = psst.NormedTensor.create_from_range(bg_range, (1,)).normalize()
    bth = psst.NormedTensor.create_from_range(bth_range, (1,)).normalize()

    bg[:] = bg_model(visc_normed_bg)
    bth[:] = bth_model(visc_normed_bth)

    return bg.unnormalize().item(), bth.unnormalize().item()


def fit_func(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    return nw_over_g_lamda_g * (1 + (nw_over_g_lamda_g / pe**2)) ** 2


def fit_func_jac(nw_over_g_lamda_g: np.ndarray, pe: float) -> np.ndarray:
    return -2 * nw_over_g_lamda_g**2 / pe**3 - 4 * nw_over_g_lamda_g**3 / pe**5


def combo_case(
    bg: float,
    bth: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    # ne/pe**2 == g*lam_g
    ne_over_pe2 = np.minimum(
        bg ** (0.056 / 0.528 / 0.764) * bth ** (0.944 / 0.528) * phi ** (-1 / 0.764),
        np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2),
    )
    lamda = np.minimum(1, bth**4 / phi)

    popt, pcov = curve_fit(
        fit_func,
        nw / ne_over_pe2,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def bg_only_case(
    bg: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    g = (bg**3 / phi) ** (1 / 0.764)
    lamda = np.minimum(1, phi ** (-0.236 / 0.764) * bg ** (2 / (0.412 * 0.764)))
    popt, pcov = curve_fit(
        fit_func,
        nw / g,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def bth_only_case(
    bth: float,
    phi: np.ndarray,
    nw: np.ndarray,
    spec_visc: np.ndarray,
) -> PeResult:
    # ne/pe**2 == g*lam_g
    ne_over_pe2 = np.minimum(bth**2 / phi ** (4 / 3), bth**6 / phi**2)
    lamda = np.minimum(1, bth**4 / phi)

    popt, pcov = curve_fit(
        fit_func,
        nw / ne_over_pe2,
        lamda * spec_visc,
        p0=(8.0,),
        bounds=(2.0, 40.0),
        jac=fit_func_jac,
    )

    return PeResult(popt[0], pcov[0][0])


def evaluate_dataset(
    range_config: psst.RangeConfig,
    eval_config: EvaluationConfig,
    device: torch.device = torch.device("cpu"),
) -> InferenceResult:
    """Perform an evaluation of experimental data given one previously trained PyTorch
    model for each of the :math:`B_g` and :math:`B_{th}` parameters.

    Args:
        bg_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_g`
          parameter.
        bth_model (torch.nn.Module): A pretrained model for evaluating the :math:`B_th`
          parameter.
        generator_config (psst.GeneratorConfig): A set of configuration settings.
          Only the Range attributes for phi, nw, visc, bg, and bth are used.
        concentration_gpL (np.ndarray): Experimental concentration data in units of
          grams per mole (1D numpy array).
        mol_weight_kgpmol (np.ndarray): Experimental molecular weight data in units of
          kilograms per mole (1D numpy array).
        specific_viscosity (np.ndarray): Experimental specific viscosity data in
          dimensionless units (1D numpy array).
        repeat_unit_length_nm (float): The projection length of the monomer/repeat unit
          in units of nanometers.
        repeat_unit_mass_gpmol (float): The molar mass of the repeat unit in units of
          grams per mole.

    Note:
        The experimental data arrays should be 1-dimensional and the same length, such
        that ``specific_viscosity[i]`` corresponds to a measurement taken at monomer
        concentration ``concentration_gpL[i]`` of a solution of chains with weight-
        average molecular weight ``mol_weight_kgpmol[i]``.

    Returns:
        InferenceResult: The results of the model inferences, complete with estimates
          for :math:`B_g` and :math:`B_{th}`; three estimates of :math:`P_e` with
          fitting uncertainties, one each for the case where both :math:`B_g` and
          :math:`B_{th}` are valid, the case where only :math:`B_g` is valid (athermal
          solvent), and the case where only :math:`B_{th}` is valid (theta solvent);
          the reduced concentration :math:`\\varphi=cl^3`; the weight-average degree of
          polymerization; and the unaltered specific viscosity.
    """

    reduced_conc, degree_polym = reduce_data(
        eval_config.concentration,
        eval_config.molecular_weight,
        eval_config.repeat_unit,
    )
    visc = eval_config.specific_viscosity
    visc_normed_bg, visc_normed_bth = transform_data(
        reduced_conc,
        degree_polym,
        visc,
        range_config.phi_range,
        range_config.nw_range,
        range_config.visc_range,
    )

    bg, bth = inference_models(
        eval_config.bg_model,
        eval_config.bth_model,
        visc_normed_bg,
        visc_normed_bth,
        range_config.bg_range,
        range_config.bth_range,
    )

    pe_combo = combo_case(bg, bth, reduced_conc, degree_polym, visc)
    pe_bg_only = bg_only_case(bg, reduced_conc, degree_polym, visc)
    pe_bth_only = bth_only_case(bth, reduced_conc, degree_polym, visc)

    return InferenceResult(
        bg,
        bth,
        pe_combo,
        pe_bg_only,
        pe_bth_only,
        reduced_conc,
        degree_polym,
        visc,
    )
