r"""Efficiently generate batched samples of molecular parameters in a uniform
distribution, construct specific viscosity as a function of concentration and 
degree of polymerization of the polymer solution from the samples of molecular
parameters, and yield the normalized values (between 0 and 1).
"""
from __future__ import annotations
import logging
from typing import Optional
from warnings import warn

import torch

import psst
from psst import Range, Parameter

__all__ = ["SampleGenerator"]


class SampleGenerator:
    """Procedurally generates batches of viscosity curves.

    The resulting object is callable and iterable with similar functionality to the
    built-in ``range`` function. It takes one parameter, the number of batches/cycles,
    and the four-element tuple it generates consists of

    1. The normalized, reduced viscosity with shape
    ``(batch_size, phi_range.num, nw_range.num)``. This is considered as a batch of
    2D images that can be used to train a neural network (e.g.,
    ``psst.models.Inception3``).

    2. The generated values of :math:`B_g` with shape ``(batch_size,)``.

    3. The generated values of :math:`B_{th}` (same shape as for :math:`B_g`).

    4. The generated values of :math:`P_e` (same shape again).

    Example:
        >>> import psst
        >>> from psst.models import Inception3
        >>>
        >>> model = Inception3()
        >>> config = psst.getConfig("config.yaml")
        >>> gen_samples = psst.SampleGenerator(**config.generator_config)
        >>> num_batches = (
        ...     config.run_config.num_samples_test
        ...     // config.generator_config.batch_size
        ... )
        >>> for viscosity, bg, bth, pe in gen_samples(num_batches):
        >>>     pred_bg = model(viscosity)

    Args:
        batch_size (int): The number of values of Bg, Bth, and Pe (and thus
          the number of viscosity curves) to generate.
        parameter (:class:`Parameter`): Either ``"Bg"`` or ``"Bth"`` for good
          solvent behavior or thermal blob behavior, respectively.
        phi_range (:class:`Range`): The min, max and number of reduced
          concentration values to use for the viscosity curves.
        nw_range (:class:`Range`): As with ``phi_range``, but for values of degree
          of polymerization.
        visc_range (:class:`Range`): The minimum and maximum values of viscosity
          to use for normalization.
        bg_range (:class:`Range`): The minimum and maximum values of the good
          solvent blob parameter to use for normalization and generation.
        bth_range (:class:`Range`): The minimum and maximum values of the thermal
          blob parameter to use for normalization and generation.
        pe_range (:class:`Range`): The minimum and maximum values of the
          entanglement packing number to use for normalization and generation.
        noise_factor (float): A Gaussian distribution of noise is added to each value
          of the viscosity before reducing or normalizing. That noise is multiplied by
          this factor. When set to ``0``, the resulting viscosity values follow the
          given equations exactly. Defaults to ``0.05``.
        device (torch.device, optional): Device on which to create batches and compute
          samples. Defaults to ``torch.device("cpu")``.
        generator (torch.Generator, optional): Random number generator to use for
          values of :math:`Bg`, :math:`Bth`, and :math:`Pe`. Most useful during
          testing, allowing a fixed seed to be used. A value of ``None`` creates a
          generic torch.Generator instance. Defaults to ``None``.
    """

    def __init__(
        self,
        parameter: Parameter,
        range_config: psst.RangeConfig,
        gen_config: psst.GeneratorConfig,
        *,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self._log = logging.getLogger("psst.main")
        self._log.info("Initializing SampleGenerator")
        self._log.debug("SampleGenerator: device = %s", str(device))

        self.device = device
        self.generator = generator

        self.parameter = parameter
        self.batch_size = gen_config.batch_size
        self.noise_factor = gen_config.noise_factor
        self.num_nw_choices = gen_config.num_nw_choices
        self.num_nw_to_select = gen_config.num_nw_to_select
        self.num_phi_to_select = gen_config.num_phi_to_select

        self.phi_range = range_config.phi_range
        self.nw_range = range_config.nw_range
        self.visc_range = range_config.visc_range
        self.bg_range = range_config.bg_range
        self.bth_range = range_config.bth_range
        self.pe_range = range_config.pe_range

        self._validate_args()
        if self.noise_factor == 0:
            self._add_noise = self._noop
        else:
            self._add_noise = self._add_noise_default

        self._init_random()
        self._init_grid_tensors()
        self._init_normed_tensors()

        self._num_batches: Optional[int] = None
        self._index: int = 0

        self._log.debug("Completed initialization")

    def _validate_args(self):
        if not isinstance(self.parameter, Parameter):
            raise TypeError("Argument `parameter` must be of type ``psst.Parameter``.")

        try:
            self.batch_size = int(self.batch_size)
        except ValueError:
            raise TypeError(
                "Argument `batch_size` must be an int (or convertable to an int)"
            )
        if self.batch_size <= 0:
            raise ValueError("Argument `batch_size` must be positive")

        if not isinstance(self.phi_range, psst.Range):
            raise TypeError("Must be of type psst.Range")
        if not isinstance(self.nw_range, psst.Range):
            raise TypeError("Must be of type psst.Range")
        if not isinstance(self.visc_range, psst.Range):
            raise TypeError("Must be of type psst.Range")
        if not isinstance(self.bg_range, psst.Range):
            raise TypeError("Must be of type psst.Range")
        if not isinstance(self.bth_range, psst.Range):
            raise TypeError("Must be of type psst.Range")
        if not isinstance(self.pe_range, psst.Range):
            raise TypeError("Must be of type psst.Range")

        try:
            self.noise_factor = float(self.noise_factor)
        except ValueError:
            raise TypeError(
                "Argument `noise_factor` must be a float (or convertable to a float)"
            )
        if abs(self.noise_factor) >= 1.0:
            warn(
                "Absolute value of parameter `noise_factor` is 1 or greater."
                " This may result in negative values of specfic viscosity."
            )

        if isinstance(self.device, str):
            self.device: torch.device = torch.device(self.device)

        if not (self.generator is None or isinstance(self.generator, torch.Generator)):
            raise TypeError(
                "If specified, argument `generator` must be an instance of"
                " `torch.Generator` or a subclass thereof."
            )

    def _init_random(self):
        if self.generator is None:
            self.generator = torch.Generator()
        self.generator.seed()
        self._log.debug("Initialized random number generator")

        assert self.phi_range.shape is not None
        assert self.nw_range.shape is not None

        self._noise = torch.zeros(
            (self.batch_size, self.phi_range.shape, self.nw_range.shape)
        )

    def _init_grid_tensors(self):
        self._phi = self.phi_range.create_grid().to(device=self.device)
        self._phi = self._phi.reshape(1, -1, 1)
        self._log.debug("Initialized self._phi with size %s", str(self._phi.shape))

        self._nw = self.nw_range.create_grid().to(device=self.device)
        self._nw = self._nw.reshape(1, 1, -1)
        self._log.debug("Initialized self._nw with size %s", str(self._nw.shape))

    def _init_normed_tensors(self):
        if self.parameter is Parameter.bg:
            self._get_single_samples = self._get_bg_samples
            self._denominator = self._nw * self._phi ** (1 / 0.764)
            self._log.debug("Initialized Bg-specific members")
        else:
            self._get_single_samples = self._get_bth_samples
            self._denominator = self._nw * self._phi**2
            self._log.debug("Initialized Bth-specific members")

        self.reduced_visc_range = Range(
            self.visc_range.min_value / self._denominator.max().item(),
            self.visc_range.max_value / self._denominator.min().item(),
            log_scale=self.visc_range.log_scale,
        )
        self._visc = torch.zeros(
            self.batch_size, self._phi.shape[1], self._nw.shape[2], device=self.device
        )
        self._log.debug("Initialized self._visc with size %s", str(self._visc.shape))

        self._bg = torch.zeros(self.batch_size, 1, 1)
        self._bth = torch.zeros(self.batch_size, 1, 1)
        self._pe = torch.zeros(self.batch_size, 1, 1)
        self._log.debug(
            "Initialized self._bg, self._bth, self._pe each with size %s",
            str(self._bg.shape),
        )

    def __call__(self, num_batches: int):
        if not isinstance(num_batches, int):
            raise TypeError(
                "Argument to SampleGenerator object must be non-negative integer"
            )
        if num_batches < 0:
            raise ValueError(
                "Argument to SampleGenerator object must be non-negative integer"
            )
        self._num_batches = num_batches
        return self

    def __iter__(self):
        if self._num_batches is None:
            raise SyntaxError(
                "SampleGenerator object must be called with an argument"
                " (number of iterations) to be iterable"
            )
        self._index = 0
        self._log.info("Starting %d iterations", self._num_batches)
        return self

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._num_batches is None:
            raise SyntaxError(
                "SampleGenerator object must be called with an argument"
                " (number of iterations) to be iterable"
            )

        if self._index >= self._num_batches:
            self._log.info("Completed all batches")
            self._num_batches = None
            raise StopIteration

        self._index += 1
        self._log.debug("Generating batch %6d/%d", self._index, self._num_batches)

        self.bg_range.generate(self._bg)
        self.bth_range.generate(self._bth)
        self.pe_range.generate(self._pe)
        self._log.debug("Sampled values for Bg, Bth, Pe")

        self._compute_visc(self._bg, self._bth, self._pe, self._visc)

        self._add_noise(self._visc)
        self._visc.div_(self._denominator)
        self._log.debug("Added noise and reduced viscosity")

        self.bg_range.normalize(self._bg)
        self.bth_range.normalize(self._bth)
        self.pe_range.normalize(self._pe)
        self.reduced_visc_range.normalize(self._visc)
        self._trim(self._visc)

        self._log.debug("Normalized results")

        return (
            self._visc.view(self.batch_size, 1, self._phi.size(1), self._nw.size(2)),
            self._bg.view(-1),
            self._bth.view(-1),
            self._pe.view(-1),
        )

    def _compute_visc(
        self,
        bg: torch.Tensor,
        bth: torch.Tensor,
        pe: torch.Tensor,
        visc: torch.Tensor,
    ):
        is_combo = torch.randint(
            2,
            size=(self.batch_size,),
            device=self.device,
            generator=self.generator,
            dtype=torch.bool,
        )

        self._log.debug("Chose combo and single samples")
        visc[is_combo] = self._get_combo_samples(
            bg[is_combo], bth[is_combo], pe[is_combo]
        )
        self._log.debug("Computed combo samples")

        visc[~is_combo] = self._get_single_samples(
            bg[~is_combo], bth[~is_combo], pe[~is_combo]
        )
        self._log.debug("Computed single samples")

    def _get_combo_samples(self, bg: torch.Tensor, bth: torch.Tensor, pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = torch.minimum(
            (bg**3 / self._phi) ** (1 / 0.764), bth**6 / self._phi**2
        )
        Ne = pe**2 * torch.minimum(
            bg ** (0.056 / (0.528 * 0.764))
            * bth ** (0.944 / 0.528)
            / self._phi ** (1 / 0.764),
            torch.minimum(
                (bth / self._phi ** (2 / 3)) ** 2, (bth**3 / self._phi) ** 2
            ),
        )
        return (
            self._nw
            * (1 + (self._nw / Ne)) ** 2
            * torch.minimum(1 / g, self._phi / bth**2)
        )

    def _get_bg_samples(self, bg: torch.Tensor, bth: torch.Tensor, pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = (bg**3 / self._phi) ** (1 / 0.764)
        Ne = pe**2 * g
        return self._nw / g * (1 + (self._nw / Ne)) ** 2

    def _get_bth_samples(self, bg: torch.Tensor, bth: torch.Tensor, pe: torch.Tensor):
        # print(Bg.shape, Bth.shape, Pe.shape, self.phi.shape, self.Nw.shape)
        g = bth**6 / self._phi**2
        Ne = pe**2 * torch.minimum(
            (bth / self._phi ** (2 / 3)) ** 2, (bth**3 / self._phi) ** 2
        )
        return (
            self._nw
            * (1 + (self._nw / Ne)) ** 2
            * torch.minimum(1 / g, self._phi / bth**2)
        )

    def _add_noise_default(self, visc: torch.Tensor):
        visc *= 1 + self.noise_factor * self._noise.normal_(generator=self.generator)

    def _noop(self, visc: torch.Tensor):
        pass

    # @torch.compile
    def _trim(
        self,
        visc: torch.Tensor,
        num_nw_to_select: int = 12,
        num_nw_choices: int = 48,
        num_phi_to_select: int = 65,
    ):
        num_batches = visc.shape[0]
        indices = torch.arange(self._nw.size(2), device=self.device)

        self._log.debug("Trimming Nw rows")
        for i in range(num_batches):
            # Get first max_num_rows_nonzero indices of Nw rows with most values of phi
            num_nonzero_per_phi = torch.sum(visc[i] > 0, dim=0)
            top_nw_inds = torch.argsort(num_nonzero_per_phi, descending=True, dim=0)[
                :num_nw_choices
            ]
            # Select max_num_rows_select of those Nw rows, set the rest to visc = 0
            selected = torch.randint(
                0,
                top_nw_inds.shape[0],
                size=(num_nw_to_select,),
                device=self.device,
                generator=self.generator,
            )
            deselected_rows = torch.isin(indices, top_nw_inds[selected], invert=True)
            visc[i, deselected_rows, :] = 0.0

        deselected_rows = torch.zeros(
            (num_phi_to_select,), dtype=torch.int, device=self.device
        )
        self._log.debug("Trimming phi rows")
        for i in range(num_batches):
            # Select num_concentrations_per_sample rows of phi
            # (that aren't entirely 0), set the rest to visc = 0
            # TODO: ensure that deselected_rows doesn't double count or undercount...
            nonzero_rows = visc[i].nonzero(as_tuple=True)[0]
            lo = int(nonzero_rows.min().item())
            hi = int(nonzero_rows.max().item())
            deselected_rows.random_(lo, hi + 1, generator=self.generator)
            visc[i, deselected_rows, :] = 0.0
