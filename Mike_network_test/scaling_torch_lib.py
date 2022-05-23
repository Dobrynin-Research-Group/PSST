import collections
from typing import Tuple
import numpy as np
import torch

Param = collections.namedtuple('Param', ('min', 'max'))

PHI = Param(1e-6, 1e-2)
NW = Param(10, 3e5)
ETA_SP = Param(torch.tensor(1), torch.tensor(3.5e6))

BG = Param(0.3, 1.6)
BTH = Param(0.2, 0.9)
PE = Param(8, 20)


# TODO: implement modified surface_generator to create correlated Bg, Bth
# This reflects experimental data, where we see Bg \approx Bth + constant
# for systems with both Bg and Bth (all systems have Bg and Bth, but for
# some we can only see one regime).

def unnormalize_params(y: torch.Tensor) -> Tuple[torch.Tensor]:
    """Simple linear normalization.
    """
    Bg = y[:, 0] * (BG.max - BG.min) + BG.min + 1e-4
    Bth = y[:, 1] * (BTH.max - BTH.min) + BTH.min + 1e-4
    Pe = y[:, 2] * (PE.max - PE.min) + PE.min + 1e-4
    return Bg, Bth, Pe


def normalize_visc(eta_sp: torch.Tensor) -> torch.Tensor:
    """Add noise, cap the values, take the log, then normalize.
    """
    eta_sp += eta_sp * 0.05 * torch.normal(
        torch.zeros_like(eta_sp),
        torch.ones_like(eta_sp)
    )
    eta_sp = torch.fmin(eta_sp, ETA_SP.max)
    eta_sp = torch.fmax(eta_sp, ETA_SP.min)
    return (torch.log(eta_sp) - torch.log(ETA_SP.min)) / \
        (torch.log(ETA_SP.max) - torch.log(ETA_SP.min))


def surface_generator(
    num_batches: int, batch_size: int, device: torch.device,
    resolution: Tuple[int] = (32, 32)
) -> Tuple[torch.Tensor]:
    """Generate `batch_size` surfaces, based on ranges for `Bg`, `Bth`, and
    `Pe`, to be used in a `for` loop.

    It defines the resolution of the surface based on either user input
    (keyword argument `resolution`). It then generates random values for `Bg`,
    `Bth`, and `Pe`, evaluates the `(phi, Nw, eta_sp)` surface, and normalizes
    the result. The normalized values of `eta_sp` and `(Bg, Bth, Pe)` are
    yielded as `X` and `y` for use in a neural network.

    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `batch_size` (`int`) : The length of the generated values.
        `device` (`torch.device`): The device to do computations on.
        `resolution` (tuple of `int`s) : The shape of the last two dimensions
            of the generated values.

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Generated,
            normalized values of `eta_sp` at indexed `phi` and `Nw`.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """

    ETA_SP.min.to(dtype=torch.float, device=device)
    ETA_SP.max.to(dtype=torch.float, device=device)

    # Create tensors for phi (concentration) and Nw (chain length)
    # Both are meshed and tiled to cover a 3D tensor of size
    # (batch_size, *resolution) for simple, element-wise operations
    phi = torch.tensor(np.geomspace(
        PHI.min,
        PHI.max,
        resolution[0],
        endpoint=True
    ), dtype=torch.float, device=device)

    Nw = torch.tensor(np.geomspace(
        NW.min,
        NW.max,
        resolution[1],
        endpoint=True
    ), dtype=torch.float, device=device)

    phi, Nw = torch.meshgrid(phi, Nw, indexing='xy')
    phi = torch.tile(phi, (batch_size, 1, 1))
    Nw = torch.tile(Nw, (batch_size, 1, 1))

    def generate_surfaces(
            Bg: torch.Tensor, Bth: torch.Tensor, Pe: torch.Tensor
    ) -> torch.Tensor:
        # First, tile params to match shape of phi and Nw for simple,
        # element-wise operations
        shape = torch.Size((1, *(phi.size()[1:])))
        Bg = torch.tile(Bg.reshape((batch_size, 1, 1)), shape)
        Bth = torch.tile(Bth.reshape((batch_size, 1, 1)), shape)
        Pe = torch.tile(Pe.reshape((batch_size, 1, 1)), shape)

        # Number of repeat units per correlation blob
        # Only defined for c < c**
        # Minimum accounts for crossover at c = c_th
        g = torch.fmin(
            Bg**(3/0.764) / phi**(1/0.764),
            Bth**6 / phi**2
        )

        # Number of repeat units per entanglement strand
        # Universal definition of Ne accounts for both
        # Kavassalis-Noolandi and Rubinstein-Colby scaling
        Ne = Pe**2 * g * torch.fmin(
            torch.tensor([1], device=device), torch.fmin(
                (Bth / Bg)**(2/(6*0.588 - 3)) / Bth**2,
                Bth**4 * phi**(2/3)
            )
        )

        # Specific viscosity crossover function from Rouse to entangled regimes
        # Viscosity crossover function for entanglements
        # Minimum accounts for crossover at c = c**
        eta_sp = Nw * (1 + (Nw / Ne)**2) * torch.fmin(
            1/g,
            phi / Bth**2
        )

        return eta_sp

    for _ in range(num_batches):
        y = torch.rand((batch_size, 3), device=device, dtype=torch.float)
        Bg, Bth, Pe = unnormalize_params(y)
        eta_sp = generate_surfaces(Bg, Bth, Pe)
        X = normalize_visc(eta_sp).to(torch.float)
        yield X, y


def voxel_image_generator(
    num_batches: int, batch_size: int, device: torch.device,
    resolution: Tuple[int] = (32, 32, 32)
) -> Tuple[torch.Tensor]:
    """Uses surface_generator to generate a surface with a resolution one more,
    then generates a 3D binary array dictating whether or not the surface
    passes through a given voxel. This is determined using the facts that:
     - The surface is continuous
     - The surface monotonically increases with increasing phi and Nw
     - phi and Nw increase with increasing index
    If the voxel corner at index (i, j, k+1) is greater than the surface value
    at (i, j), and if the corner at index (i+1, j+1, k) is less than the value
    at (i+1, j+1), then the surface passes through.
    Input:
        `num_batches` (`int`) : The number of loops to be iterated through.
        `batch_size` (`int`) : The length of the generated values.
        `device` (`torch.device`): The device to do computations on.
        `resolution` (tuple of `int`s) : The shape of the last three dimensions
            of the generated values.

    Output:
        `X` (`torch.Tensor` of size `(batch_size, *resolution)`) : Binary array
            dictating whether or not the surface passes through the indexed
            voxel.
        `y` (`torch.Tensor` of size `(batch_size, 3)`) : Generated, normalized
            values of `(Bg, Bth, Pe)`.
    """
    s_res = (resolution[0] + 1, resolution[1] + 1)

    eta_sp = normalize_visc(torch.tensor(
        np.geomspace(ETA_SP.min, ETA_SP.max, resolution[2]+1, endpoint=True),
        dtype=torch.float,
        device=device
    ))

    for X, y in surface_generator(
        num_batches, batch_size, device, resolution=s_res
    ):
        surf = torch.tile(
            X.reshape((batch_size, *s_res, 1)),
            (1, 1, 1, resolution[2]+1)
        )

        # if <= or >=, we would include capped values, which we don't want
        image = torch.logical_and(
            surf[:, :-1, :-1, :-1] < eta_sp[1:],
            surf[:, 1:, 1:, 1:] > eta_sp[:-1]
        ).to(dtype=torch.float, device=device)

        yield image, y
