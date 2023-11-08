import matplotlib.pyplot as plt
import numpy as np

from psst.evaluation.evaluate import InferenceResult


def raw_plot(
    arr: np.ndarray, markers: list[str], colors: list[str], masks: list[np.ndarray]
):
    plt.figure("Raw")
    for mask, m, c in zip(masks, markers, colors):
        phi, nw, visc = arr[:, mask]
        plt.plot(phi, visc, ls="None", marker=m, c=c, ms=12, label=f"$N_w={nw[0]}$")


def plot_131(
    arr: np.ndarray,
    markers: list[str],
    colors: list[str],
    masks: list[np.ndarray],
    bg: float,
):
    plt.figure("Bg Plateau")
    for mask, m, c in zip(masks, markers, colors):
        phi, nw, visc = arr[:, mask]
        plt.plot(
            phi,
            visc / nw / phi ** (1 / (3 * 0.588 - 1)),
            ls="None",
            marker=m,
            c=c,
            ms=12,
            label=f"$N_w={nw[0]}$",
        )

    plt.plot([])

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("$cl^3$")
    plt.ylabel("$\\eta_{sp}/N_w(cl^3)^{1.31}$")


def plot_2(data: list[np.ndarray], markers: list[str], colors: list[str], bth: float):
    plt.figure("Bth Plateau")
    for (phi, nw, visc), m, c in zip(data, markers, colors):
        plt.plot(
            phi,
            visc / nw / phi**2,
            ls="None",
            marker=m,
            c=c,
            ms=12,
            label=f"$N_w={nw}$",
        )


def plot(result: InferenceResult):
    arr = np.stack(
        (result.reduced_conc, result.degree_polym, result.specific_visc), axis=0
    )
    indices = np.argsort(result.degree_polym, axis=0)
    arr = arr[:, indices]
    masks = list()
    data = list()
    for n in np.unique(arr[1]):
        masks.append(arr[1] == n)
        data.append(arr[:, masks[-1]])

    plt.figure("")
