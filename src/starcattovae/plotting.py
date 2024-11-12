from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# 1, 2, 3, sigmas
SIGMA_QUANTS = [0.68, 0.96, 0.99]

def plot_waveform_grid(
    signals: np.ndarray,
    max_value: float,
    num_cols: int = 4,
    num_rows: int = 2,
    fname: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(
        num_cols, num_rows, figsize=(num_cols * 4, num_rows * 3)
    )

    axes = axes.flatten()

    # plot each signal on a separate subplot
    for i, ax in enumerate(axes):
        x = [i / 4096 for i in range(0, 256)]
        x = [value - (53 / 4096) for value in x]
        y = signals[i, :, :].flatten()
        y = y * max_value
        ax.set_ylim(-600, 300)
        ax.plot(x, y, color="red")

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True)

        # remove y-axis ticks for the right-hand column
        if i % num_cols == num_cols - 1:
            ax.yaxis.set_ticklabels([])

        # remove x-axis tick labels for all but the bottom two plots
        if i <= 11:
            ax.xaxis.set_ticklabels([])

    for i in range(512, 8 * 4):
        fig.delaxes(axes[i])

    fig.supxlabel('time (s)', fontsize=32)
    fig.supylabel('hD (cm)', fontsize=32)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, axes

def plot_reconstruction(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_value: float,
    fname: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(15, 4))

    x = [i / 4096 for i in range(0, 256)]
    x = [value - (53 / 4096) for value in x]

    # plot the original signal
    y_original = original.flatten() * max_value
    ax.plot(x, y_original, color="blue", label="Original Signal")
    
    # plot the reconstructed signal
    y_reconstructed = reconstructed.flatten() * max_value
    ax.plot(x, y_reconstructed, color="red", label="Reconstructed Signal")

    ax.set_ylim(-600, 300)
    ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax.grid(True)
    ax.set_title("Original and Reconstructed Signals")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hD (cm)")
    ax.legend()

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, ax

def plot_loss(
    losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    axes.plot(losses, label="VAE Loss (ELBO)")
    axes.set_xlabel("Batch", size=20)
    axes.set_ylabel("Loss", size=20)
    # axes.set_ylim(0, 100)
    axes.legend(fontsize=16)
    plt.tight_layout()
    if fname:
        plt.savefig(fname)
    return axes.get_figure()

