from typing import List, Optional, Tuple

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torch

import src.starcattovae.nn.encoder as Encoder
import src.starcattovae.nn.decoder as Decoder

# 1, 2, 3, sigmas
SIGMA_QUANTS = [0.68, 0.96, 0.99]

def plot_waveform_grid(
    signals: np.ndarray,
    max_value: float,
    num_cols: int = 2,
    num_rows: int = 4,
    fname: str = None,
) -> Tuple[plt.Figure, plt.Axes]:
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(10, 15)
    )

    axes = axes.flatten()

    # plot each signal on a separate subplot
    for i, ax in enumerate(axes):
        x = [i / 4096 for i in range(0, 256)]
        x = [value - (53 / 4096) for value in x]
        y = signals[i].flatten()
        y = y * max_value
        ax.set_ylim(-600, 300)
        ax.plot(x, y, color="red")

        ax.axvline(x=0, color="black", linestyle="--", alpha=0.5)
        ax.grid(True)

        # remove y-axis ticks for the right-hand column
        if i % num_cols == num_cols - 1:
            ax.yaxis.set_ticklabels([])

        # remove x-axis tick labels for all but the bottom two plots
        if i < num_cols * (num_rows - 1):
            ax.xaxis.set_ticklabels([])

    # for i in range(512, 8 * 4):
    #     fig.delaxes(axes[i])

    fig.supxlabel('time (s)', fontsize=32)
    fig.supylabel('hD (cm)', fontsize=32)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()
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
    ax.plot(x, y_reconstructed, color="orange", label="Decoder Reconstructed Signal")

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

    plt.show()
    return fig, ax

def plot_loss(
    losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()
    axes.plot(losses, label="Total Training Loss)")
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
    # axes.set_ylim(0, 100)
    axes.legend(fontsize=16)
    
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
    
    return axes.get_figure()

def plot_training_validation_loss(
    losses: List[float],
    validation_losses: List[float],
    fname: str = None,
    axes: plt.Axes = None,
):
    if axes is None:
        fig = plt.figure(figsize=(10, 6))
        axes = fig.gca()

    axes.plot(losses, label="Total Training Loss", color='orange')
    axes.plot(validation_losses, label="Total Validation Loss", color='grey')
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
    axes.legend(fontsize=16)
    
    plt.tight_layout()

    if fname:
        plt.savefig(fname)
    
    return axes.get_figure()

def plot_latent_morphs(
    encoder: Encoder, 
    decoder: Decoder, 
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    steps=10
):
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        mean_1, _ = encoder(signal_1)
        mean_2, _ = encoder(signal_2)

        interpolated_latents = [mean_1 * (1 - alpha) + mean_2 * alpha for alpha in np.linspace(0, 1, steps)]
        morphed_signals = [decoder(latent).cpu().detach().numpy() for latent in interpolated_latents]

    num_plots = steps + 2
    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))
    axes = axes.flatten()

    # X-axis values (shared across all plots)
    x_vals = [i / 4096 for i in range(0, 256)]
    x_vals = [value - (53 / 4096) for value in x_vals]

    # Plot signal_1 (blue)
    y1 = signal_1.cpu().detach().numpy().flatten() * max_value
    axes[0].plot(x_vals, y1, color="blue")
    axes[0].set_ylim(-600, 300)
    axes[0].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[0].grid(True)
    axes[0].set_title("Original Signal 1 (Start)")

    # Plot the interpolated signals (red)
    for i, signal in enumerate(morphed_signals):
        y_interp = signal.flatten() * max_value
        # y_interp = signal.flatten()
        axes[i + 1].plot(x_vals, y_interp, color="red")
        axes[i + 1].set_ylim(-600, 300)
        axes[i + 1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
        axes[i + 1].grid(True)
        axes[i + 1].set_title(f"Interpolated Signal {i + 1}")

    # Plot signal_2 (blue)
    y2 = signal_2.cpu().detach().numpy().flatten() * max_value
    axes[-1].plot(x_vals, y2, color="blue")
    axes[-1].set_ylim(-600, 300)
    axes[-1].axvline(x=0, color="black", linestyle="--", alpha=0.5)
    axes[-1].grid(True)
    axes[-1].set_title("Original Signal 2 (End)")

    # Keep all tick labels
    fig.supxlabel('time (s)', fontsize=16)
    fig.supylabel('hD (cm)', fontsize=16)

    plt.tight_layout()
    plt.show()

def plot_signal_distribution(
    signals: np.ndarray, # (x_length, num_signals)
    generated: bool = True,
    fname: str = None,
):
    if generated == True:
        distribution_color = 'red'
    else:
        distribution_color = 'blue'
    
    num_signals = signals.shape[0]
    signal_length = signals.shape[1]

    signals_df = pd.DataFrame(signals)

    median_line = signals_df.median(axis=1)

    # Transform x values
    x = [i / 4096 for i in range(0, 256)]
    x = [value - (53 / 4096) for value in x]

    percentile_2_5 = signals_df.quantile(0.025, axis=1)
    percentile_97_5 = signals_df.quantile(0.975, axis=1)
    plt.fill_between(x, percentile_2_5, percentile_97_5, color=distribution_color, alpha=0.25, label='Central 95%')

    percentile_25 = signals_df.quantile(0.25, axis=1)
    percentile_75 = signals_df.quantile(0.75, axis=1)
    plt.fill_between(x, percentile_25, percentile_75, color=distribution_color, alpha=0.5, label='Central 50%')

    plt.plot(x, median_line.values, color='k', linestyle='-', linewidth=1, alpha=1.0, label='Median of signals')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)  
    plt.ylim(-600, 300)
    plt.xlabel('time (s)', size=20)
    plt.ylabel('hD (cm)', size=20)
    plt.grid(True)
    plt.legend()

    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    plt.show()