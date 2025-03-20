from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import pandas as pd

import torch

import src.starcattovae.nn.vae as VAE

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman']
})

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

def plot_individual_loss(
    total_losses: List[float],
    reconstruction_losses: List[float],
    kld_losses: List[float],
    fname: str = None,
): 
    fig = plt.figure(figsize=(10, 6))
    axes = fig.gca()

    axes.plot(total_losses, label="Total Training Loss", color='orange')
    axes.plot(reconstruction_losses, label="Total Validation Loss", color='yellow')
    axes.plot(kld_losses, label="Total Validation Loss", color='red')
    axes.set_xlabel("Epoch", size=20)
    axes.set_ylabel("Loss", size=20)
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
    model: VAE, 
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    steps=10
):
    model.eval()

    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        interpolated_latents = [mean_1 * (1 - alpha) + mean_2 * alpha for alpha in np.linspace(0, 1, steps)]
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() for latent in interpolated_latents]

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

def animate_latent_morphs(
    model,  # Assuming model is a VAE instance
    signal_1: torch.Tensor,
    signal_2: torch.Tensor,
    max_value: float, 
    train_dataset,
    steps=10,
    interval=200,
    save_path=None
):
    model.eval()

    with torch.no_grad():
        mean_1, _ = model.encoder(signal_1)
        mean_2, _ = model.encoder(signal_2)

        # Forward and backward interpolation
        forward_interpolated = [mean_1 * (1 - alpha) + mean_2 * alpha for alpha in np.linspace(0, 1, steps)]
        backward_interpolated = [mean_2 * (1 - alpha) + mean_1 * alpha for alpha in np.linspace(0, 1, steps)]
        interpolated_latents = forward_interpolated + backward_interpolated
        morphed_signals = [model.decoder(latent).cpu().detach().numpy() for latent in interpolated_latents]

        # Compute the posterior distribution
        all_means = []
        for x, y in train_dataset:
            x = torch.tensor(x).to(model.DEVICE)
            mean, _ = model.encoder(x)
            all_means.append(mean.cpu().numpy())
        all_means = np.concatenate(all_means, axis=0)

    fig = plt.figure(figsize=(10, 17))  # Adjust the figure size for vertical stacking

    # Create 3D plot for latent space
    ax_latent = fig.add_subplot(211, projection='3d')  # First plot (top) in vertical layout
    ax_latent.scatter(all_means[:, 0], all_means[:, 1], all_means[:, 2], color='gray', alpha=0.2, label='Posterior Distribution')
    ax_latent.scatter(mean_1[0].cpu().numpy(), mean_1[1].cpu().numpy(), mean_1[2].cpu().numpy(), color='blue', s=50, label='Signal 1')
    ax_latent.scatter(mean_2[0].cpu().numpy(), mean_2[1].cpu().numpy(), mean_2[2].cpu().numpy(), color='green', s=50, label='Signal 2')
    ax_latent.plot([mean_1[0].cpu().numpy(), mean_2[0].cpu().numpy()],
                   [mean_1[1].cpu().numpy(), mean_2[1].cpu().numpy()],
                   [mean_1[2].cpu().numpy(), mean_2[2].cpu().numpy()], color='red', linestyle='--', label='Interpolation Path', linewidth=2)
    moving_point, = ax_latent.plot([], [], [], 'ro', markersize=7, label='Interpolated Point')
    # ax_latent.set_title('Latent Space Interpolation')
    ax_latent.set_xlabel('Latent Dim 1')
    ax_latent.set_ylabel('Latent Dim 2')
    ax_latent.set_zlabel('Latent Dim 3')
    # ax_latent.legend()

    # Create plot for signal morphing
    ax_signal = fig.add_subplot(212)  # Second plot (bottom) in vertical layout

    # X-axis values (shared across all plots)
    x_vals = [i / 4096 for i in range(0, 256)]
    x_vals = [value - (53 / 4096) for value in x_vals]

    # Initialize the plot
    line, = ax_signal.plot([], [], color="red")
    ax_signal.set_xlim(min(x_vals), max(x_vals))
    ax_signal.set_ylim(-600, 300)
    ax_signal.axvline(x=0, color="black", linestyle="--", alpha=0.5)
    ax_signal.grid(True)
    ax_signal.set_xlabel('time (s)', fontsize=16)
    ax_signal.set_ylabel('hD (cm)', fontsize=16)

    def init():
        line.set_data([], [])
        moving_point.set_data([], [])
        moving_point.set_3d_properties([])
        return line, moving_point

    def update(frame):
        y_interp = morphed_signals[frame].flatten() * max_value
        line.set_data(x_vals, y_interp)
        # ax_signal.set_title(f"Interpolated Signal {frame + 1}")

        # Update the moving point in the latent space
        latent_point = interpolated_latents[frame].cpu().numpy()
        moving_point.set_data(latent_point[0], latent_point[1])
        moving_point.set_3d_properties(latent_point[2])
        return line, moving_point

    ani = animation.FuncAnimation(fig, update, frames=len(interpolated_latents), init_func=init, blit=True, interval=interval, repeat=True)

    if save_path:
        ani.save(save_path, writer='imagemagick', fps=30)

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

import numpy as np
import matplotlib.pyplot as plt

def plot_single_signal(
    signal: np.ndarray,
    max_value: float,
    fname: str = None,  # Added missing comma
):
    # Generate x-axis values
    x = [i / 4096 for i in range(0, 256)]
    x = [value - (53 / 4096) for value in x]

    # Process signal for plotting
    y = signal.flatten()
    y = y * max_value

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, color='blue')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.ylim(-600, 300)
    plt.xlabel('time (s)', size=20)
    plt.ylabel('hD (cm)', size=20)
    plt.grid(True)

    # Save or show the plot
    if fname:
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()
