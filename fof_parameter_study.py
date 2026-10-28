"""
fof_parameter_study.py — Visualize the effect of α (bandwidth) and β (skirt width).

Generates two plots:
  1. Spectral comparison across different bandwidths (α sweep)
  2. Time-domain comparison across different attack rates (β sweep)

Usage:
    python fof_parameter_study.py
    python fof_parameter_study.py --save-dir output
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fof_burst import fof_burst


def plot_bandwidth_comparison(fc, fs, save_dir=None):
    """Compare FOF spectra with different bandwidths (α sweep)."""
    bw_list = [60, 100, 180]
    colors = ['#3266ad', '#1D9E75', '#D85A30']

    fig, ax = plt.subplots(figsize=(10, 5))

    for bw, color in zip(bw_list, colors):
        burst = fof_burst(fc, bw, 1.0, fs)

        NFFT = 2 ** (int(np.ceil(np.log2(len(burst)))) + 2)
        S = np.abs(np.fft.fft(burst, NFFT))[:NFFT // 2]
        f_axis = fs * np.arange(NFFT // 2) / NFFT
        S_db = 20 * np.log10(S + 1e-12)

        ax.plot(f_axis, S_db, label=f"BW = {bw} Hz", color=color)

    ax.set_title("Spectral comparison of FOF bursts with different bandwidths")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_xlim(0, 5000)
    ax.set_ylim(-80, None)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "fof_bandwidth_comparison.png", dpi=300)
        print("Saved bandwidth comparison plot.")
    else:
        plt.show()
    plt.close(fig)


def plot_beta_comparison(fc, fs, save_dir=None):
    """Compare FOF time-domain waveforms with different β values."""
    # β as explicit attack durations in seconds (not tied to ωc)
    attack_ms_list = [1.0, 3.0, 8.0]
    colors = ['#534AB7', '#BA7517', '#1D9E75']

    fig, ax = plt.subplots(figsize=(10, 5))

    for atk_ms, color in zip(attack_ms_list, colors):
        beta = np.pi / (atk_ms / 1000 * fs)
        burst = fof_burst(fc, 100, 1.0, fs, beta=beta)
        t_ms = np.arange(len(burst)) / fs * 1000

        ax.plot(t_ms, burst,
                label=f"attack = {atk_ms} ms (β = {beta:.4f})",
                color=color, linewidth=0.8)

    ax.set_title("Time-domain comparison of FOF bursts with different attack rates (β)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, 40)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "fof_beta_comparison.png", dpi=300)
        print("Saved β comparison plot.")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="FOF parameter study: bandwidth and attack rate"
    )
    parser.add_argument("--fc", type=float, default=753,
                        help="Center frequency for comparison (default: 753)")
    parser.add_argument("--fs", type=int, default=44100,
                        help="Sampling rate (default: 44100)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Save plots to directory (default: show)")
    args = parser.parse_args()

    save = args.save_dir
    if save:
        Path(save).mkdir(parents=True, exist_ok=True)

    try:
        plot_bandwidth_comparison(args.fc, args.fs, save)
        plot_beta_comparison(args.fc, args.fs, save)
    except ValueError as e:
        parser.error(str(e))


if __name__ == "__main__":
    main()
