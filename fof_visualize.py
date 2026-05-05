"""
fof_visualize.py — Visualize a single FOF burst in time and frequency domain.

Usage:
    python fof_visualize.py
    python fof_visualize.py --fc 500 --bw 80
    python fof_visualize.py --save burst_plot.png
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from fof_burst import fof_burst


def plot_burst(fc, bw, amp, fs, beta=None, save_path=None):
    """Plot one FOF burst: time domain + magnitude spectrum."""
    burst = fof_burst(fc, bw, amp, fs, beta=beta)
    t_ms = np.arange(len(burst)) / fs * 1000

    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # --- Time domain ---
    axs[0].plot(t_ms, burst, color='steelblue', linewidth=0.8)
    axs[0].set_title(f"FOF burst (time domain) — Fc = {fc} Hz, BW = {bw} Hz")
    axs[0].set_xlabel("Time (ms)")
    axs[0].set_ylabel("Amplitude")
    axs[0].set_xlim(0, min(40, t_ms[-1]))

    # --- Frequency domain ---
    # Use zero-padding instead of windowing to preserve spectral shape
    NFFT = 2 ** (int(np.ceil(np.log2(len(burst)))) + 2)
    S = np.abs(np.fft.fft(burst, NFFT))[:NFFT // 2]
    f_axis = fs * np.arange(NFFT // 2) / NFFT
    S_db = 20 * np.log10(S + 1e-12)

    axs[1].plot(f_axis, S_db, color='black', linewidth=0.8)
    axs[1].set_title("FOF burst (frequency domain)")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Magnitude (dB)")
    axs[1].set_xlim(0, 5000)

    # Mark formant center
    axs[1].axvline(fc, color='red', linestyle='--', alpha=0.7)
    peak_db = np.max(S_db)
    axs[1].text(fc + 200, peak_db - 20,
                f"Fc = {fc} Hz", color='red', fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a single FOF burst"
    )
    parser.add_argument("--fc", type=float, default=753,
                        help="Center frequency in Hz (default: 753)")
    parser.add_argument("--bw", type=float, default=100,
                        help="Bandwidth in Hz (default: 100)")
    parser.add_argument("--amp", type=float, default=1.0,
                        help="Amplitude (default: 1.0)")
    parser.add_argument("--fs", type=int, default=44100,
                        help="Sampling rate (default: 44100)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save plot to file instead of displaying")
    args = parser.parse_args()

    plot_burst(args.fc, args.bw, args.amp, args.fs, save_path=args.save)


if __name__ == "__main__":
    main()
