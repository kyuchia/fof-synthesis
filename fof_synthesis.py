"""
fof_synthesis.py — FOF synthesis of five Mandarin vowels.

Generates .wav files and spectrogram plots for /a/, /i/, /u/, /o/, /ei/.
Uses overlap-add of parallel FOF bursts at the fundamental frequency.

Usage:
    python fof_synthesis.py
    python fof_synthesis.py --f0 180 --duration 1.5
    python fof_synthesis.py --output-dir ./my_output
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from scipy.signal import spectrogram, windows

from fof_burst import fof_burst
from vowel_data import MANDARIN_VOWELS


def synthesize_vowel(vowel, f0, duration, fs, beta=None):
    """Synthesize one vowel using parallel FOF overlap-add.

    Parameters
    ----------
    vowel : VowelParams
        Formant parameters for this vowel.
    f0 : float
        Fundamental frequency in Hz.
    duration : float
        Duration in seconds.
    fs : int
        Sampling rate in Hz.
    beta : float or None
        Attack rate. None = default (~3 ms).

    Returns
    -------
    np.ndarray
        Normalized audio signal.
    """
    T = round(fs / f0)          # samples per pitch period
    N = int(fs * duration)      # total samples
    signal = np.zeros(N)

    amps = vowel.amps_linear    # dB → linear conversion

    for onset in range(0, N, T):
        for fc, bw, amp in zip(vowel.freqs, vowel.bws, amps):
            burst = fof_burst(fc, bw, amp, fs, beta=beta)
            burst_len = len(burst)

            # Handle end-of-signal boundary
            end = min(onset + burst_len, N)
            signal[onset:end] += burst[:end - onset]

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return signal


def plot_vowel(signal, label, fs, duration, output_dir):
    """Plot time-domain waveform and spectrogram, save to PNG."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    # Time domain
    t_axis = np.linspace(0, duration, len(signal))
    axs[0].plot(t_axis, signal, linewidth=0.5)
    axs[0].set_title(f"FOF synthesized Mandarin /{label}/")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    # Spectrogram
    f, t_spec, Sxx = spectrogram(
        signal, fs,
        window=windows.hamming(1024),
        nperseg=1024,
        noverlap=768,
    )
    # Avoid log(0) warnings
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    axs[1].pcolormesh(t_spec, f / 1000, Sxx_db, shading='gouraud')
    axs[1].set_ylim(0, 5)
    axs[1].set_title(f"Spectrogram of /{label}/")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (kHz)")

    plt.tight_layout()
    fig.savefig(output_dir / f"fof_{label}_plot.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="FOF synthesis of Mandarin vowels"
    )
    parser.add_argument("--f0", type=float, default=220,
                        help="Fundamental frequency in Hz (default: 220)")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="Duration in seconds (default: 1.0)")
    parser.add_argument("--fs", type=int, default=44100,
                        help="Sampling rate in Hz (default: 44100)")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Output directory (default: output)")
    parser.add_argument("--vowels", nargs="+", default=None,
                        help="Vowels to synthesize (default: all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vowel_keys = args.vowels or list(MANDARIN_VOWELS.keys())

    for label in vowel_keys:
        if label not in MANDARIN_VOWELS:
            print(f"Warning: unknown vowel '{label}', skipping.")
            continue

        vowel = MANDARIN_VOWELS[label]
        print(f"Synthesizing /{label}/ ... ", end="", flush=True)

        signal = synthesize_vowel(vowel, args.f0, args.duration, args.fs)

        # Save audio (16-bit PCM)
        wav_path = output_dir / f"fof_mandarin_{label}.wav"
        wav_write(str(wav_path), args.fs,
                  (signal * 32767).astype(np.int16))

        # Save plot
        plot_vowel(signal, label, args.fs, args.duration, output_dir)

        print(f"done → {wav_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
