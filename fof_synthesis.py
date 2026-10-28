"""
fof_synthesis.py — FOF synthesis with vibrato, jitter, and dual preset support.

Supports both Mandarin (3-formant) and Bass singer (5-formant) vowel presets.
Includes CHANT-style vibrato and jitter for natural vocal quality.

Usage:
    python fof_synthesis.py
    python fof_synthesis.py --preset bass --f0 110
    python fof_synthesis.py --vibrato-depth 15 --vibrato-rate 5.5
    python fof_synthesis.py --no-vibrato
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from scipy.signal import spectrogram, windows

from fof_burst import fof_burst
from vowel_data import MANDARIN_VOWELS, BASS_VOWELS


# ---------------------------------------------------------------------------
# Parameter checks
# ---------------------------------------------------------------------------

def check_synthesis_params(f0, duration, fs):
    """Raise ValueError for f0 / duration / fs values that cannot be synthesized."""
    if not fs > 0:
        raise ValueError(f"fs must be positive, got {fs}")
    if not 0 < f0 < fs / 2:
        raise ValueError(f"f0 must be in (0, fs/2 = {fs / 2:g}) Hz, got {f0}")
    if not duration > 0 or int(fs * duration) < 1:
        raise ValueError(f"duration must be at least one sample (1/fs s), got {duration}")


# ---------------------------------------------------------------------------
# Vibrato & jitter — CHANT-style f0 modulation
# ---------------------------------------------------------------------------

def make_f0_contour(f0_base, duration, fs,
                    vibrato_rate=5.5, vibrato_depth_cents=12.0,
                    jitter_amount=0.005):
    """Generate a per-sample f0 contour with vibrato and jitter.

    Parameters
    ----------
    f0_base : float
        Base fundamental frequency in Hz.
    duration : float
        Duration in seconds.
    fs : int
        Sampling rate.
    vibrato_rate : float
        Vibrato LFO frequency in Hz (typical singing: 4–7 Hz).
    vibrato_depth_cents : float
        Vibrato depth in cents (typical: 10–30 cents).
        0 = no vibrato.
    jitter_amount : float
        Random f0 perturbation as fraction of f0 (0.005 = ±0.5%).
        Models the 1/f micro-perturbations described in the CHANT paper.
        0 = no jitter.

    Returns
    -------
    np.ndarray
        f0 value at each sample (length = fs * duration).
    """
    N = int(fs * duration)
    t = np.arange(N) / fs

    # Vibrato: sinusoidal LFO on f0
    if vibrato_depth_cents > 0:
        # cents → frequency ratio: Δf = f0 * (2^(cents/1200) - 1)
        depth_ratio = 2 ** (vibrato_depth_cents / 1200) - 1
        vibrato = depth_ratio * np.sin(2 * np.pi * vibrato_rate * t)
    else:
        vibrato = np.zeros(N)

    # Jitter: 1/f-like random perturbation (3-layer approach from CHANT)
    if jitter_amount > 0:
        # Three layers of band-limited noise at different time scales
        rng = np.random.default_rng(42)
        jitter = np.zeros(N)
        for period_ms in [2, 10, 50]:  # fast, medium, slow fluctuation
            block = int(period_ms / 1000 * fs)
            n_blocks = N // block + 1
            noise = rng.standard_normal(n_blocks)
            jitter += np.repeat(noise, block)[:N] * (jitter_amount / 3)
    else:
        jitter = np.zeros(N)

    f0_contour = f0_base * (1 + vibrato + jitter)

    # Large vibrato/jitter settings can push f0 to zero or below (the pulse
    # train would then never advance) or above Nyquist.
    lo, hi = f0_contour.min(), f0_contour.max()
    if lo <= 0 or hi >= fs / 2:
        raise ValueError(
            f"vibrato/jitter push f0 to {lo:.1f}-{hi:.1f} Hz, outside (0, fs/2); "
            f"reduce the vibrato depth or jitter amount")
    return f0_contour


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

def pitch_onsets(f0_contour, fs):
    """Fractional sample positions of successive pitch pulses.

    Each period is fs / f0 evaluated at the current pulse, kept as a float
    (no rounding to whole samples), so the realised pitch and vibrato follow
    the f0 contour exactly instead of jumping in whole-sample steps.

    Parameters
    ----------
    f0_contour : np.ndarray
        f0 in Hz at each sample (see make_f0_contour).
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Onset times in samples (floats), all < len(f0_contour).
    """
    N = len(f0_contour)
    onsets = []
    t = 0.0
    while True:
        # Snap onsets within 1e-6 samples of an integer: otherwise float
        # round-off decides whether ceil() starts the burst one sample later,
        # which moves the T60 truncation point by one sample.
        r = round(t)
        if abs(t - r) < 1e-6:
            t = float(r)
        if t >= N:
            break
        onsets.append(t)
        t += fs / f0_contour[int(t)]
    return np.array(onsets)


def add_burst(signal, burst, onset):
    """Overlap-add a burst whose first sample is at ceil(onset)."""
    start = int(np.ceil(onset))
    end = min(start + len(burst), len(signal))
    if start < end:
        signal[start:end] += burst[:end - start]


def synthesize_vowel(vowel, f0, duration, fs, beta=None,
                     vibrato_rate=5.5, vibrato_depth_cents=12.0,
                     jitter_amount=0.005):
    """Synthesize one vowel using parallel FOF overlap-add.

    Parameters
    ----------
    vowel : VowelParams
        Formant parameters (works with any number of formants).
    f0 : float
        Base fundamental frequency in Hz.
    duration : float
        Duration in seconds.
    fs : int
        Sampling rate in Hz.
    beta : float or None
        Attack rate. None = default (~3 ms).
    vibrato_rate : float
        Vibrato rate in Hz (0 to disable).
    vibrato_depth_cents : float
        Vibrato depth in cents (0 to disable).
    jitter_amount : float
        Jitter as fraction of f0 (0 to disable).

    Returns
    -------
    np.ndarray
        Normalized audio signal.
    """
    check_synthesis_params(f0, duration, fs)
    N = int(fs * duration)
    signal = np.zeros(N)
    amps = vowel.amps_linear

    # Generate f0 contour (vibrato + jitter)
    f0_contour = make_f0_contour(
        f0, duration, fs,
        vibrato_rate=vibrato_rate,
        vibrato_depth_cents=vibrato_depth_cents,
        jitter_amount=jitter_amount,
    )

    # Overlap-add at fractional pitch-pulse positions
    for onset in pitch_onsets(f0_contour, fs):
        offset = np.ceil(onset) - onset   # sub-sample delay in [0, 1)
        for fc, bw, amp in zip(vowel.freqs, vowel.bws, amps):
            burst = fof_burst(fc, bw, amp, fs, beta=beta, offset=offset)
            add_burst(signal, burst, onset)

    # Normalize
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak

    return signal


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_vowel(signal, label, fs, duration, output_dir, prefix="fof"):
    """Plot time-domain waveform and spectrogram, save to PNG."""
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))

    t_axis = np.linspace(0, duration, len(signal))
    axs[0].plot(t_axis, signal, linewidth=0.5)
    axs[0].set_title(f"FOF synthesized /{label}/")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")

    f, t_spec, Sxx = spectrogram(
        signal, fs,
        window=windows.hamming(1024),
        nperseg=1024,
        noverlap=768,
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-12)
    axs[1].pcolormesh(t_spec, f / 1000, Sxx_db, shading='gouraud')
    axs[1].set_ylim(0, 5)
    axs[1].set_title(f"Spectrogram of /{label}/")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (kHz)")

    plt.tight_layout()
    fig.savefig(output_dir / f"{prefix}_{label}_plot.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="FOF synthesis with vibrato, jitter, and dual preset support"
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
    parser.add_argument("--preset", choices=["mandarin", "bass"],
                        default="mandarin",
                        help="Vowel preset: mandarin (3-formant) or "
                             "bass (5-formant CHANT) (default: mandarin)")
    # Vibrato controls
    parser.add_argument("--vibrato-rate", type=float, default=5.5,
                        help="Vibrato rate in Hz (default: 5.5)")
    parser.add_argument("--vibrato-depth", type=float, default=12.0,
                        help="Vibrato depth in cents (default: 12)")
    parser.add_argument("--jitter", type=float, default=0.005,
                        help="Jitter amount as fraction of f0 (default: 0.005)")
    parser.add_argument("--no-vibrato", action="store_true",
                        help="Disable vibrato and jitter entirely")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select preset
    if args.preset == "bass":
        vowel_table = BASS_VOWELS
        prefix = "fof_bass"
    else:
        vowel_table = MANDARIN_VOWELS
        prefix = "fof_mandarin"

    vowel_keys = args.vowels or list(vowel_table.keys())

    # Vibrato settings
    vib_rate = 0 if args.no_vibrato else args.vibrato_rate
    vib_depth = 0 if args.no_vibrato else args.vibrato_depth
    jitter = 0 if args.no_vibrato else args.jitter

    for label in vowel_keys:
        if label not in vowel_table:
            print(f"Warning: unknown vowel '{label}' in {args.preset}, skipping.")
            continue

        vowel = vowel_table[label]
        print(f"Synthesizing /{label}/ ({args.preset}, "
              f"{'no vibrato' if args.no_vibrato else f'vib={vib_depth}¢'}) ... ",
              end="", flush=True)

        try:
            signal = synthesize_vowel(
                vowel, args.f0, args.duration, args.fs,
                vibrato_rate=vib_rate,
                vibrato_depth_cents=vib_depth,
                jitter_amount=jitter,
            )
        except ValueError as e:
            print()
            parser.error(str(e))

        # Save audio (16-bit PCM)
        wav_path = output_dir / f"{prefix}_{label}.wav"
        wav_write(str(wav_path), args.fs,
                  (signal * 32767).astype(np.int16))

        # Save plot
        plot_vowel(signal, label, args.fs, args.duration, output_dir,
                   prefix=prefix)

        print(f"done → {wav_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
