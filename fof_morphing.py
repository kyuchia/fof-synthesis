"""
fof_morphing.py — Vowel morphing and diphthong synthesis.

Interpolates between two sets of formant parameters over time,
producing continuous vowel transitions. Also demonstrates proper
diphthong /ei/ synthesis with time-varying formants.

Usage:
    python fof_morphing.py
    python fof_morphing.py --from a --to i --duration 2.0
    python fof_morphing.py --diphthong ei --duration 1.0
    python fof_morphing.py --preset bass --from a --to u
"""

import argparse
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.io.wavfile import write as wav_write
from scipy.signal import spectrogram, windows

from fof_burst import fof_burst
from fof_synthesis import (make_f0_contour, pitch_onsets, add_burst,
                           check_synthesis_params)
from vowel_data import MANDARIN_VOWELS, BASS_VOWELS, VowelParams


def interpolate_vowels(vowel_a, vowel_b, morph):
    """Linearly interpolate between two vowel parameter sets.

    Parameters
    ----------
    vowel_a, vowel_b : VowelParams
        Source and target vowels. Must have the same number of formants.
    morph : float
        Interpolation factor (0.0 = vowel_a, 1.0 = vowel_b).

    Returns
    -------
    VowelParams
        Interpolated vowel parameters.

    Raises
    ------
    ValueError
        If the two vowels have a different number of formants (for example
        a 3-formant Mandarin vowel and a 5-formant bass vowel).
    """
    if len(vowel_a.freqs) != len(vowel_b.freqs):
        raise ValueError(
            f"cannot interpolate /{vowel_a.label}/ ({len(vowel_a.freqs)} formants) "
            f"and /{vowel_b.label}/ ({len(vowel_b.freqs)} formants): "
            f"the formant counts must match")
    n = len(vowel_a.freqs)
    freqs = [vowel_a.freqs[i] * (1 - morph) + vowel_b.freqs[i] * morph
             for i in range(n)]
    bws = [vowel_a.bws[i] * (1 - morph) + vowel_b.bws[i] * morph
           for i in range(n)]
    amps_db = [vowel_a.amps_db[i] * (1 - morph) + vowel_b.amps_db[i] * morph
               for i in range(n)]

    return VowelParams(
        label=f"{vowel_a.label}→{vowel_b.label}",
        freqs=freqs,
        bws=bws,
        amps_db=amps_db,
    )


def synthesize_morph(vowel_a, vowel_b, f0, duration, fs,
                     morph_curve="linear",
                     glide_start=0.0,
                     vibrato_rate=5.5, vibrato_depth_cents=12.0,
                     jitter_amount=0.005):
    """Synthesize a vowel morph from vowel_a to vowel_b.

    Parameters
    ----------
    vowel_a, vowel_b : VowelParams
        Source and target vowels.
    f0 : float
        Fundamental frequency.
    duration : float
        Total duration in seconds.
    fs : int
        Sampling rate.
    morph_curve : str
        "linear" or "scurve" (smooth S-curve transition).
    glide_start : float
        Fraction of duration (0–1) to hold vowel_a steady before
        starting the glide. E.g. 0.6 means the first 60% is pure
        vowel_a, then glide happens in the remaining 40%.
        Used for diphthongs where the first element is held longer.

    Returns
    -------
    np.ndarray
        Audio signal.
    """
    check_synthesis_params(f0, duration, fs)
    N = int(fs * duration)
    signal = np.zeros(N)

    f0_contour = make_f0_contour(
        f0, duration, fs,
        vibrato_rate=vibrato_rate,
        vibrato_depth_cents=vibrato_depth_cents,
        jitter_amount=jitter_amount,
    )

    for onset in pitch_onsets(f0_contour, fs):
        # Morph factor: 0 during hold, then ramp during glide
        t_frac = onset / N
        if t_frac <= glide_start:
            raw_morph = 0.0
        else:
            glide_len = 1.0 - glide_start
            raw_morph = (t_frac - glide_start) / glide_len if glide_len > 0 else 1.0

        if morph_curve == "scurve":
            morph = 0.5 * (1 - np.cos(np.pi * raw_morph))
        else:
            morph = raw_morph

        # Interpolate vowel parameters
        vowel_now = interpolate_vowels(vowel_a, vowel_b, morph)
        amps = vowel_now.amps_linear

        offset = np.ceil(onset) - onset   # sub-sample delay in [0, 1)
        for fc, bw, amp in zip(vowel_now.freqs, vowel_now.bws, amps):
            burst = fof_burst(fc, bw, amp, fs, offset=offset)
            add_burst(signal, burst, onset)

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal /= peak
    return signal


# --- Diphthong presets ---
# Each entry: (start_VowelParams, end_key, glide_ratio)
# glide_ratio = fraction of duration spent on starting vowel before
# transitioning. Real diphthongs are asymmetric: the first element
# is held longer, then glides quickly to the second.

# Start vowels are taken from MANDARIN_VOWELS so the data lives in one place.
# /ei/ starts from Chen & Wang's measured /ei/ formants, NOT /a/; it is
# relabelled /e/ because it is the first element of the glide.
# The end vowels are looked up in MANDARIN_VOWELS too, so diphthongs are
# defined for the Mandarin preset only.
DIPHTHONGS_MANDARIN = {
    'ei': (dataclasses.replace(MANDARIN_VOWELS['ei'], label='e'), 'i', 0.6),  # mid front -> high front
    'ai': (MANDARIN_VOWELS['a'], 'i', 0.5),                                  # open -> high front
    'ou': (MANDARIN_VOWELS['o'], 'u', 0.6),                                  # mid back -> high back
}


def main():
    parser = argparse.ArgumentParser(
        description="FOF vowel morphing and diphthong synthesis"
    )
    parser.add_argument("--from", dest="vowel_from", type=str, default=None,
                        help="Source vowel (e.g. 'a')")
    parser.add_argument("--to", dest="vowel_to", type=str, default=None,
                        help="Target vowel (e.g. 'i')")
    parser.add_argument("--diphthong", type=str, default=None,
                        help="Synthesize a diphthong (e.g. 'ei')")
    parser.add_argument("--preset", choices=["mandarin", "bass"],
                        default="mandarin",
                        help="Vowel preset (default: mandarin)")
    parser.add_argument("--f0", type=float, default=220)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--curve", choices=["linear", "scurve"],
                        default="scurve",
                        help="Morph curve shape (default: scurve)")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--no-vibrato", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = BASS_VOWELS if args.preset == "bass" else MANDARIN_VOWELS
    vib_r = 0 if args.no_vibrato else 5.5
    vib_d = 0 if args.no_vibrato else 12.0
    jit = 0 if args.no_vibrato else 0.005

    # Determine what to synthesize
    # Each item: (vowel_a, vowel_b, name, glide_start)
    jobs = []

    if args.diphthong:
        if args.preset != "mandarin":
            parser.error("--diphthong is only defined for --preset mandarin "
                         "(the diphthong start vowels are Mandarin measurements)")
        if args.diphthong in DIPHTHONGS_MANDARIN:
            start_vowel, end_key, glide_ratio = DIPHTHONGS_MANDARIN[args.diphthong]
            jobs.append((start_vowel, MANDARIN_VOWELS[end_key],
                         f"diphthong_{args.diphthong}", glide_ratio))
        else:
            parser.error(f"unknown diphthong '{args.diphthong}'; "
                         f"available: {', '.join(DIPHTHONGS_MANDARIN)}")
    elif args.vowel_from and args.vowel_to:
        for key in (args.vowel_from, args.vowel_to):
            if key not in table:
                parser.error(f"unknown vowel '{key}' for --preset {args.preset}; "
                             f"available: {', '.join(table)}")
        jobs.append((table[args.vowel_from], table[args.vowel_to],
                     f"morph_{args.vowel_from}_to_{args.vowel_to}", 0.0))
    else:
        # Default demo: morph through all adjacent pairs
        keys = list(table.keys())
        for i in range(len(keys) - 1):
            jobs.append((table[keys[i]], table[keys[i + 1]],
                         f"morph_{keys[i]}_to_{keys[i+1]}", 0.0))

    for vowel_a, vowel_b, name, glide_start in jobs:
        print(f"Morphing /{vowel_a.label}/ → /{vowel_b.label}/ "
              f"({args.curve}"
              f"{f', hold {glide_start:.0%}' if glide_start > 0 else ''}"
              f") ... ",
              end="", flush=True)

        try:
            signal = synthesize_morph(
                vowel_a, vowel_b,
                args.f0, args.duration, args.fs,
                morph_curve=args.curve,
                glide_start=glide_start,
                vibrato_rate=vib_r,
                vibrato_depth_cents=vib_d,
                jitter_amount=jit,
            )
        except ValueError as e:
            print()
            parser.error(str(e))

        # Save audio
        wav_path = output_dir / f"fof_{name}.wav"
        wav_write(str(wav_path), args.fs,
                  (signal * 32767).astype(np.int16))

        # Plot spectrogram (morphing is best visualized as spectrogram)
        fig, ax = plt.subplots(figsize=(10, 4))
        f, t_spec, Sxx = spectrogram(
            signal, args.fs,
            window=windows.hamming(1024),
            nperseg=1024, noverlap=768,
        )
        Sxx_db = 10 * np.log10(Sxx + 1e-12)
        ax.pcolormesh(t_spec, f / 1000, Sxx_db, shading='gouraud')
        ax.set_ylim(0, 5)
        ax.set_title(f"/{vowel_a.label}/ → /{vowel_b.label}/ ({args.curve})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (kHz)")
        plt.tight_layout()
        fig.savefig(output_dir / f"fof_{name}_spectrogram.png", dpi=300)
        plt.close(fig)

        print(f"done → {wav_path}")

    print(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
