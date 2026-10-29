# FOF Synthesis

Time-domain Formant Wave Function (FOF) synthesis engine in Python, based on Rodet's 1984 formulation. Features vibrato/jitter, vowel morphing, and diphthong synthesis, with Mandarin (Chen & Wang 2011) and Bass singer (CHANT/Csound) formant presets.

> **C++ port:** a C++17 command-line port of the synthesis core is available at [fof-synth-cpp](https://github.com/kyuchia/fof-synth-cpp). With vibrato and jitter off, its output is sample-identical to this implementation.

![Vowel morph /a/ → /i/](output/fof_morph_a_to_i_spectrogram.png)

## Quick start

```bash
pip install numpy scipy matplotlib
python fof_synthesis.py
```

Output `.wav` files and spectrogram plots are saved to `output/`.

## Project structure

```
fof_burst.py              # Core FOF burst generator (Rodet 1984 formula)
vowel_data.py             # Vowel formant parameters: Mandarin (3-formant) + Bass singer (5-formant)
fof_synthesis.py           # Main synthesis with vibrato/jitter + dual preset support
fof_morphing.py            # Vowel morphing and diphthong synthesis
fof_visualize.py           # Single-burst time/frequency visualization
fof_parameter_study.py     # α (bandwidth) and β (skirt width) comparison plots
output/                    # Generated audio and plots
matlab_original/           # Original MATLAB version (fixed)
```

## Usage examples

```bash
# Synthesize Mandarin vowels (default)
python fof_synthesis.py

# Bass singer preset (5 formants, CHANT data) at 110 Hz
python fof_synthesis.py --preset bass --f0 110

# Disable vibrato for comparison
python fof_synthesis.py --vowels a --no-vibrato

# Custom vibrato settings
python fof_synthesis.py --vibrato-rate 6.0 --vibrato-depth 20

# Vowel morphing: /a/ → /i/ with S-curve transition
python fof_morphing.py --from a --to i --duration 2.0

# Diphthong /ei/ with correct hold-glide timing (Mandarin preset only)
python fof_morphing.py --diphthong ei

# Morph through all Bass singer vowels
python fof_morphing.py --preset bass

# Visualize a single burst
python fof_visualize.py --fc 753 --bw 100 --save output/burst_753.png

# Parameter study plots
python fof_parameter_study.py --save-dir output
```

## Features

**Dual vowel presets** — Mandarin Chinese (3-formant) and Bass singer (5-formant, CHANT/Csound). Selectable via `--preset mandarin` or `--preset bass`. The Mandarin formant frequencies are measurements from Chen & Wang (2011); the bandwidths and amplitudes are estimates.

**Exact pitch** — pitch pulses are placed at fractional sample positions instead of rounding the period to whole samples, and each burst is evaluated analytically from its exact onset. The measured pitch error is 0.00 cents from 110 to 880 Hz, and vibrato follows a smooth sine instead of whole-sample steps.

**Vibrato and jitter** — CHANT-style f0 modulation: sinusoidal vibrato (default 5.5 Hz, 12 cents) plus smooth 1/f-like jitter for natural vocal quality. The jitter sums three linearly interpolated Gaussian noise layers at 10, 40 and 160 ms time scales; `--jitter` sets its standard deviation as a fraction of f0 (default 0.005 = 0.5%). Controllable via `--vibrato-rate`, `--vibrato-depth`, `--jitter`, or `--no-vibrato`.

**Vowel morphing** — Time-varying formant interpolation between any two vowels, with linear or S-curve transition. Supports asymmetric hold-glide timing for diphthong synthesis (e.g. `/ei/`: 60% hold on /e/, 40% glide to /i/).

## FOF formula

Each formant is synthesized as a damped sinusoidal burst with a smoothed attack:

**Attack phase** (0 ≤ k ≤ π/β):

$$s(k) = \frac{1}{2}(1 - \cos\beta k) \cdot e^{-\alpha k} \cdot \sin(\omega_c k + \phi)$$

**Decay phase** (k > π/β):

$$s(k) = e^{-\alpha k} \cdot \sin(\omega_c k + \phi)$$

Where:

- **ωc** = 2πfc/fs — formant center frequency
- **α** = πBW/fs — exponential decay rate (controls -6 dB bandwidth)
- **β** — attack rate (controls skirt width, **independent** of bandwidth)
- **φ** — initial phase

Multiple formants are summed in parallel, and bursts repeat at the fundamental period T = fs/f0 via overlap-add. T is kept fractional: each burst starts at its exact (non-integer) onset, so k = n + offset with offset in [0, 1).

Formant amplitudes are given in dB relative to F1 and scale each burst in the time domain. They are not spectral peak levels: a formant's spectral peak also depends on its bandwidth, so a wider formant peaks lower than its table value (e.g. /a/ F3, −12 dB in the table, peaks about −17 dB relative to F1).

## Key improvements over original MATLAB version

- **β decoupled from fc** — β is now an independent parameter (default: ~3 ms attack), matching Rodet's design intent
- **Dynamic burst length** — computed from T60 criterion instead of fixed 100 ms
- **dB-scale amplitudes** — formant amplitudes in dB relative to F1, CHANT-style (time-domain burst amplitudes; see above)
- **Fractional pitch period** — exact f0 and smooth vibrato instead of periods rounded to whole samples
- **Input validation** — invalid f0, duration, sampling rate, bandwidth or vibrato/jitter settings give a one-line CLI error instead of a traceback or a hang
- **Single source of truth** — `fof_burst()` is called everywhere, no code duplication
- **Vectorized** — numpy array operations replace sample-by-sample Python loops

## References

1. X. Rodet, "Time-Domain Formant-Wave-Function Synthesis," _Computer Music Journal_, vol. 8, no. 3, pp. 9–14, 1984.
2. X. Rodet, Y. Potard, J.-B. Barrière, "The CHANT Project: From the Synthesis of the Singing Voice to Synthesis in General," _Computer Music Journal_, vol. 8, no. 3, pp. 15–31, 1984.
3. H. C. Chen and M. J. Wang, "An acoustic analysis of Chinese and English vowels," _CALR Linguistic Journal_, vol. 1, pp. 1–19, Jan. 2011.
4. M. Clarke and X. Rodet, "Real-time FOF and FOG Synthesis in MSP and its Integration with PSOLA," _ICMC_, 2003.
