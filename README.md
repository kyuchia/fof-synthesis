# FOF Synthesis of Mandarin Vowels

Time-domain Formant Wave Function (FOF) synthesis of five Mandarin Chinese vowels: /a/, /i/, /u/, /o/, /ei/.

Based on Rodet's 1984 formulation, using formant measurements from Chen & Wang (2011) for male Mandarin speakers.

## Quick start

```bash
pip install numpy scipy matplotlib
python fof_synthesis.py
```

Output `.wav` files and spectrogram plots are saved to `output/`.

## Project structure

```
fof_burst.py              # Core FOF burst generator (Rodet 1984 formula)
vowel_data.py             # Mandarin vowel formant parameters (F1–F3)
fof_synthesis.py          # Main synthesis: overlap-add → .wav + spectrograms
fof_visualize.py          # Single-burst time/frequency visualization
fof_parameter_study.py    # α (bandwidth) and β (skirt width) comparison plots
output/                   # Generated audio and plots
docs/                     # Report PDF
matlab_original/          # Original MATLAB version (archived)
```

## Usage examples

```bash
# Synthesize all vowels at 220 Hz, 1 second
python fof_synthesis.py

# Custom pitch and duration
python fof_synthesis.py --f0 180 --duration 2.0

# Synthesize only specific vowels
python fof_synthesis.py --vowels a i u

# Visualize a single burst
python fof_visualize.py --fc 753 --bw 100 --save output/burst_753.png

# Parameter study plots
python fof_parameter_study.py --save-dir output
```

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

Multiple formants are summed in parallel, and bursts repeat at the fundamental period T = fs/f0 via overlap-add.

## Key improvements over original MATLAB version

- **β decoupled from fc** — β is now an independent parameter (default: ~3 ms attack), matching Rodet's design intent
- **Dynamic burst length** — computed from T60 criterion instead of fixed 100 ms
- **dB-scale amplitudes** — formant amplitudes in dB relative to F1, matching CHANT conventions
- **Single source of truth** — `fof_burst()` is called everywhere, no code duplication
- **Vectorized** — numpy array operations replace sample-by-sample Python loops

## References

1. X. Rodet, "Time-Domain Formant-Wave-Function Synthesis," *Computer Music Journal*, vol. 8, no. 3, pp. 9–14, 1984.
2. X. Rodet, Y. Potard, J.-B. Barrière, "The CHANT Project: From the Synthesis of the Singing Voice to Synthesis in General," *Computer Music Journal*, vol. 8, no. 3, pp. 15–31, 1984.
3. H. C. Chen and M. J. Wang, "An Acoustic Analysis of Chinese and English Vowels," 2011.
4. M. Clarke and X. Rodet, "Real-time FOF and FOG Synthesis in MSP and its Integration with PSOLA," *ICMC*, 2003.

## License

MIT
