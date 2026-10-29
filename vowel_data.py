"""
vowel_data.py — Mandarin vowel formant parameters.

Formant frequencies (F1-F3) from:
    H. C. Chen and M. J. Wang, "An acoustic analysis of Chinese and
    English vowels," CALR Linguistic Journal, vol. 1, pp. 1-19, 2011.

Bandwidth and amplitude values estimated from typical male speaker
measurements and CHANT-style conventions.

Amplitudes are stored in dB (relative to F1 = 0 dB) and converted
to linear scale via the helper function. They set the time-domain
amplitude of each formant's FOF burst, not its spectral peak level:
the peak also depends on the formant's bandwidth, so a wider formant
peaks lower than its table value (e.g. /a/ F3, -12 dB in the table,
peaks about -17 dB relative to F1).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class VowelParams:
    """Parameters for one vowel."""
    label: str
    freqs: list       # F1, F2, F3 in Hz
    bws: list         # bandwidths in Hz
    amps_db: list     # amplitudes in dB (relative to F1)

    @property
    def amps_linear(self):
        """Convert dB amplitudes to linear scale."""
        return [10 ** (a / 20) for a in self.amps_db]


# --- Mandarin vowel formant table ---
# Amplitudes in dB relative to F1 (0 dB)
MANDARIN_VOWELS = {
    'a':  VowelParams('a',  [753, 1298, 3068], [100, 120, 150], [0, -7, -12]),
    'i':  VowelParams('i',  [309, 2416, 3438], [90,  110, 130], [0, -9, -16]),
    'u':  VowelParams('u',  [349,  759, 2883], [80,  100, 120], [0, -9, -16]),
    'o':  VowelParams('o',  [505,  856, 3052], [90,  110, 130], [0, -7, -12]),
    'ei': VowelParams('ei', [460, 2453, 3168], [100, 120, 150], [0, -7, -12]),
}

# --- Bass singer formant table (CHANT / Csound FLOSS Manual) ---
# For future FOF Lab use — 5 formants, dB scale
BASS_VOWELS = {
    'a': VowelParams('a', [600, 1040, 2250, 2450, 2750],
                          [60,   70,  110,  120,  130],
                          [0,    -7,   -9,   -9,  -20]),
    'e': VowelParams('e', [400, 1620, 2400, 2800, 3100],
                          [40,   80,  100,  120,  120],
                          [0,   -12,   -9,  -12,  -18]),
    'i': VowelParams('i', [250, 1750, 2600, 3050, 3340],
                          [60,   90,  100,  120,  120],
                          [0,   -30,  -16,  -22,  -28]),
    'o': VowelParams('o', [400,  750, 2400, 2600, 2900],
                          [40,   80,  100,  120,  120],
                          [0,   -11,  -21,  -20,  -40]),
    'u': VowelParams('u', [350,  600, 2400, 2675, 2950],
                          [40,   80,  100,  120,  120],
                          [0,   -20,  -32,  -28,  -36]),
}
