"""
fof_burst.py — Core FOF (Formant Wave Function) burst generator.

Implements Rodet's 1984 FOF formula with corrected β decoupling.

Reference:
    X. Rodet, "Time-Domain Formant-Wave-Function Synthesis,"
    Computer Music Journal, vol. 8, no. 3, pp. 9–14, 1984.
"""

import numpy as np


def fof_burst(fc, bw, amp, fs, beta=None, phi=0.0):
    """Generate a single FOF burst for one formant.

    Parameters
    ----------
    fc : float
        Formant center frequency in Hz.
    bw : float
        Formant bandwidth (-6 dB) in Hz. Controls α = π·BW/fs.
    amp : float
        Amplitude (linear scale).
    fs : int
        Sampling rate in Hz.
    beta : float or None
        Attack rate (rad/sample). Controls skirt width independently
        of bandwidth — this is the key decoupled parameter in FOF.
        If None, defaults to π / (0.003 * fs), giving ~3 ms attack.
    phi : float
        Initial phase of the sinusoidal carrier (radians).

    Returns
    -------
    np.ndarray
        The FOF burst waveform (1-D array).

    Notes
    -----
    The FOF waveform is defined in two stages:

    Attack (0 ≤ k ≤ π/β):
        s(k) = ½(1 − cos(βk)) · e^(−αk) · sin(ωc·k + φ)

    Decay (k > π/β):
        s(k) = e^(−αk) · sin(ωc·k + φ)

    The burst is truncated at the T60 decay time: t60 = 6.91 / (π·BW).
    """
    # Default β: ~3 ms attack (independent of fc)
    if beta is None:
        beta = np.pi / (0.003 * fs)

    # Derived parameters
    alpha = np.pi * bw / fs          # exponential decay rate
    omega_c = 2 * np.pi * fc / fs    # angular center frequency

    # Dynamic burst length based on T60 decay criterion
    t60 = 6.91 / (np.pi * bw)        # T60 in seconds
    fof_len = max(int(np.ceil(t60 * fs)), 64)  # minimum 64 samples

    # Attack duration in samples
    k_a = int(np.round(np.pi / beta))

    # Vectorized computation (no Python for-loop)
    k = np.arange(fof_len)
    env = np.exp(-alpha * k)
    carrier = np.sin(omega_c * k + phi)

    # Attack envelope: smooth onset via half-cosine window
    attack = np.where(k <= k_a, 0.5 * (1 - np.cos(beta * k)), 1.0)

    burst = amp * attack * env * carrier
    return burst


def fof_burst_legacy(fc, bw, amp, fs):
    """Legacy version matching the original MUMT-501 MATLAB code.

    Kept for regression testing — β = ωc/10, fixed 100 ms duration.
    """
    fof_len = round(0.1 * fs)
    alpha = np.pi * bw / fs
    omega_c = 2 * np.pi * fc / fs
    beta = omega_c / 10
    k_a = round(np.pi / beta)

    k = np.arange(fof_len)
    env = np.exp(-alpha * k)
    carrier = np.sin(omega_c * k)
    attack = np.where(k <= k_a, 0.5 * (1 - np.cos(beta * k)), 1.0)

    return amp * attack * env * carrier
