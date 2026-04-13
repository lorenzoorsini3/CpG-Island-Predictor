# Copyright (C) 2026 Lorenzo Orsini
# Author(s):
#   - Lorenzo Orsini, lorenzo.orsini3@gmail.com
#   - Alex Carnoli, alex.carnoli@gmail.com
# This file is part of CpG Island Predictor (CIP).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from collections import Counter

import numpy as np
from scipy.fft import fft, fftfreq

# Ordered list of feature names used for model input.
# The order must match the one used during training.
FEATURES_ORDER = [
    "length",
    "gc_power",
    "gc_peak",
    "lz_complexity",
    "c_content",
    "g_content",
    "a_content",
    "t_content",
    "aa_content",
    "ag_content",
    "ac_content",
    "tg_content",
    "tc_content",
    "tt_content",
    "ca_content",
    "ct_content",
    "cc_content",
    "ga_content",
    "gt_content",
    "gg_content",
]


def _gc_periodicity_features(seq: str, min_period: int = 9, max_period: int = 12) -> tuple[float, float]:
    """Compute GC periodicity power and peak frequency in the 9-12 bp range.

    Uses FFT on a binary GC signal (1 for G/C, 0 for A/T) to identify
    periodic patterns characteristic of CpG islands.

    Args:
        seq: Uppercase DNA sequence (only A/T/G/C characters).
        min_period: Lower bound of the period range in base pairs.
        max_period: Upper bound of the period range in base pairs.

    Returns:
        A tuple (power, peak_freq) where power is the mean spectral power
        in the target frequency band and peak_freq is the dominant frequency.
        Returns (0.0, 0.0) if the sequence is too short.
    """
    signal = np.array([1 if base in ("G", "C") else 0 for base in seq])
    length = len(signal)

    if length < max_period * 2:
        return 0.0, 0.0

    # Compute power spectrum of the mean-centered signal
    spectrum = np.abs(fft(signal - np.mean(signal))) ** 2
    freqs = fftfreq(length, d=1)  # frequency in cycles/bp

    # Keep only positive frequencies
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    spectrum = spectrum[pos_mask]

    # Isolate the target frequency band
    band_mask = (freqs >= 1 / max_period) & (freqs <= 1 / min_period)
    if not np.any(band_mask):
        return 0.0, 0.0

    band_spectrum = spectrum[band_mask]
    band_freqs = freqs[band_mask]

    power = float(np.mean(band_spectrum))
    peak_freq = float(band_freqs[np.argmax(band_spectrum)])

    return power, peak_freq


def _lz_complexity(seq: str) -> int:
    """Compute the Lempel-Ziv complexity of a sequence.

    Counts the number of distinct substrings added when scanning the
    sequence left to right, as defined by the LZ76 algorithm.

    Args:
        seq: Input string (DNA sequence).

    Returns:
        Integer complexity count.
    """
    i = 0
    n = len(seq)
    seen = set()
    count = 0

    while i < n:
        length = 1
        while i + length <= n and seq[i : i + length] in seen:
            length += 1
        seen.add(seq[i : i + length])
        i += length
        count += 1

    return count


def extract_features(seq: str) -> dict:
    """Extract sequence-based features from a DNA string.

    Non-ATCG characters are silently removed before processing.
    Features include sequence length, GC periodicity, Lempel-Ziv complexity,
    mono-nucleotide counts, and selected di-nucleotide counts.

    Args:
        seq: Raw DNA sequence string (case-insensitive).

    Returns:
        A dict mapping each feature name in FEATURES_ORDER to its value.
        On error, all features default to 0.
    """
    try:
        seq = "".join(ch for ch in seq.upper() if ch in ("A", "T", "C", "G"))
        length = len(seq)

        mono_counts = Counter(seq)
        di_counts = Counter(seq[i : i + 2] for i in range(length - 1)) if length >= 2 else Counter()

        gc_power, gc_peak = _gc_periodicity_features(seq)
        lz = _lz_complexity(seq)

        features = {
            "length": length,
            "gc_power": gc_power,
            "gc_peak": gc_peak,
            "lz_complexity": lz,
            # Mono-nucleotide counts
            "c_content": mono_counts.get("C", 0),
            "g_content": mono_counts.get("G", 0),
            "a_content": mono_counts.get("A", 0),
            "t_content": mono_counts.get("T", 0),
            # Di-nucleotide counts
            "aa_content": di_counts.get("AA", 0),
            "ag_content": di_counts.get("AG", 0),
            "ac_content": di_counts.get("AC", 0),
            "tg_content": di_counts.get("TG", 0),
            "tc_content": di_counts.get("TC", 0),
            "tt_content": di_counts.get("TT", 0),
            "ca_content": di_counts.get("CA", 0),
            "ct_content": di_counts.get("CT", 0),
            "cc_content": di_counts.get("CC", 0),
            "ga_content": di_counts.get("GA", 0),
            "gt_content": di_counts.get("GT", 0),
            "gg_content": di_counts.get("GG", 0),
        }

        return {k: features.get(k, 0) for k in FEATURES_ORDER}

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {k: 0 for k in FEATURES_ORDER}
