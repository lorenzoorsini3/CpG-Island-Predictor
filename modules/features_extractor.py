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

# numpy and scipy.fft were used by _gc_periodicity_features and _lz_complexity,
# which were removed in v2.1.0 after feature ablation showed zero importance.

# Ordered list of feature names used for model input.
# The order must match the one used during training.
# Features removed after ablation analysis (v2.1.0):
#   - length        : zero permutation importance (dataset design artefact)
#   - gc_power      : near-zero importance, potentially ill-constructed
#   - gc_peak       : near-zero importance, potentially ill-constructed
#   - lz_complexity : near-zero importance, no relevant biological justification
#   - ac_content    : near-zero importance
#   - tc_content    : near-zero importance
#   - ga_content    : near-zero importance
#   - gt_content    : near-zero importance
#   - cc_content    : collinear with c_content (VIF > 10)
#   - gg_content    : collinear with g_content (VIF > 10)
FEATURES_ORDER = [
    "c_content",
    "g_content",
    "a_content",
    "t_content",
    "aa_content",
    "ag_content",
    "tg_content",
    "tt_content",
    "ca_content",
    "ct_content",
]




def extract_features(seq: str) -> dict:
    """Extract sequence-based features from a DNA string.

    Non-ATCG characters are silently removed before processing.
    Features include mono-nucleotide counts and selected di-nucleotide
    counts that survived the feature ablation analysis.

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
        di_counts   = (
            Counter(seq[i : i + 2] for i in range(length - 1))
            if length >= 2 else Counter()
        )

        features = {
            # Mono-nucleotide counts
            "c_content":  mono_counts.get("C", 0),
            "g_content":  mono_counts.get("G", 0),
            "a_content":  mono_counts.get("A", 0),
            "t_content":  mono_counts.get("T", 0),
            # Di-nucleotide counts
            "aa_content": di_counts.get("AA", 0),
            "ag_content": di_counts.get("AG", 0),
            "tg_content": di_counts.get("TG", 0),
            "tt_content": di_counts.get("TT", 0),
            "ca_content": di_counts.get("CA", 0),
            "ct_content": di_counts.get("CT", 0),
        }

        return {k: features.get(k, 0) for k in FEATURES_ORDER}

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {k: 0 for k in FEATURES_ORDER}
