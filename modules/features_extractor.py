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

"""Sequence feature extraction for CIP.

Computes mono-, di-, and trinucleotide counts that survived the feature
ablation analysis in the trainer. The order in ``FEATURES_ORDER`` must
match the column order used during training (extended_stacking_trainer.py).

Excluded features
-----------------
- Canonical CGI metrics: cg_observed_expected_ratio, cg_percentage,
  cg_content, ta_percentage, ta_content, at_content, gc_content,
  cgc_content, gcg_content, ccg_content, cgg_content, cga_content,
  cgt_content, tcg_content, acg_content
- Zero / near-zero or negative permutation importance: gc_peak, gc_power,
  lz_complexity, gt_content, tc_content, ga_content, ac_content,
  t_content, a_content, caa_content, aaa_content, act_content, tt_content,
  gaa_content, aat_content, att_content, taa_content, tta_content,
  ttt_content, tag_content, tgg_content, gca_content, ctc_content,
  aga_content, acc_content, gtg_content, atc_content, ctt_content,
  aac_content, aag_content, cac_content, ttc_content, gtt_content
- Dataset design artefact: length
- Collinear with g_content / c_content: gg_content, cc_content

Note: ``genomic_position`` is a categorical column handled separately by
the pipeline (one-hot encoding) and is NOT computed here.
"""

from collections import Counter
from .logger import log

# Feature names in the exact order expected by the model.
FEATURES_ORDER = [
    # Mono-nucleotide
    "c_content",
    "g_content",
    # Di-nucleotide
    "aa_content",
    "ag_content",
    "tg_content",
    "ca_content",
    "ct_content",
    # Tri-nucleotide
    "aca_content",
    "agc_content",
    "agg_content",
    "agt_content",
    "ata_content",
    "atg_content",
    "cag_content",
    "cat_content",
    "cca_content",
    "ccc_content",
    "cct_content",
    "cta_content",
    "ctg_content",
    "gac_content",
    "gag_content",
    "gat_content",
    "gcc_content",
    "gct_content",
    "gga_content",
    "ggc_content",
    "ggg_content",
    "ggt_content",
    "gta_content",
    "gtc_content",
    "tac_content",
    "tat_content",
    "tca_content",
    "tcc_content",
    "tct_content",
    "tga_content",
    "tgc_content",
    "tgt_content",
    "ttg_content",
]

# Map each feature name to the k-mer it counts (stripped of "_content").
_KMER_MAP = {f: f[:-8].upper() for f in FEATURES_ORDER}


def extract_features(seq: str) -> dict | None:
    """Extract sequence-based features from a DNA string.

    Non-ATCG characters are silently removed before processing.

    Args:
        seq: Raw DNA sequence string (case-insensitive).

    Returns:
        A dict mapping each name in ``FEATURES_ORDER`` to its raw count,
        or ``None`` on error.
    """
    try:
        for ch in seq.upper():
            if ch not in "ATCG":
                log.warning("Base %s excluded from inference", ch)
        seq = "".join(ch for ch in seq.upper() if ch in "ATCG")
        n = len(seq)

        mono  = Counter(seq)
        di    = Counter(seq[i:i+2] for i in range(n - 1)) if n >= 2 else Counter()
        tri   = Counter(seq[i:i+3] for i in range(n - 2)) if n >= 3 else Counter()
        pool  = {**mono, **di, **tri}

        return {f: pool.get(_KMER_MAP[f], 0) for f in FEATURES_ORDER}

    except Exception as e:
        print(f"Error extracting features: {e}")
        return None
