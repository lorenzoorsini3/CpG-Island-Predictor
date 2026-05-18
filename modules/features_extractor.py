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

# Ordered list of feature names used for model input.
# The order must match the one used during training (extended_stacking_trainer.py).
#
# Excluded features:
#   - Canonical CGI metrics : cg_observed_expected_ratio, cg_percentage, cg_content,
#                             ta_percentage, ta_content, at_content, gc_content,
#                             cgc_content, gcg_content, ccg_content, cgg_content,
#                             cga_content, cgt_content, tcg_content, acg_content
#   - Zero / near-zero or negative permutation importance:
#                             gc_peak, gc_power, lz_complexity,
#                             gt_content, tc_content, ga_content, ac_content,
#                             t_content, a_content,
#                             caa_content, aaa_content, act_content, tt_content,
#                             gaa_content, aat_content, att_content, taa_content,
#                             tta_content, ttt_content, tag_content, tgg_content,
#                             gca_content, ctc_content, aga_content, acc_content,
#                             gtg_content, atc_content, ctt_content, aac_content,
#                             aag_content, cac_content, ttc_content, gtt_content
#   - Dataset design artefact : length
#   - Collinear with g_content / c_content : gg_content, cc_content
#
# NOTE: genomic_position is a categorical column handled separately by the
#       pipeline (one-hot encoding); it is NOT computed here.
FEATURES_ORDER = [
    # ── Mono-nucleotide ───────────────────────────────────────────────────────
    "c_content",
    "g_content",
    # ── Di-nucleotide ─────────────────────────────────────────────────────────
    "aa_content",
    "ag_content",
    "tg_content",
    "ca_content",
    "ct_content",
    # ── Tri-nucleotide ────────────────────────────────────────────────────────
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


def extract_features(seq: str) -> dict:
    """Extract sequence-based features from a DNA string.

    Non-ATCG characters are silently removed before processing.
    Features include mono-nucleotide, di-nucleotide, and tri-nucleotide
    counts that survived the feature ablation analysis in the trainer.

    Args:
        seq: Raw DNA sequence string (case-insensitive).

    Returns:
        A dict mapping each feature name in FEATURES_ORDER to its raw count.
        On error, all features default to 0.
    """
    try:
        seq = "".join(ch for ch in seq.upper() if ch in ("A", "T", "C", "G"))
        length = len(seq)

        mono_counts = Counter(seq)
        di_counts = (
            Counter(seq[i : i + 2] for i in range(length - 1))
            if length >= 2 else Counter()
        )
        tri_counts = (
            Counter(seq[i : i + 3] for i in range(length - 2))
            if length >= 3 else Counter()
        )

        features = {
            # ── Mono-nucleotide ───────────────────────────────────────────────
            "c_content":   mono_counts.get("C", 0),
            "g_content":   mono_counts.get("G", 0),
            # ── Di-nucleotide ─────────────────────────────────────────────────
            "aa_content":  di_counts.get("AA", 0),
            "ca_content":  di_counts.get("CA", 0),
            "tg_content":  di_counts.get("TG", 0),
            "ag_content":  di_counts.get("AG", 0),
            "ct_content":  di_counts.get("CT", 0),
            # ── Tri-nucleotide ────────────────────────────────────────────────
            "aca_content": tri_counts.get("ACA", 0),
            "agc_content": tri_counts.get("AGC", 0),
            "agg_content": tri_counts.get("AGG", 0),
            "agt_content": tri_counts.get("AGT", 0),
            "ata_content": tri_counts.get("ATA", 0),
            "atg_content": tri_counts.get("ATG", 0),
            "cag_content": tri_counts.get("CAG", 0),
            "cat_content": tri_counts.get("CAT", 0),
            "cca_content": tri_counts.get("CCA", 0),
            "ccc_content": tri_counts.get("CCC", 0),
            "cct_content": tri_counts.get("CCT", 0),
            "cta_content": tri_counts.get("CTA", 0),
            "ctg_content": tri_counts.get("CTG", 0),
            "gac_content": tri_counts.get("GAC", 0),
            "gag_content": tri_counts.get("GAG", 0),
            "gat_content": tri_counts.get("GAT", 0),
            "gcc_content": tri_counts.get("GCC", 0),
            "gct_content": tri_counts.get("GCT", 0),
            "gga_content": tri_counts.get("GGA", 0),
            "ggc_content": tri_counts.get("GGC", 0),
            "ggg_content": tri_counts.get("GGG", 0),
            "ggt_content": tri_counts.get("GGT", 0),
            "gta_content": tri_counts.get("GTA", 0),
            "gtc_content": tri_counts.get("GTC", 0),
            "tac_content": tri_counts.get("TAC", 0),
            "tat_content": tri_counts.get("TAT", 0),
            "tca_content": tri_counts.get("TCA", 0),
            "tcc_content": tri_counts.get("TCC", 0),
            "tct_content": tri_counts.get("TCT", 0),
            "tga_content": tri_counts.get("TGA", 0),
            "tgc_content": tri_counts.get("TGC", 0),
            "tgt_content": tri_counts.get("TGT", 0),
            "ttg_content": tri_counts.get("TTG", 0),
        }

        return {k: features.get(k, 0) for k in FEATURES_ORDER}

    except Exception as e:
        print(f"Error extracting features: {e}")
        return {k: 0 for k in FEATURES_ORDER}
