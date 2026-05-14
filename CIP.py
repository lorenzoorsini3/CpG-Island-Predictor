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

VERSION = "2.1.0"
print(f"CpG Island Predictor (CIP) v{VERSION}")

import json
import sys
import os
import re
from datetime import datetime

import joblib
import pandas as pd
from Bio import SeqIO
from colorama import Fore, deinit, init

from modules.features_extractor import FEATURES_ORDER, extract_features

_MODEL_FILE    = "./config/model.pkl"
_METADATA_FILE = "./config/metadata.json"

# One-hot column names and valid position labels, used as fallback if
# metadata.json is not available.
# After ablation (v2.1.0) only pos_intergenic is retained as a feature;
# the other three positions are still used for multi-context inference
# but their one-hot columns are not passed to the model.
_ONE_HOT_COLS      = ['pos_intergenic']
_POSITION_CLASSES  = ['upstream', 'gene_body', 'downstream', 'intergenic']

_FILE_ERRORS = {
    FileNotFoundError: lambda e, path: f"FileNotFoundError - File '{path}' not found.",
    IsADirectoryError: lambda e, path: f"IsADirectoryError - '{path}' is a directory, not a file.",
    PermissionError:   lambda e, path: f"PermissionError - '{path}': Permission denied.",
}


def _handle_io_error(e: Exception, path: str) -> str:
    handler = _FILE_ERRORS.get(type(e))
    return handler(e, path) if handler else f"ERROR - Error reading file '{path}': {e}"


def _load_metadata() -> dict | None:
    """Load metadata.json from config, return None if not found."""
    try:
        with open(_METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Warning: could not read metadata.json: {e}")
        return None


def _parse_position_from_header(header: str, valid_positions: list[str]) -> str | None:
    """
    Try to extract a genomic position label from a FASTA sequence header.

    Looks for any of the valid position labels as a standalone word
    (case-insensitive) anywhere in the header string.

    Args:
        header: The full FASTA record description (rec.description).
        valid_positions: List of accepted position labels.

    Returns:
        The matched position label (lowercase) or None if not found.
    """
    header_lower = header.lower()
    for pos in valid_positions:
        # Match as a whole word to avoid partial matches
        if re.search(rf'\b{re.escape(pos)}\b', header_lower):
            return pos
    return None


def _build_feature_row(base_feats: dict, position: str, position_classes: list[str]) -> dict:
    """
    Combine base sequence features with the one-hot encoded genomic position.

    Only the position columns that were retained after feature ablation are
    included (currently only ``pos_intergenic``).  The full list of position
    classes is still used for multi-context inference — each position produces
    a distinct row — but only the ``pos_intergenic`` flag is passed to the
    model as a numeric feature.

    Args:
        base_feats: Dict of numerical features from extract_features().
        position: One of the valid genomic position labels.
        position_classes: Ordered list of all position labels.

    Returns:
        Dict with all numerical features + the retained one-hot pos_* columns.
    """
    row = {k: base_feats[k] for k in FEATURES_ORDER}
    # Only pos_intergenic survived ablation; the other three were dropped.
    row["pos_intergenic"] = 1 if position == "intergenic" else 0
    return row


def _full_feature_columns(position_classes: list[str]) -> list[str]:
    """Return the ordered list of all feature column names passed to the model."""
    return FEATURES_ORDER + ["pos_intergenic"]


def predict_from_fasta(model, fasta_path: str, position_classes: list[str]) -> None:
    """
    Run predictions on all sequences in a FASTA file.

    For each sequence:
    - If the FASTA header contains a recognised genomic position label,
      that position is used directly and a single prediction is made.
    - Otherwise, inference is run 4 times (once per position class) and
      the result with the highest CpG island probability is selected for
      terminal output; all 4 results are written to the CSV.

    Args:
        model: A fitted scikit-learn estimator.
        fasta_path: Path to the input FASTA file.
        position_classes: Ordered list of valid genomic position labels.
    """
    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except Exception as e:
        print(_handle_io_error(e, fasta_path))
        return

    if not records:
        print("No sequences found in the provided FASTA file.")
        return

    all_columns  = _full_feature_columns(position_classes)
    output_rows  = []   # all rows for the CSV (may be 4x per sequence)
    print_items  = []   # one item per sequence for terminal output

    for rec in records:
        seq = str(rec.seq).upper()

        if not set(seq).issubset({"A", "C", "G", "T", "N"}):
            print(
                f"Warning: sequence '{rec.id}' contains invalid characters. "
                "Please use only A, C, T, G or N."
            )
            continue

        base_feats = extract_features(seq)
        known_pos  = _parse_position_from_header(rec.description, position_classes)

        if known_pos is not None:
            # ── Single prediction: position is known from the header ──────────
            row   = _build_feature_row(base_feats, known_pos, position_classes)
            X     = pd.DataFrame([row], columns=all_columns)
            pred  = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None

            output_rows.append({
                "id":         rec.id,
                "position":   known_pos,
                "prediction": pred,
                "probability": f"{proba:.4f}" if proba is not None else "N/A",
            })
            print_items.append({
                "id":          rec.id,
                "pred":        pred,
                "proba":       proba,
                "position":    None,   # None = position was known, no need to annotate
                "inferred":    False,
            })

        else:
            # ── Multi-prediction: run inference for all 4 positions ───────────
            results = []
            for pos in position_classes:
                row   = _build_feature_row(base_feats, pos, position_classes)
                X     = pd.DataFrame([row], columns=all_columns)
                pred  = int(model.predict(X)[0])
                proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None

                output_rows.append({
                    "id":          rec.id,
                    "position":    pos,
                    "prediction":  pred,
                    "probability": f"{proba:.4f}" if proba is not None else "N/A",
                })
                results.append({"pred": pred, "proba": proba, "position": pos})

            # Pick the result with the highest CpG island probability for terminal
            best = max(results, key=lambda r: r["proba"] if r["proba"] is not None else r["pred"])
            print_items.append({
                "id":       rec.id,
                "pred":     best["pred"],
                "proba":    best["proba"],
                "position": best["position"],
                "inferred": True,
            })

    # ── Write CSV ─────────────────────────────────────────────────────────────
    os.makedirs("outs", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outs/{timestamp}.csv"
    pd.DataFrame(output_rows).to_csv(output_path, index=False, sep=",")

    # ── Terminal output ───────────────────────────────────────────────────────
    for item in print_items:
        prob_str = f"{item['proba']:.4f}" if item['proba'] is not None else "N/A"

        prob_note = f"Probability of CpG island: {prob_str}"

        if item["pred"] == 1:
            label = f"{Fore.LIGHTGREEN_EX}Sequence '{item['id']}' is a CpG island.{Fore.RESET}"
        else:
            label = f"{Fore.LIGHTRED_EX}Sequence '{item['id']}' is not a CpG island.{Fore.RESET}"

        pos_note = ""
        if item["inferred"]:
            pos_note = f"Best position: {item['position']}"

        print(f"- {label}\n\t{prob_note}\n\t{pos_note}")

    print(f"\n    Results saved to: {output_path}")


def _exit(code: int = 0) -> None:
    """Clean up colorama and exit the program."""
    deinit()
    sys.exit(code)


if __name__ == "__main__":
    init()
    print(Fore.LIGHTBLACK_EX + "Copyright: AGPL-3.0-or-later (see LICENSE file)")
    print("See https://github.com/lorenzoorsini3/CpG-Island-Predictor for source code" + Fore.RESET)

    # ── Load metadata ─────────────────────────────────────────────────────────
    metadata = _load_metadata()
    if metadata:
        position_classes = metadata.get("position_classes", _POSITION_CLASSES)
        arch_version     = metadata.get("architecture_version", "unknown")
        n_features       = metadata.get("n_features", None)
        train_species    = metadata.get("training_species", [])
        test_species     = metadata.get("test_species", [])
        print(f"    Model architecture : {arch_version}")
        if train_species:
            print(f"    Trained on         : {', '.join(train_species)}")
        if test_species:
            print(f"    Evaluated on       : {', '.join(test_species)}")
    else:
        print("    Warning: metadata.json not found, using built-in defaults.")
        position_classes = _POSITION_CLASSES
        n_features       = None

    # ── Load model ────────────────────────────────────────────────────────────
    try:
        model = joblib.load(_MODEL_FILE)
    except Exception as e:
        print(_handle_io_error(e, _MODEL_FILE))
        _exit(1)

    # Sanity check: verify expected feature count matches metadata
    if n_features is not None:
        expected = len(FEATURES_ORDER) + len(position_classes)
        if expected != n_features:
            print(
                f"Warning: metadata expects {n_features} features but "
                f"this version of CIP would build {expected}. "
                "Results may be unreliable."
            )

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            fasta_path = input("\nEnter path to FASTA file or /quit to close: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            _exit()

        if fasta_path == "/quit":
            _exit()
        else:
            predict_from_fasta(model, fasta_path, position_classes)
