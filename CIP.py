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

VERSION = "2.2.2"
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

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

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

# Supported extra output formats (passed as flags on the input line).
_SUPPORTED_FORMATS = {"bed", "gff3"}


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
        if re.search(rf'\b{re.escape(pos)}\b', header_lower):
            return pos
    return None


def _parse_coords_from_header(header: str) -> tuple[str, int, int] | None:
    """
    Try to extract genomic coordinates from a FASTA sequence header.

    Matches common formats produced by bedtools/samtools, e.g.:
        chr1:10000-20000
        chr1:10,000-20,000

    Args:
        header: The full FASTA record description (rec.description).

    Returns:
        A (chrom, start, end) tuple (0-based half-open, BED convention) or
        None if no coordinate pattern is found.
    """
    m = re.search(r'([\w.]+):(\d[\d,]*)-(\d[\d,]*)', header)
    if m:
        chrom = m.group(1)
        start = int(m.group(2).replace(',', ''))
        end   = int(m.group(3).replace(',', ''))
        return chrom, start, end
    return None


def _build_feature_row(base_feats: dict, position: str, position_classes: list[str]) -> dict:
    """
    Combine base sequence features with the one-hot encoded genomic position.

    Only the position columns that were retained after feature ablation are
    included (currently only ``pos_intergenic``).

    Args:
        base_feats: Dict of numerical features from extract_features().
        position: One of the valid genomic position labels.
        position_classes: Ordered list of all position labels.

    Returns:
        Dict with all numerical features + the retained one-hot pos_* columns.
    """
    row = {k: base_feats[k] for k in FEATURES_ORDER}
    row["pos_intergenic"] = 1 if position == "intergenic" else 0
    return row


def _full_feature_columns(position_classes: list[str]) -> list[str]:
    """Return the ordered list of all feature column names passed to the model."""
    return FEATURES_ORDER + ["pos_intergenic"]


# ── Extra output writers ──────────────────────────────────────────────────────

def _write_bed(print_items: list[dict], output_path: str) -> None:
    """
    Write predictions to a BED9 file (one line per sequence).

    Coordinates are taken from the FASTA header when available
    (chr:start-end format); otherwise the sequence id is used as chrom
    with 0-based coordinates spanning the full sequence length.

    Colour coding: green (0,200,0) = CpG island; red (200,0,0) = non-island.
    The BED score field contains int(probability × 1000), clamped to [0, 1000].

    Args:
        print_items: One dict per sequence, as built by predict_from_fasta.
        output_path: Destination file path (created or overwritten).
    """
    lines = [
        'track name="CIP_predictions" '
        'description="CpG Island Predictor output" '
        'itemRgb="On"\n'
    ]
    for item in print_items:
        chrom  = item.get("chrom", item["id"])
        start  = item.get("coord_start", 0)
        end    = item.get("coord_end", item.get("seq_len", 0))
        score  = min(int(item["proba"] * 1000), 1000) if item["proba"] is not None else 0
        strand = "."
        rgb    = "0,200,0" if item["pred"] == 1 else "200,0,0"
        lines.append(
            f"{chrom}\t{start}\t{end}\t{item['id']}\t{score}\t"
            f"{strand}\t{start}\t{end}\t{rgb}\n"
        )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def _write_gff3(print_items: list[dict], output_path: str) -> None:
    """
    Write predictions to a GFF3 file (one feature per sequence).

    Coordinates follow GFF3 convention (1-based, closed intervals).
    The feature type is ``CpG_island`` for positive predictions and
    ``non_CpG_island`` for negative ones.  Probability, prediction flag,
    and inferred position context are stored in the attributes column.

    Args:
        print_items: One dict per sequence, as built by predict_from_fasta.
        output_path: Destination file path (created or overwritten).
    """
    lines = [
        "##gff-version 3\n",
        f"# Generated by CIP v{VERSION} on {datetime.now().isoformat(timespec='seconds')}\n",
    ]
    for item in print_items:
        chrom      = item.get("chrom", item["id"])
        # GFF3 is 1-based closed: add 1 to 0-based BED start
        gff_start  = item.get("coord_start", 0) + 1
        gff_end    = item.get("coord_end", item.get("seq_len", 0))
        score_str  = f"{item['proba']:.4f}" if item["proba"] is not None else "."
        feat_type  = "CpG_island" if item["pred"] == 1 else "non_CpG_island"
        pos_ctx    = item.get("position") or "known"
        inferred   = "true" if item.get("inferred") else "false"
        attrs = (
            f"ID={item['id']};"
            f"prediction={item['pred']};"
            f"probability={score_str};"
            f"position_context={pos_ctx};"
            f"position_inferred={inferred}"
        )
        lines.append(
            f"{chrom}\tCIP\t{feat_type}\t{gff_start}\t{gff_end}\t"
            f"{score_str}\t.\t.\t{attrs}\n"
        )
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


# ── Main prediction function ──────────────────────────────────────────────────

def predict_from_fasta(
    model,
    fasta_path: str,
    position_classes: list[str],
    output_formats: set[str] | None = None,
) -> None:
    """
    Run predictions on all sequences in a FASTA file.

    For each sequence:
    - If the FASTA header contains a recognised genomic position label,
      that position is used directly and a single prediction is made.
    - Otherwise, inference is run 4 times (once per position class) and
      the result with the highest CpG island probability is selected for
      terminal output; all 4 results are written to the CSV.

    If ``output_formats`` contains ``"bed"`` and/or ``"gff3"``, additional
    files are written alongside the CSV (same timestamp, different extension).
    Each extra-format file contains one row per sequence (best result only).

    Args:
        model:            A fitted scikit-learn estimator.
        fasta_path:       Path to the input FASTA file.
        position_classes: Ordered list of valid genomic position labels.
        output_formats:   Set of extra output formats (``"bed"``, ``"gff3"``).
    """
    if output_formats is None:
        output_formats = set()

    try:
        records = list(SeqIO.parse(fasta_path, "fasta"))
    except Exception as e:
        print(_handle_io_error(e, fasta_path))
        return

    if not records:
        print("No sequences found in the provided FASTA file.")
        return

    all_columns  = _full_feature_columns(position_classes)
    output_rows  = []   # all rows for the CSV (may be 4× per sequence)
    print_items  = []   # one item per sequence for terminal output + extra formats

    # ── Wrap the record iterator with a progress bar when available ───────────
    record_iter = (
        tqdm(records, desc="Predicting", unit="seq", ncols=80)
        if _TQDM_AVAILABLE
        else records
    )

    for rec in record_iter:
        seq = str(rec.seq).upper()

        if not set(seq).issubset({"A", "C", "G", "T", "N"}):
            print(
                f"Warning: sequence '{rec.id}' contains invalid characters. "
                "Please use only A, C, T, G or N."
            )
            continue

        # Strip N's for feature extraction (consistent with features_extractor)
        seq_clean  = "".join(ch for ch in seq if ch in ("A", "T", "C", "G"))
        seq_len    = len(seq_clean)
        base_feats = extract_features(seq)
        known_pos  = _parse_position_from_header(rec.description, position_classes)
        coords     = _parse_coords_from_header(rec.description)

        # Genomic coordinate fields for BED/GFF3
        if coords is not None:
            chrom, coord_start, coord_end = coords
        else:
            chrom, coord_start, coord_end = rec.id, 0, seq_len

        if known_pos is not None:
            # ── Single prediction: position is known from the header ──────────
            row   = _build_feature_row(base_feats, known_pos, position_classes)
            X     = pd.DataFrame([row], columns=all_columns)
            pred  = int(model.predict(X)[0])
            proba = float(model.predict_proba(X)[0, 1]) if hasattr(model, "predict_proba") else None

            output_rows.append({
                "id":          rec.id,
                "position":    known_pos,
                "prediction":  pred,
                "probability": f"{proba:.4f}" if proba is not None else "N/A",
            })
            print_items.append({
                "id":          rec.id,
                "pred":        pred,
                "proba":       proba,
                "position":    None,
                "inferred":    False,
                "seq_len":     seq_len,
                "chrom":       chrom,
                "coord_start": coord_start,
                "coord_end":   coord_end,
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

            best = max(results, key=lambda r: r["proba"] if r["proba"] is not None else r["pred"])
            print_items.append({
                "id":          rec.id,
                "pred":        best["pred"],
                "proba":       best["proba"],
                "position":    best["position"],
                "inferred":    True,
                "seq_len":     seq_len,
                "chrom":       chrom,
                "coord_start": coord_start,
                "coord_end":   coord_end,
            })

    # ── Write outputs ─────────────────────────────────────────────────────────
    os.makedirs("outs", exist_ok=True)
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path   = f"outs/{timestamp}"

    csv_path = f"{base_path}.csv"
    pd.DataFrame(output_rows).to_csv(csv_path, index=False, sep=",")
    written_files = [csv_path]

    if "bed" in output_formats:
        bed_path = f"{base_path}.bed"
        _write_bed(print_items, bed_path)
        written_files.append(bed_path)

    if "gff3" in output_formats:
        gff3_path = f"{base_path}.gff3"
        _write_gff3(print_items, gff3_path)
        written_files.append(gff3_path)

    # ── Terminal output ───────────────────────────────────────────────────────
    for item in print_items:
        prob_str  = f"{item['proba']:.4f}" if item['proba'] is not None else "N/A"
        prob_note = f"Probability of CpG island: {prob_str}"

        if item["pred"] == 1:
            label = f"{Fore.LIGHTGREEN_EX}Sequence '{item['id']}' is a CpG island.{Fore.RESET}"
        else:
            label = f"{Fore.LIGHTRED_EX}Sequence '{item['id']}' is not a CpG island.{Fore.RESET}"

        pos_note = ""
        if item["inferred"]:
            pos_note = f"Best position: {item['position']}"

        print(f"- {label}\n\t{prob_note}\n\t{pos_note}")

    print(f"\n    Results saved to:")
    for path in written_files:
        print(f"      {path}")


def _exit(code: int = 0) -> None:
    """Clean up colorama and exit the program."""
    deinit()
    sys.exit(code)


if __name__ == "__main__":
    init()
    print(Fore.LIGHTBLACK_EX + "Copyright: AGPL-3.0-or-later (see LICENSE file)")
    print("See https://github.com/lorenzoorsini3/CpG-Island-Predictor for source code" + Fore.RESET)

    if not _TQDM_AVAILABLE:
        print("    Note: install 'tqdm' for progress bars (pip install tqdm).")

    # ── Load metadata ─────────────────────────────────────────────────────────
    try:
        metadata = _load_metadata()
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
    except Exception as e:
        print(_handle_io_error(e, _METADATA_FILE))
        print("Warning: metadata not found, using built-in defaults.")
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
        expected = len(FEATURES_ORDER) + len(_ONE_HOT_COLS)
        if expected != n_features:
            print(
                f"Warning: metadata expects {n_features} features but "
                f"this version of CIP would build {expected}. "
                "Results may be unreliable."
            )

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            raw = input(
                "\nEnter FASTA path [--bed] [--gff3] or /quit: "
            ).strip()
        except (KeyboardInterrupt, EOFError):
            print()
            _exit()

        if not raw:
            continue

        parts          = raw.split()
        fasta_path     = parts[0]
        flags          = {p.lstrip("-").lower() for p in parts[1:]}
        output_formats = flags & _SUPPORTED_FORMATS

        unknown_flags = flags - _SUPPORTED_FORMATS
        if unknown_flags:
            print(f"    Warning: unrecognised flags ignored: {', '.join('--' + f for f in unknown_flags)}")

        if fasta_path == "/quit":
            _exit()
        else:
            predict_from_fasta(model, fasta_path, position_classes, output_formats)