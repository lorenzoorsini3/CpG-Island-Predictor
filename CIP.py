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

VERSION = "1.0.0"
print(f"CpG Island Predictor (CIP) v{VERSION}")

import sys

import joblib
import pandas as pd
from Bio import SeqIO
from colorama import Fore, deinit, init

from modules.features_extractor import FEATURES_ORDER, extract_features


def predict_from_fasta(model, fasta_path: str) -> None:
    """Run predictions on all sequences in a FASTA file.

    For each sequence, extracts features, runs the model, and prints
    the prediction along with the CpG island probability (if available).

    Args:
        model: A fitted scikit-learn estimator with a predict() method.
        fasta_path: Path to the input FASTA file.
    """
    records = list(SeqIO.parse(fasta_path, "fasta"))
    if not records:
        print("No sequences found in the provided FASTA file.")
        return

    seq_ids = [rec.id for rec in records]
    rows = [[extract_features(str(rec.seq))[k] for k in FEATURES_ORDER] for rec in records]

    X = pd.DataFrame(rows, columns=FEATURES_ORDER)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None

    for i, seq_id in enumerate(seq_ids):
        prob_str = f"{y_proba[i]:.4f}" if y_proba is not None else "N/A"
        if y_pred[i] == 1:
            label = f"{Fore.LIGHTGREEN_EX}Sequence '{seq_id}' is a CpG island.{Fore.RESET}"
        else:
            label = f"{Fore.LIGHTRED_EX}Sequence '{seq_id}' is not a CpG island.{Fore.RESET}"
        print(f"- {label}\n\tProbability of CpG island: {prob_str}")


def _exit(code: int = 0) -> None:
    """Clean up colorama and exit the program."""
    deinit()
    sys.exit(code)


if __name__ == "__main__":
    init()
    print(Fore.LIGHTBLACK_EX + "Copyright: AGPL-3.0-or-later (see LICENSE file)")
    print("See https://github.com/lorenzoorsini3/CpG-Island-Predictor for source code" + Fore.RESET)

    model = joblib.load("./config/model.pkl")

    while True:
        try:
            fasta_path = input("\nEnter path to FASTA file or /quit to close: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            _exit()

        if fasta_path == "/quit":
            _exit()
        else:
            predict_from_fasta(model, fasta_path)
