# CpG Island Predictor (CIP)

***C***pG ***I***sland ***P***redictor (CIP) is a machine learning tool for predicting CpG islands in vertebrate genomes.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [License](#license)

## Description

CIP uses a stacked ensemble of two base models — a Random Forest Classifier and a Gradient Boosting Classifier — whose predictions are combined by a Logistic Regression meta-model. Each base model was trained on approximately 81,000 sequences from human and mouse genomes, covering both CpG island (CGI) and non-CGI sequences. Non-CGI sequences were extracted after masking known CpG islands. The meta-model reduces false positives and negatives, producing a more robust and less biased predictor.

CIP achieves strong performance on a held-out test set: **94.6% accuracy**, **91.6% precision**, **98.1% recall**, **94.8% F1-score**, and **98.3% ROC-AUC**. A shuffle test confirms the model is not making random predictions (~50% accuracy on shuffled data). While trained on human and mouse genomes, CIP is expected to generalize to other vertebrates with similar CpG island properties, though performance outside mammals has not been benchmarked.

Unlike traditional CpG island predictors, CIP does not rely on GC content or observed/expected CpG ratio. Instead, it uses alternative sequence-based features including GC periodicity (via FFT), Lempel-Ziv complexity, and mono/di-nucleotide counts. See `modules/features_extractor.py` for the full feature list.

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run `CIP.py` from the command line (or by double-clicking it). When prompted, provide the path to a FASTA file containing the sequences to analyze:

```
CpG Island Predictor (CIP) v1.0.0
Copyright: AGPL-3.0-or-later (see LICENSE file)

Enter path to FASTA file or /quit to close: sequences.fasta
```

CIP will print a prediction and probability for each sequence:

```
- Sequence 'seq_1' is not a CpG island.
        Probability of CpG island: 0.0037
- Sequence 'seq_2' is a CpG island.
        Probability of CpG island: 0.9777
- Sequence 'seq_3' is not a CpG island.
        Probability of CpG island: 0.3733
```

Type `/quit` or press `Ctrl+C` to exit.

## Project Structure

```
CpG_Island_Predictor/
├── CIP.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # AGPL-3.0 license
├── README.md                   # This file
├── test.txt                    # Sample FASTA sequences for testing
├── config/
│   └── model.pkl               # Pre-trained stacked ensemble model
└── modules/
    ├── __init__.py
    └── features_extractor.py   # Feature extraction logic
```

## Authors

- **Lorenzo Orsini** — Initial work, data preparation, implementation, modeling, documentation — [lorenzo.orsini3@gmail.com](mailto:lorenzo.orsini3@gmail.com)
- **Alex Carnoli** — Implementation, modeling — [alex.carnoli@gmail.com](mailto:alex.carnoli@gmail.com)

## License

This project is distributed under the GNU Affero General Public License v3 (AGPL-3.0) or later. See the [LICENSE](LICENSE) file for full details.
