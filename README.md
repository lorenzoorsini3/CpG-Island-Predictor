# CpG Island Predictor (CIP)

***C***pG ***I***sland ***P***redictor (CIP) is a machine learning tool for predicting CpG islands in vertebrate genomes.

## Table of Contents

- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Genomic Position](#genomic-position)
- [Project Structure](#project-structure)
- [Authors](#authors)
- [License](#license)

## Description

CIP uses a stacked ensemble of two base models — a Random Forest Classifier and a Gradient Boosting Classifier — whose predictions are combined by a Logistic Regression meta-model. Each base model was trained on approximately 93,000 sequences from human (hg38) and mouse (mm39) genomes, covering both CpG island (CGI) and non-CGI sequences. Non-CGI sequences were extracted after masking known CpG islands. The meta-model reduces false positives and negatives, producing a more robust and less biased predictor.

In addition to sequence-based features, the model incorporates the genomic position of each sequence (upstream, gene body, downstream, or intergenic) as a one-hot encoded input, improving discrimination across different regulatory contexts.

### Performance

Performance was evaluated on a held-out test set derived exclusively from dog (canFam6) — a species never seen during training — providing a cross-species generalization benchmark.

| Metric    | Validation (dog) | Test (dog) | Cross-val (human+mouse) |
|-----------|-----------------|------------|------------------------|
| Accuracy  | 97.45%          | 97.47%     | 98.14%                 |
| Precision | 96.09%          | 96.26%     | 97.70%                 |
| Recall    | 98.70%          | 98.78%     | 98.60%                 |
| F1        | 97.38%          | 97.50%     | 98.15%                 |
| ROC-AUC   | 99.53%          | 99.55%     | 99.72%                 |

A shuffle test confirms the model is not making random predictions (~50% accuracy on permuted labels).

Unlike traditional CpG island predictors, CIP does not rely on GC content or observed/expected CpG ratio. Instead, it uses alternative sequence-based features such as mono/di-nucleotide counts. See `modules/features_extractor.py` and `config/metadata.json` for the full feature list.

## Installation

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run `CIP.py` from the command line (or by double-clicking it). When prompted, provide the path to a FASTA file containing the sequences to analyze:

```
CpG Island Predictor (CIP) v2.2.3
Copyright: AGPL-3.0-or-later (see LICENSE file)
See https://github.com/lorenzoorsini3/CpG-Island-Predictor for source code
    Model architecture : v3.1.0
    Trained on         : human (hg38), mouse (mm39)
    Evaluated on       : dog (canFam6)

Enter path to FASTA file or /quit to close: sequences.fasta
```

CIP will print a prediction and probability for each sequence:

```
- Sequence 'seq_1' is not a CpG island.
        Probability of CpG island: 0.0037
        Best position: gene_body
- Sequence 'seq_2' is a CpG island.
        Probability of CpG island: 0.9777
        Best position: gene_body
- Sequence 'seq_3' is not a CpG island.
        Probability of CpG island: 0.3733
        Best position: intergenic
```

When the genomic position is not specified in the FASTA header (see [Genomic Position](#genomic-position)), CIP runs inference for all four positions and shows the result with the highest CpG island probability, annotated with `Best position: ...`. All four results are always written to the output CSV.

Results are saved to `outs/<timestamp>.csv`. Using --bed or --gff3 flags along with the sequence(s) file the output will be saved also to `outs/<timestamp>.bed` or `outs/<timestamp>.gff3`. Type `/quit` or press `Ctrl+C` to exit.

## Genomic Position

The model uses the genomic position of each sequence as an input feature. CIP handles this in two ways:

**Position specified in the FASTA header** — if the header contains one of the four recognised labels as a standalone word (case-insensitive), that position is used directly:

```
>seq_1 upstream
>seq_2 gene_body
>seq_3 downstream
>seq_4 intergenic
```

**Position not specified** — CIP runs inference four times (once per position) and reports the result with the highest CpG island probability in the terminal. All four predictions are written to the output CSV with their respective position labels.

Valid position labels: `upstream`, `gene_body`, `downstream`, `intergenic`.

## Project Structure

```
CpG_Island_Predictor/
├── CIP.py                      # Main entry point
├── requirements.txt            # Python dependencies
├── LICENSE                     # AGPL-3.0 license
├── README.md                   # This file
├── test.txt                    # Sample FASTA sequences for testing
├── config/
│   ├── model.pkl               # Pre-trained stacked ensemble model
│   └── metadata.json           # Model metadata (features, version, species)
└── modules/
    ├── __init__.py
    └── features_extractor.py   # Feature extraction logic
```

## Authors

- **Lorenzo Orsini** — Initial work, data preparation, implementation, modeling, documentation — [lorenzo.orsini3@gmail.com](mailto:lorenzo.orsini3@gmail.com)
- **Alex Carnoli** — Implementation, modeling — [alex.carnoli@gmail.com](mailto:alex.carnoli@gmail.com)

## License

This project is distributed under the GNU Affero General Public License v3 (AGPL-3.0) or later. See the [LICENSE](LICENSE) file for full details.
