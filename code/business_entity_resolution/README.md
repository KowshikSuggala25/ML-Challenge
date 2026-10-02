# Business Entity Resolution Pipeline

## Overview

End-to-end ML pipeline for the ML Challenge 2026 — Business Entity Resolution.
Matches business records across three noisy independent data sources (S1, S2, S3).

## Structure

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # Normalize names & addresses
│   ├── blocking.py        # Candidate generation via multi-signal blocking
│   ├── matching.py        # Feature engineering + XGBoost classifier
│   └── pipeline.py        # Main entry point
├── requirements.txt
└── README.md
```

## Requirements

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Full end-to-end (train + predict):
```bash
python src/pipeline.py \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output \
    --validate
```

### Candidate generation only:
```bash
python src/blocking.py \
    --test-dir dataset/test \
    --output output/candidate_pairs.tsv \
    --max-candidates 500
```

### Training only:
```bash
python src/matching.py \
    --train-dir dataset/train
```

## How It Works

### 1. Preprocessing
- Normalize business names (lowercase, strip legal suffixes, punctuation)
- Normalize addresses (expand abbreviations, clean)
- Standardize country labels

### 2. Blocking (Candidate Generation)
Three independent blocking signals ensure high recall while keeping candidate
sets small:
- **Name bigrams**: Two-character substrings of normalized names
- **Geographic**: Country code + city/state tokens
- **PIN-code**: Numeric tokens in addresses

This multi-signal approach means a true match is likely to appear in at least
one blocking index, while producing a manageable candidate set per S1 entity.

### 3. Matching Model
- **Features** (14): Name token/ngram Jaccard, Levenshtein, prefix/suffix
  matches; Address token/ngram Jaccard, Levenshtein, PIN match; Country
  match; Name/address length ratios
- **Model**: XGBoost classifier with `class_weight` balancing
- **Threshold**: Optimized on validation data for F_0.5 (precision-heavy)

### 4. Output
- `output/matching_results.tsv` — Final matches for each S1 entity
- `output/candidate_pairs.tsv` — Candidate set from blocking stage

## Validation
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test \
    --check-ids
```

## Model License
MIT License.
