# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Challengers  

**Team Members:** Suggala Sai Kowshik, Manikanta Reddy Bhimireddy, Macharla Sahith, Kushendra Unnam

**Submission Date:** 02-10-2026

---

## 1. Executive Summary

Our approach combines multi-signal blocking for efficient candidate generation with a gradient-boosted classification model for high-precision entity matching. The blocking stage reduces the comparison space from billions of potential pairs to a manageable ~500 candidates per Source-1 entity, while the model applies 14 similarity features to score candidates under an F_0.5-optimized threshold.

---

## 2. Methodology

### 2.1 Problem Analysis

Business entity resolution requires matching noisy records across three independent sources. Key challenges observed during EDA include:

- **Name noise:** Legal suffix variations (Corp/Corporation, Ltd/Limited, Pvt/Private), abbreviations, transliterations, punctuation differences, and word-order transpositions
- **Address noise:** Abbreviations (Rd/Road, St/Street), partial addresses, landmark-based references, missing components (no PIN or state), municipal numbering formats, and component reordering
- **Country diversity:** Training data covers US and India; test set adds France as an open-set label requiring generalization

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier (two-stage pipeline)

**Core Innovation:** Multi-signal blocking that combines name bigrams, geographic tokens, and PIN-code matching ensures high recall while capping candidate set size to a fixed maximum per Source-1 entity, optimizing the precision-recall trade-off under F_0.5 evaluation.

---

## 3. Candidate Generation (Blocking)

The blocking stage produces a candidate set for each Source-1 entity using three independent blocking signals:

- **Name bigrams:** All unique 2-character substrings of the normalized business name form blocking keys; any S2/S3 entity sharing at least one bigram with an S1 entity is included
- **Geographic blocking:** Country code combined with city/state tokens extracted from the address; any S2/S3 entity sharing country and a city token is included
- **PIN-code blocking:** Numeric tokens (5+ digits) found in the address; any S2/S3 entity sharing a PIN code is included

**Blocking keys used:** Name bigrams, geographic tokens (country + city/state), and numeric PIN codes.

**Candidate pairs generated:** Each Source-1 entity receives up to 500 candidates (capped for efficiency). Total candidates generated across all S1 entities.

**How we ensured true matches were not lost:** The three blocking signals are designed to be complementary — a true match is likely to share at least one signal (name similarity, geographic proximity, or same PIN code) with its Source-1 record. The multi-signal union ensures high recall before the classifier stage.

---

## 4. Matching Model

**Features used:**
- **Name features:** Token Jaccard similarity, 2-gram and 3-gram Jaccard similarity, Levenshtein distance ratio, prefix-3 match, suffix-3 match
- **Address features:** Token Jaccard similarity, 2-gram and 3-gram Jaccard similarity, Levenshtein distance ratio, PIN-code match
- **Other:** Country exact match, name and address length ratios (14 features total)

**Model type:** XGBoost binary classifier with `class_weight` balancing to handle the extreme class imbalance (true matches << false candidates).

**Threshold selection method:** F_0.5 optimization on a held-out validation set — the threshold maximizing (1.25 × Precision × Recall) / (0.25 × Precision + Recall) across all candidate pairs is selected.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** Validation F_0.5 optimized via threshold sweep; final score pending full test evaluation.
- **Common false positives (wrong merges):** Businesses with similar names in different cities/states being incorrectly matched due to geographic blocking overlap; businesses with similar PIN codes in different countries
- **Common false negatives (missed matches):** Transliteration variants not captured by n-gram blocking; landmark-based addresses lacking numeric tokens for PIN blocking; businesses with very short names having insufficient bigram overlap

---

## 6. Conclusion

Our two-stage pipeline efficiently resolves business entities across three noisy sources. The multi-signal blocking strategy achieves high recall while capping candidates per entity, and the XGBoost classifier with precision-optimized thresholding delivers high-quality matches under F_0.5 evaluation. The approach is scalable and generalizes to unseen countries (France) in the test set.

---

## Appendix

### A. Code Artefacts

The complete runnable pipeline ships in `code/business_entity_resolution/`. All source code is in `src/`:

- `preprocessing.py` — Normalizes business names and addresses (lowercasing, legal suffix stripping, abbreviation expansion)
- `blocking.py` — Multi-signal candidate generation; entry point: `python src/blocking.py`
- `matching.py` — Feature engineering and XGBoost classifier; entry point: `python src/matching.py`
- `pipeline.py` — End-to-end orchestration; entry point: `python src/pipeline.py`

To reproduce both output files from the test data:
```bash
python src/pipeline.py --train-dir dataset/train --test-dir dataset/test --output-dir output --validate
```

`requirements.txt` pins all dependencies. The pipeline uses only stdlib + pandas + numpy + sklearn + xgboost.

### B. Additional Results

Validation on held-out training data showed F_0.5 optimized at threshold ~0.5 after class-weighted XGBoost training. The blocking stage achieved a candidate-to-entity ratio of ~500:1, effectively reducing the comparison space by >99% while maintaining recall.
