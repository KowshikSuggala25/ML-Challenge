"""Matching model: feature engineering + classifier for entity resolution."""
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set, Optional
from preprocessing import normalize_name, normalize_address


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _token_set(text: str) -> set:
    return set(text.split()) if text else set()


def _char_ngrams(text: str, n: int = 3) -> set:
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def _levenshtein_sim(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity (1 - distance/max_len)."""
    if not s1 or not s2:
        return 0.0
    # Quick check
    if s1 == s2:
        return 1.0
    # Simple DP for distance
    m, n = len(s1), len(s2)
    if m > n:
        s1, s2 = s2, s1
        m, n = n, m
    prev = list(range(m + 1))
    for i, c2 in enumerate(s2, 1):
        curr = [i] + [0] * m
        for j, c1 in enumerate(s1, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j-1] + 1,
                prev[j-1] + (c1 != c2)
            )
        prev = curr
    return 1.0 - prev[m] / max(len(s1), len(s2))


def _country_match(c1: str, c2: str) -> float:
    return 1.0 if c1 and c1 == c2 else 0.0


def _extract_pin(address: str) -> str:
    import re
    pins = re.findall(r"\b\d{5,}\b", address)
    return pins[0] if pins else ""


def _pin_match(a1: str, a2: str) -> float:
    p1, p2 = _extract_pin(a1), _extract_pin(a2)
    return 1.0 if p1 and p1 == p2 else 0.0


def _name_overlap(name1: str, name2: str) -> float:
    """Token overlap ratio (intersection over union of tokens)."""
    return _jaccard_similarity(_token_set(name1), _token_set(name2))


def _name_ngram_jaccard(name1: str, name2: str, n: int = 3) -> float:
    return _jaccard_similarity(_char_ngrams(name1, n), _char_ngrams(name2, n))


def _addr_token_jaccard(addr1: str, addr2: str) -> float:
    return _jaccard_similarity(_token_set(addr1), _token_set(addr2))


def _addr_ngram_jaccard(addr1: str, addr2: str, n: int = 3) -> float:
    return _jaccard_similarity(_char_ngrams(addr1, n), _char_ngrams(addr2, n))


def compute_features(
    s1_name: str, s1_addr: str, s1_country: str,
    s2_name: str, s2_addr: str, s2_country: str
) -> np.ndarray:
    """Compute feature vector for a candidate pair."""
    feats = []

    # Name features
    feats.append(_name_overlap(s1_name, s2_name))
    feats.append(_name_ngram_jaccard(s1_name, s2_name, 2))
    feats.append(_name_ngram_jaccard(s1_name, s2_name, 3))
    feats.append(_levenshtein_sim(s1_name, s2_name))
    feats.append(1.0 if s1_name and s2_name and s1_name[:3] == s2_name[:3] else 0.0)
    feats.append(1.0 if s1_name and s2_name and s1_name[-3:] == s2_name[-3:] else 0.0)

    # Address features
    feats.append(_addr_token_jaccard(s1_addr, s2_addr))
    feats.append(_addr_ngram_jaccard(s1_addr, s2_addr, 2))
    feats.append(_addr_ngram_jaccard(s1_addr, s2_addr, 3))
    feats.append(_levenshtein_sim(s1_addr, s2_addr))
    feats.append(_pin_match(s1_addr, s2_addr))

    # Country feature
    feats.append(_country_match(s1_country, s2_country))

    # Length ratios (can signal partial matches)
    feats.append(len(s1_name) / max(1, len(s2_name)))
    feats.append(len(s1_addr) / max(1, len(s2_addr)))

    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = [
    "name_token_jaccard", "name_2gram_jaccard", "name_3gram_jaccard",
    "name_levenshtein", "name_prefix3_match", "name_suffix3_match",
    "addr_token_jaccard", "addr_2gram_jaccard", "addr_3gram_jaccard",
    "addr_levenshtein", "pin_match",
    "country_match",
    "name_len_ratio", "addr_len_ratio",
]


# ---------------------------------------------------------------------------
# Training / inference
# ---------------------------------------------------------------------------

def load_ground_truth(filepath: str) -> Dict[str, Set[str]]:
    """Load train_ground_truth.tsv → {s1_eid: set(matched_s2_s3_eids)}."""
    import pandas as pd
    df = pd.read_csv(filepath, sep="\t", dtype=str, na_filter=False)
    gt = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        matched = str(row.get("matched_entity_ids", "")).strip()
        if matched:
            gt[s1] = {m.strip() for m in matched.split(",") if m.strip()}
        else:
            gt[s1] = set()
    return gt


def prepare_training_data(
    train_dir: str,
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str,
    sample_ratio: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build training data from ground truth + negative samples."""
    from preprocessing import preprocess_record
    import random

    print("Loading training sources...")
    # Load and preprocess all training records
    s1_records = {}
    for df in [pd.read_csv(s1_path, sep="\t", dtype=str, na_filter=False)]:
        for _, row in df.iterrows():
            eid = str(row["entity_id"]).strip()
            _, n, a, c = preprocess_record(eid, row["business_name"], row["business_address"], row["country"])
            s1_records[eid] = (n, a, c)

    s2_records = {}
    for df in [pd.read_csv(s2_path, sep="\t", dtype=str, na_filter=False)]:
        for _, row in df.iterrows():
            eid = str(row["entity_id"]).strip()
            _, n, a, c = preprocess_record(eid, row["business_name"], row["business_address"], row["country"])
            s2_records[eid] = (n, a, c)

    s3_records = {}
    for df in [pd.read_csv(s3_path, sep="\t", dtype=str, na_filter=False)]:
        for _, row in df.iterrows():
            eid = str(row["entity_id"]).strip()
            _, n, a, c = preprocess_record(eid, row["business_name"], row["business_address"], row["country"])
            s3_records[eid] = (n, a, c)

    all_non_s1 = {**s2_records, **s3_records}
    gt = load_ground_truth(gt_path)

    X, y = [], []

    print("Building positive pairs from ground truth...")
    pos_count = 0
    for s1_eid, matches in gt.items():
        if s1_eid not in s1_records:
            continue
        s1_name, s1_addr, s1_country = s1_records[s1_eid]
        for m_eid in matches:
            if m_eid in all_non_s1:
                s2_name, s2_addr, s2_country = all_non_s1[m_eid]
                X.append(compute_features(s1_name, s1_addr, s1_country, s2_name, s2_addr, s2_country))
                y.append(1)
                pos_count += 1

    print(f"  Positive pairs: {pos_count}")

    # Build negative samples (same blocking used in production)
    print("Building negative pairs via blocking...")
    from blocking import build_candidates
    candidates = build_candidates(s1_records, s2_records, s3_records, max_candidates_per_s1=1000)

    neg_count = 0
    for s1_eid, cands in candidates.items():
        if s1_eid not in s1_records:
            continue
        true_matches = gt.get(s1_eid, set())
        s1_name, s1_addr, s1_country = s1_records[s1_eid]
        for c_eid in cands:
            if c_eid in true_matches:
                continue
            if c_eid in all_non_s1:
                s2_name, s2_addr, s2_country = all_non_s1[c_eid]
                X.append(compute_features(s1_name, s1_addr, s1_country, s2_name, s2_addr, s2_country))
                y.append(0)
                neg_count += 1

    print(f"  Negative pairs: {neg_count}")

    X = np.array(X)
    y = np.array(y)

    # Optionally subsample negatives for balanced training
    if sample_ratio < 1.0 and neg_count > pos_count:
        neg_idx = np.where(y == 0)[0]
        pos_idx = np.where(y == 1)[0]
        keep_neg = int(len(pos_idx) * sample_ratio)
        if len(neg_idx) > keep_neg:
            selected = np.random.choice(neg_idx, keep_neg, replace=False)
            keep = np.concatenate([pos_idx, selected])
            X, y = X[keep], y[keep]
            np.random.shuffle(np.arange(len(y)))  # in-place shuffle index
            X, y = X[keep], y[keep]

    print(f"Training data shape: {X.shape}, positive rate: {y.mean():.3f}")
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> object:
    """Train XGBoost classifier optimized for F_0.5."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score, fbeta_score

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        # XGBoost parameters tuned for precision (F_0.5)
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max(1, (y_train == 1).sum()),
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Find optimal threshold for F_0.5
        val_probs = model.predict_proba(X_val)[:, 1]
        best_thresh, best_f05 = 0.5, 0.0
        for thresh in np.linspace(0.1, 0.9, 81):
            y_pred = (val_probs >= thresh).astype(int)
            p = precision_score(y_val, y_pred, zero_division=0)
            r = recall_score(y_val, y_pred, zero_division=0)
            if p + r > 0:
                f05 = (1.25 * p * r) / (0.25 * p + r)
                if f05 > best_f05:
                    best_f05, best_thresh = f05, thresh

        print(f"Validation F_0.5: {best_f05:.4f} at threshold {best_thresh:.3f}")
        model.best_threshold = best_thresh
        return model

    except ImportError:
        print("xgboost not available, falling back to LogisticRegression")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ])
        pipe.fit(X, y)
        pipe.best_threshold = 0.5
        return pipe


def score_candidates(
    model: object,
    s1_records: Dict[str, Tuple],
    s2_records: Dict[str, Tuple],
    s3_records: Dict[str, Tuple],
    candidates: Dict[str, Set[str]],
    threshold: Optional[float] = None,
) -> Dict[str, Set[str]]:
    """Score all candidate pairs and return matches above threshold."""
    all_non_s1 = {**s2_records, **s3_records}
    thresh = threshold if threshold is not None else getattr(model, "best_threshold", 0.5)

    matches: Dict[str, Set[str]] = {}

    for s1_eid, cand_set in candidates.items():
        if s1_eid not in s1_records:
            matches[s1_eid] = set()
            continue

        s1_name, s1_addr, s1_country = s1_records[s1_eid]
        scored = []

        for c_eid in cand_set:
            if c_eid not in all_non_s1:
                continue
            s2_name, s2_addr, s2_country = all_non_s1[c_eid]
            feats = compute_features(s1_name, s1_addr, s1_country, s2_name, s2_addr, s2_country).reshape(1, -1)
            prob = model.predict_proba(feats)[0, 1]
            if prob >= thresh:
                scored.append((c_eid, prob))

        # Sort by confidence and take top matches
        scored.sort(key=lambda x: x[1], reverse=True)
        matches[s1_eid] = {eid for eid, _ in scored}

    return matches


def write_matching_results(matches: Dict[str, Set[str]], output_path: str):
    """Write matching_results.tsv in required format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_eid in sorted(matches.keys()):
            match_list = sorted(matches[s1_eid])
            f.write(f"{s1_eid}\t{','.join(match_list)}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train matching model and score candidates")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--candidates", default="output/candidate_pairs.tsv")
    parser.add_argument("--output", default="output/matching_results.tsv")
    args = parser.parse_args()

    # For standalone use, would need to load candidates
    print("Run via pipeline.py for full end-to-end execution.")