"""End-to-end pipeline for Business Entity Resolution Challenge.

This is the main entry point. It orchestrates:
1. Candidate generation (blocking)
2. Model training (on training data)
3. Inference on test candidates
4. Output generation (matching_results.tsv, candidate_pairs.tsv)
"""
import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np

# Local modules
from preprocessing import preprocess_record, normalize_name, normalize_address
from blocking import build_candidates, load_source
from matching import (
    compute_features, load_ground_truth, prepare_training_data,
    train_model, score_candidates, write_matching_results,
    FEATURE_NAMES
)


def run_full_pipeline(
    train_dir: str = "dataset/train",
    test_dir: str = "dataset/test",
    output_dir: str = "output",
    model_path: str = "models/entity_resolution_model.pkl",
    max_candidates_per_s1: int = 500,
    retrain: bool = True,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Execute the complete entity resolution pipeline.

    Returns
    -------
    (candidates, matches) : both are Dict[source1_eid, Set[candidate_eid]]
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # ============================================================
    # STEP 1: Load test data and generate candidate pairs
    # ============================================================
    print("=" * 60)
    print("STEP 1: Candidate Generation (Blocking)")
    print("=" * 60)

    s1_test = load_source(os.path.join(test_dir, "test_source1.tsv"))
    s2_test = load_source(os.path.join(test_dir, "test_source2.tsv"))
    s3_test = load_source(os.path.join(test_dir, "test_source3.tsv"))

    print(f"Test data: {len(s1_test):,} S1, {len(s2_test):,} S2, {len(s3_test):,} S3")

    test_candidates = build_candidates(
        s1_test, s2_test, s3_test,
        block_name=True,
        block_geo=True,
        block_pin=True,
        max_candidates_per_s1=max_candidates_per_s1,
    )

    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    total_cands = sum(len(v) for v in test_candidates.values())
    avg_cands = total_cands / len(test_candidates) if test_candidates else 0
    print(f"Candidates: {total_cands:,} total, {avg_cands:.1f} avg per S1")

    # Write candidate_pairs.tsv
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_eid in sorted(test_candidates.keys()):
            cand_list = sorted(test_candidates[s1_eid])
            f.write(f"{s1_eid}\t{','.join(cand_list)}\n")
    print(f"Written: {candidate_path}")

    # ============================================================
    # STEP 2: Train matching model on training data
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 2: Train Matching Model")
    print("=" * 60)

    model = None
    if retrain or not os.path.exists(model_path):
        s1_train = load_source(os.path.join(train_dir, "train_source1.tsv"))
        s2_train = load_source(os.path.join(train_dir, "train_source2.tsv"))
        s3_train = load_source(os.path.join(train_dir, "train_source3.tsv"))

        print(f"Train data: {len(s1_train):,} S1, {len(s2_train):,} S2, {len(s3_train):,} S3")

        # Prepare training data
        X, y = prepare_training_data(
            train_dir,
            os.path.join(train_dir, "train_source1.tsv"),
            os.path.join(train_dir, "train_source2.tsv"),
            os.path.join(train_dir, "train_source3.tsv"),
            os.path.join(train_dir, "train_ground_truth.tsv"),
            sample_ratio=1.0,  # use all negatives from blocking
        )

        # Train model
        model = train_model(X, y)

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_names": FEATURE_NAMES,
                "threshold": getattr(model, "best_threshold", 0.5),
            }, f)
        print(f"Model saved to {model_path}")
    else:
        print(f"Loading existing model from {model_path}")
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        model = saved["model"]
        print(f"Loaded model with threshold {saved.get('threshold', 0.5):.3f}")

    # ============================================================
    # STEP 3: Score test candidates and generate final matches
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 3: Score Candidates & Generate Final Matches")
    print("=" * 60)

    test_matches = score_candidates(
        model,
        s1_test, s2_test, s3_test,
        test_candidates,
        threshold=getattr(model, "best_threshold", 0.5),
    )

    match_path = os.path.join(output_dir, "matching_results.tsv")
    write_matching_results(test_matches, match_path)
    print(f"Written: {match_path}")

    # Summary stats
    total_matches = sum(len(v) for v in test_matches.values())
    s1_with_matches = sum(1 for v in test_matches.values() if v)
    avg_matches = total_matches / len(test_matches) if test_matches else 0
    print(f"Final matches: {total_matches:,} total, {s1_with_matches:,} S1 entities matched, {avg_matches:.2f} avg per S1")

    return test_candidates, test_matches


def validate_outputs(output_dir: str, test_dir: str) -> bool:
    """Run the submission validator on generated outputs."""
    import subprocess
    validator = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "validate_submission.py")
    if not os.path.exists(validator):
        print("Validator not found, skipping validation")
        return True

    result = subprocess.run([
        sys.executable, validator,
        "--matching", os.path.join(output_dir, "matching_results.tsv"),
        "--candidate", os.path.join(output_dir, "candidate_pairs.tsv"),
        "--test-dir", test_dir,
        "--check-ids"
    ], capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return result.returncode == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run full entity resolution pipeline")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-path", default="models/entity_resolution_model.pkl")
    parser.add_argument("--max-candidates", type=int, default=500)
    parser.add_argument("--no-retrain", action="store_true", help="Use existing model")
    parser.add_argument("--validate", action="store_true", help="Run validator after")
    args = parser.parse_args()

    candidates, matches = run_full_pipeline(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        max_candidates_per_s1=args.max_candidates,
        retrain=not args.no_retrain,
    )

    if args.validate:
        print("\n" + "=" * 60)
        print("VALIDATION")
        print("=" * 60)
        ok = validate_outputs(args.output_dir, args.test_dir)
        if ok:
            print("✓ Validation PASSED")
        else:
            print("✗ Validation FAILED")
            sys.exit(1)