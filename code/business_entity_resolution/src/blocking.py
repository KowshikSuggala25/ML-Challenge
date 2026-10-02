"""Blocking / candidate-generation strategies for entity resolution.

Design goal: produce a *small* candidate set per Source-1 entity while
maintaining high recall on true matches.

Blocking keys are built from multiple independent signals so that a true
match is likely to share at least one key with its source-1 record.
"""
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Blocking key generators
# ---------------------------------------------------------------------------

def _name_bigrams(name: str) -> List[str]:
    """Return sorted unique bigrams from a normalized name."""
    if len(name) < 2:
        return [name] if name else []
    return sorted({name[i:i+2] for i in range(len(name) - 1)})


def _phonetic_key(name: str) -> str:
    """Simple phonetic hash (first 2 chars + length). Not Soundex but fast."""
    name = name.strip()
    if not name:
        return ""
    # Use first 2 chars + last 2 chars + length as a rough phonetic fingerprint
    return f"{name[:2]}_{name[-2:]}_{len(name)}" if len(name) >= 4 else name


def _country_key(country: str) -> str:
    return country.strip().lower()


def _pin_like_key(address: str) -> List[str]:
    """Extract PIN-like tokens (5+ digit sequences) from address."""
    import re
    return re.findall(r"\b\d{5,}\b", address)


def _city_state_key(address: str) -> List[str]:
    """Extract city/state tokens (2+ letter words) from address."""
    import re
    # Get tokens that look like city names (2+ chars, no digits)
    tokens = re.findall(r"\b[a-z]{2,}\b", address)
    # Filter out common address words
    stop = {"road", "street", "avenue", "boulevard", "drive", "lane", "court",
            "place", "square", "park", "way", "highway", "trail", "circle",
            "plaza", "terrace", "expressway", "freeway", "parkway", "suite",
            "floor", "building", "department", "unit", "near", "behind",
            "above", "below", "opposite", "next", "corner", "block"}
    return sorted({t for t in tokens if t not in stop})


# ---------------------------------------------------------------------------
# Main blocking function
# ---------------------------------------------------------------------------

def build_candidates(
    s1_records: Dict[str, Tuple],
    s2_records: Dict[str, Tuple],
    s3_records: Dict[str, Tuple],
    block_name: bool = True,
    block_geo: bool = True,
    block_pin: bool = True,
    max_candidates_per_s1: int = 500,
) -> Dict[str, Set[str]]:
    """Generate candidate (S1 → S2/S3) pairs using multi-signal blocking.

    Parameters
    ----------
    s1_records : dict  {entity_id: (norm_name, norm_address, country)}
    s2_records : dict  {entity_id: (norm_name, norm_address, country)}
    s3_records : dict  {entity_id: (norm_name, norm_address, country)}
    block_name : use name bigram blocking
    block_geo  : use country + city/state blocking
    block_pin  : use PIN-code blocking
    max_candidates_per_s1 : cap candidates per S1 entity

    Returns
    -------
    dict {s1_entity_id: set(candidate_s2_s3_ids)}
    """
    all_non_s1 = {**s2_records, **s3_records}

    # Build inverted indices per blocking signal
    bigram_index: Dict[str, Set[str]] = defaultdict(set)
    geo_index: Dict[str, Set[str]] = defaultdict(set)
    pin_index: Dict[str, Set[str]] = defaultdict(set)

    # Index non-S1 records
    for eid, (name, addr, country) in all_non_s1.items():
        if block_name and name:
            for bg in _name_bigrams(name):
                bigram_index[bg].add(eid)
        if block_geo:
            ck = _country_key(country)
            geo_index[ck].add(eid)
            for tok in _city_state_key(addr):
                geo_index[f"{ck}_{tok}"].add(eid)
        if block_pin:
            for pin in _pin_like_key(addr):
                pin_index[pin].add(eid)

    candidate_sets: Dict[str, Set[str]] = {}

    for s1_eid, (s1_name, s1_addr, s1_country) in s1_records.items():
        candidates: Set[str] = set()

        # --- Name bigram blocking ---
        if block_name and s1_name:
            for bg in _name_bigrams(s1_name):
                candidates.update(bigram_index.get(bg, set()))

        # --- Geographic blocking ---
        if block_geo:
            ck = _country_key(s1_country)
            candidates.update(geo_index.get(ck, set()))
            for tok in _city_state_key(s1_addr):
                candidates.update(geo_index.get(f"{ck}_{tok}", set()))

        # --- PIN-code blocking ---
        if block_pin:
            for pin in _pin_like_key(s1_addr):
                candidates.update(pin_index.get(pin, set()))

        # Remove self-references and S1 IDs
        candidates.discard(s1_eid)
        candidates = {c for c in candidates if not c.startswith("S1-")}

        # Cap to keep candidate set manageable
        if len(candidates) > max_candidates_per_s1:
            # Deterministic truncation (sorted for reproducibility)
            candidates = set(sorted(candidates)[:max_candidates_per_s1])

        candidate_sets[s1_eid] = candidates

    return candidate_sets


def load_source(filepath: str) -> Dict[str, Tuple]:
    """Load a source TSV and return {entity_id: (norm_name, norm_address, country)}."""
    import pandas as pd
    from preprocessing import preprocess_record

    df = pd.read_csv(filepath, sep="\t", dtype=str, na_filter=False)
    records = {}
    for _, row in df.iterrows():
        eid = str(row["entity_id"]).strip()
        name = str(row.get("business_name", "")).strip()
        addr = str(row.get("business_address", "")).strip()
        country = str(row.get("country", "")).strip()
        _, norm_name, norm_addr, norm_country = preprocess_record(eid, name, addr, country)
        records[eid] = (norm_name, norm_addr, norm_country)
    return records


def generate_candidates(
    test_dir: str,
    output_path: str,
    max_candidates_per_s1: int = 500,
) -> Dict[str, Set[str]]:
    """Full pipeline: load test data → block → write candidate_pairs.tsv."""
    print("Loading Source 1 (test)...")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s1_records = load_source(s1_path)
    print(f"  {len(s1_records):,} S1 entities")

    print("Loading Source 2 (test)...")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s2_records = load_source(s2_path)
    print(f"  {len(s2_records):,} S2 entities")

    print("Loading Source 3 (test)...")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    s3_records = load_source(s3_path)
    print(f"  {len(s3_records):,} S3 entities")

    print("Building candidate pairs via blocking...")
    candidates = build_candidates(
        s1_records, s2_records, s3_records,
        block_name=True,
        block_geo=True,
        block_pin=True,
        max_candidates_per_s1=max_candidates_per_s1,
    )

    # Write candidate_pairs.tsv
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_cands = sum(len(v) for v in candidates.values())
    avg_cands = total_cands / len(candidates) if candidates else 0
    print(f"  Candidates generated: {total_cands:,} total, {avg_cands:.1f} avg per S1")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_eid in sorted(candidates.keys()):
            cand_list = sorted(candidates[s1_eid])
            f.write(f"{s1_eid}\t{','.join(cand_list)}\n")

    print(f"  Written to {output_path}")
    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate candidate pairs via blocking")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args()

    generate_candidates(args.test_dir, args.output, args.max_candidates)
