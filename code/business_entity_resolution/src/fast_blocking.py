"""Optimized blocking for large-scale entity resolution.

Uses memory-efficient chunked processing and optimized data structures.
"""
import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set

# Pre-compiled regex patterns
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\b\d{5,}\b")

# Legal suffixes pattern
SUFFIX_RE = re.compile(
    r"\b(ltd|inc|corp|llc|private|limited|corporation|company|pvt|ag|sa|bv|gmbh|ab|oy|sl|plc)\b",
    re.IGNORECASE
)


def normalize_name(name: str) -> str:
    """Fast name normalization."""
    if not name:
        return ""
    t = PUNCT_RE.sub(" ", str(name).lower())
    t = MULTI_SPACE_RE.sub(" ", t)
    t = SUFFIX_RE.sub("", t)
    return t.strip()


def normalize_address(addr: str) -> str:
    """Fast address normalization."""
    if not addr:
        return ""
    t = PUNCT_RE.sub(" ", str(addr).lower())
    return MULTI_SPACE_RE.sub(" ", t).strip()


def extract_bigrams(name: str) -> List[str]:
    """Extract 2-char bigrams from normalized name."""
    if len(name) < 2:
        return [name] if name else []
    return list({name[i:i+2] for i in range(len(name) - 1)})


def extract_geo_tokens(addr: str, country: str) -> List[str]:
    """Extract geographic tokens from address + country."""
    tokens = []
    ck = str(country).strip().lower()
    if ck:
        tokens.append(ck)
    # City-like tokens (2+ char words that aren't address words)
    stop = {"road", "street", "ave", "avenue", "blvd", "drive", "lane", "court",
            "place", "square", "park", "way", "boulevard", "st", "rd", "nr", "near"}
    for tok in NUM_RE.sub("", str(addr).lower()).split():
        if len(tok) >= 3 and tok not in stop:
            tokens.append(tok)
    return tokens


def extract_pin(addr: str) -> List[str]:
    """Extract PIN codes from address."""
    return NUM_RE.findall(str(addr).lower())


def build_blocking_index(filepath: str, source_type: str, block_geo: bool = True):
    """Build blocking indices from a source file.

    Returns dict of {block_key: set(entity_ids)} for name and geo keys.
    """
    name_index = defaultdict(set)
    geo_index = defaultdict(set)
    pin_index = defaultdict(set)

    count = 0
    with open(filepath, 'rb') as bf:
        content = bf.read().decode('utf-8', errors='replace')

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue

        eid = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        addr = parts[2].strip() if len(parts) > 2 else ""
        country = parts[3].strip() if len(parts) > 3 else ""

        norm_name = normalize_name(name)
        norm_addr = normalize_address(addr)

        # Name blocking
        for bg in extract_bigrams(norm_name):
            if bg:
                name_index[bg].add(eid)

        # Geographic blocking
        if block_geo:
            for tok in extract_geo_tokens(norm_addr, country):
                if tok:
                    geo_index[tok].add(eid)

        # PIN blocking
        for pin in extract_pin(addr):
            if pin:
                pin_index[pin].add(eid)

        count += 1

    return name_index, geo_index, pin_index, count


def generate_candidates_fast(
    test_dir: str,
    output_path: str,
    max_candidates_per_s1: int = 500,
):
    """Generate candidate pairs using memory-efficient blocking."""
    import time

    print("Loading and building blocking indices...")
    start = time.time()

    # Build indices for S1, S2, S3
    s1_name_idx, s1_geo_idx, s1_pin_idx, s1_count = build_blocking_index(
        os.path.join(test_dir, "test_source1.tsv"), "S1"
    )
    s2_name_idx, s2_geo_idx, s2_pin_idx, s2_count = build_blocking_index(
        os.path.join(test_dir, "test_source2.tsv"), "S2"
    )
    s3_name_idx, s3_geo_idx, s3_pin_idx, s3_count = build_blocking_index(
        os.path.join(test_dir, "test_source3.tsv"), "S3"
    )

    # Combine S2 and S3 into a single lookup
    # We need to map from entity_id to (name, addr, country) but for blocking
    # we just need the indices

    all_name_idx = defaultdict(set)
    all_geo_idx = defaultdict(set)
    all_pin_idx = defaultdict(set)

    for k, v in s2_name_idx.items():
        all_name_idx[k].update(v)
    for k, v in s3_name_idx.items():
        all_name_idx[k].update(v)
    for k, v in s2_geo_idx.items():
        all_geo_idx[k].update(v)
    for k, v in s3_geo_idx.items():
        all_geo_idx[k].update(v)
    for k, v in s2_pin_idx.items():
        all_pin_idx[k].update(v)
    for k, v in s3_pin_idx.items():
        all_pin_idx[k].update(v)

    print(f"Indices built: S1={s1_count:,}, S2+S3={(s2_count+s3_count):,}")
    print(f"Building candidates...")

    # Now read S1 entities and find candidates
    candidates = {}

    with open(os.path.join(test_dir, "test_source1.tsv"), 'rb') as bf:
        content = bf.read().decode('utf-8', errors='replace')

    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("entity_id"):
            continue
        parts = line.split('\t')
        if len(parts) < 4:
            continue

        s1_eid = parts[0].strip()
        name = parts[1].strip() if len(parts) > 1 else ""
        addr = parts[2].strip() if len(parts) > 2 else ""
        country = parts[3].strip() if len(parts) > 3 else ""

        norm_name = normalize_name(name)
        norm_addr = normalize_address(addr)

        cand_set = set()

        # Name-based candidates
        for bg in extract_bigrams(norm_name):
            if bg in all_name_idx:
                cand_set.update(all_name_idx[bg])

        # Geo-based candidates
        for tok in extract_geo_tokens(norm_addr, country):
            if tok in all_geo_idx:
                cand_set.update(all_geo_idx[tok])

        # PIN-based candidates
        for pin in extract_pin(addr):
            if pin in all_pin_idx:
                cand_set.update(all_pin_idx[pin])

        # Filter out S1 IDs
        cand_set.discard(s1_eid)
        cand_set = {c for c in cand_set if c.startswith(("S2-", "S3-"))}

        # Cap
        if len(cand_set) > max_candidates_per_s1:
            cand_set = set(sorted(cand_set)[:max_candidates_per_s1])
        candidates[s1_eid] = cand_set

        if len(candidates) % 50000 == 0:
            print(f"  Processed {len(candidates):,} S1 entities...")

    print(f"Built candidates for {len(candidates):,} S1 entities in {time.time()-start:.1f}s")

    # Count stats
    total = sum(len(v) for v in candidates.values())
    avg = total / len(candidates) if candidates else 0
    print(f"Total candidates: {total:,}, avg per S1: {avg:.1f}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_eid in sorted(candidates.keys()):
            cands = sorted(candidates[s1_eid])
            f.write(f"{s1_eid}\t{','.join(cands)}\n")

    print(f"Written to {output_path}")
    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args()

    generate_candidates_fast(args.test_dir, args.output, args.max_candidates)