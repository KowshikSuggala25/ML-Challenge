"""Streaming blocking for large-scale entity resolution.

Processes data in batches to avoid memory issues with large datasets.
"""
import os
import sys
import re
from collections import defaultdict
from typing import Dict, List, Set, Iterator, Tuple

# Pre-compiled regex patterns
PUNCT_RE = re.compile(r"[^\w\s]")
MULTI_SPACE_RE = re.compile(r"\s+")
NUM_RE = re.compile(r"\b\d{5,}\b")
SUFFIX_RE = re.compile(
    r"\b(ltd|inc|corp|llc|private|limited|corporation|company|pvt|ag|sa|bv|gmbh|ab|oy|sl|plc)\b",
    re.IGNORECASE
)


def normalize_name(name: str) -> str:
    if not name:
        return ""
    t = PUNCT_RE.sub(" ", str(name).lower())
    t = MULTI_SPACE_RE.sub(" ", t)
    t = SUFFIX_RE.sub("", t)
    return t.strip()


def normalize_address(addr: str) -> str:
    if not addr:
        return ""
    t = PUNCT_RE.sub(" ", str(addr).lower())
    return MULTI_SPACE_RE.sub(" ", t).strip()


def extract_bigrams(name: str) -> List[str]:
    if len(name) < 2:
        return [name] if name else []
    return list({name[i:i+2] for i in range(len(name) - 1)})


def extract_geo_tokens(addr: str, country: str) -> List[str]:
    tokens = []
    ck = str(country).strip().lower()
    if ck:
        tokens.append(ck)
    stop = {"road", "street", "ave", "avenue", "blvd", "drive", "lane", "court",
            "place", "square", "park", "way", "boulevard", "st", "rd", "nr", "near"}
    for tok in NUM_RE.sub("", str(addr).lower()).split():
        if len(tok) >= 3 and tok not in stop:
            tokens.append(tok)
    return tokens


def extract_pin(addr: str) -> List[str]:
    return NUM_RE.findall(str(addr).lower())


def read_tsv_chunks(filepath: str, chunk_size: int = 100000):
    """Read TSV file line by line, yielding (eid, name, addr, country)."""
    with open(filepath, 'rb') as bf:
        # Read and skip header
        header = bf.readline()
        for line in bf:
            if not line.strip():
                continue
            parts = line.decode('utf-8', errors='replace').rstrip('\n').rstrip('\r').split('\t')
            if len(parts) >= 4:
                yield (parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip())


def build_indices_streaming(sources: list, chunk_size: int = 500000):
    """Build blocking indices from multiple sources in streaming fashion."""
    name_index = defaultdict(set)
    geo_index = defaultdict(set)
    pin_index = defaultdict(set)

    for source_path in sources:
        count = 0
        for eid, name, addr, country in read_tsv_chunks(source_path, chunk_size):
            norm_name = normalize_name(name)
            norm_addr = normalize_address(addr)

            for bg in extract_bigrams(norm_name):
                if bg:
                    name_index[bg].add(eid)

            for tok in extract_geo_tokens(norm_addr, country):
                if tok:
                    geo_index[tok].add(eid)

            for pin in extract_pin(addr):
                if pin:
                    pin_index[pin].add(eid)

            count += 1
            if count % 100000 == 0:
                print(f"  Indexed {count:,} records from {os.path.basename(source_path)}")

    return name_index, geo_index, pin_index


def stream_candidates_s1(s_path: str, all_name_idx, all_geo_idx, all_pin_idx, max_cands: int):
    """Stream S1 entities and find candidates using indices."""
    candidates = {}

    for eid, name, addr, country in read_tsv_chunks(s_path):
        norm_name = normalize_name(name)
        norm_addr = normalize_address(addr)

        cand_set = set()

        for bg in extract_bigrams(norm_name):
            if bg in all_name_idx:
                cand_set.update(all_name_idx[bg])

        for tok in extract_geo_tokens(norm_addr, country):
            if tok in all_geo_idx:
                cand_set.update(all_geo_idx[tok])

        for pin in extract_pin(addr):
            if pin in all_pin_idx:
                cand_set.update(all_pin_idx[pin])

        cand_set.discard(eid)
        cand_set = {c for c in cand_set if c.startswith(("S2-", "S3-"))}

        if len(cand_set) > max_cands:
            cand_set = set(sorted(cand_set)[:max_cands])

        candidates[eid] = cand_set

        if len(candidates) % 50000 == 0:
            print(f"  Processed {len(candidates):,} S1 entities...")

    return candidates


def run_streaming_blocking(test_dir: str, output_path: str, max_candidates: int = 500):
    """Main entry point for streaming blocking."""
    import time

    print("Building blocking indices from S2/S3...")
    start = time.time()

    indices = build_indices_streaming([
        os.path.join(test_dir, "test_source2.tsv"),
        os.path.join(test_dir, "test_source3.tsv")
    ])

    all_name_idx, all_geo_idx, all_pin_idx = indices
    print(f"Indices built in {time.time()-start:.1f}s")
    print(f"  Name keys: {len(all_name_idx):,}")
    print(f"  Geo keys: {len(all_geo_idx):,}")
    print(f"  PIN keys: {len(all_pin_idx):,}")

    print("\nGenerating candidates from S1...")
    start = time.time()
    candidates = stream_candidates_s1(
        os.path.join(test_dir, "test_source1.tsv"),
        all_name_idx, all_geo_idx, all_pin_idx,
        max_candidates
    )

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_eid in candidates:
            cands = sorted(candidates[s1_eid])
            f.write(f"{s1_eid}\t{','.join(cands)}\n")

    total = sum(len(v) for v in candidates.values())
    avg = total / len(candidates) if candidates else 0
    print(f"\nCompleted in {time.time()-start:.1f}s")
    print(f"Total candidates: {total:,}, avg per S1: {avg:.1f}")
    print(f"Written to {output_path}")

    return candidates


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output", default="output/candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=500)
    args = parser.parse_args()

    run_streaming_blocking(args.test_dir, args.output, args.max_candidates)