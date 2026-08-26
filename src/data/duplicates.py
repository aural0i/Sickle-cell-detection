"""Exact and near-duplicate image detection.

The primary dataset has no patient/slide/sample ID, so we can't group images
by their true biological source (see docs/dataset_findings.md for why). As a
partial substitute, we detect exact and near-duplicate *images* and treat
each duplicate cluster as one group during splitting, so a duplicate (or
near-duplicate) photo can never end up in both the training and test sets.

This only catches literal or near-identical copies of the same photo - it
does NOT catch different photos of the same patient/slide that happen to
look different. That residual leakage risk is real and stays documented as
a limitation; this step reduces it, it doesn't eliminate it.
"""
import hashlib
from collections import defaultdict

import imagehash
from PIL import Image

DEFAULT_NEAR_DUP_THRESHOLD = 5  # Hamming distance on the default 64-bit phash


def file_hash(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def perceptual_hash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def find_duplicate_groups(paths, near_dup_threshold=DEFAULT_NEAR_DUP_THRESHOLD):
    """Groups images that are exact or near-duplicates of each other.

    Returns (group_ids, exact_hashes, phashes):
      - group_ids: dict mapping each path -> integer group id. Images with
        no duplicates get their own unique group id, so this degrades
        gracefully to "one image = one group" when nothing is found.
      - exact_hashes / phashes: dicts of the raw hash values, for reporting.
    """
    paths = list(paths)
    exact_hashes = {p: file_hash(p) for p in paths}
    phashes = {p: perceptual_hash(p) for p in paths}

    parent = {p: p for p in paths}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Exact duplicates: group directly by identical file hash (cheap, O(n)).
    by_exact_hash = defaultdict(list)
    for p, h in exact_hashes.items():
        by_exact_hash[h].append(p)
    for group in by_exact_hash.values():
        for p in group[1:]:
            union(group[0], p)

    # Near duplicates: pairwise perceptual-hash comparison. O(n^2), but n is
    # in the low hundreds here so this is fast (well under a minute).
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            a, b = paths[i], paths[j]
            if phashes[a] - phashes[b] <= near_dup_threshold:
                union(a, b)

    roots = {p: find(p) for p in paths}
    unique_roots = sorted(set(roots.values()))
    root_to_gid = {r: i for i, r in enumerate(unique_roots)}
    group_ids = {p: root_to_gid[roots[p]] for p in paths}
    return group_ids, exact_hashes, phashes
