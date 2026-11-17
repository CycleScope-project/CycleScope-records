#!/usr/bin/env python3
"""Check unpacked CycleScope execution records against the manifests.

    python3 check.py <dir> [<dir> ...]

<dir> is where archives were unpacked: it holds <device>-<family>/<case>/
directories.  For every case found, each ledger is decompressed and its
size and SHA-256 compared with the manifest row of that case, and
result.bin is compared by SHA-256.  Exit status 0 when every case found
matches.  Needs only the Python standard library.
"""
import csv
import hashlib
import lzma
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def digest(fh):
    h = hashlib.sha256()
    size = 0
    while True:
        b = fh.read(1 << 22)
        if not b:
            return size, h.hexdigest()
        h.update(b)
        size += len(b)


def main() -> int:
    rows = {}
    for m in sorted(HERE.glob("manifest-*.tsv")):
        with m.open() as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                rows[(f"{r['device']}-{r['family']}", r["case"])] = r
    checked = bad = 0
    for root in map(Path, sys.argv[1:] or ["."]):
        for d in sorted(root.glob("*-*/*/")):
            r = rows.get((d.parent.name, d.name))
            if r is None:
                continue
            why = []
            for name in ("journal", "wait"):
                with lzma.open(d / f"{name}.bin.xz") as fh:
                    size, sha = digest(fh)
                if (str(size), sha) != (r[f"{name}_raw_bytes"],
                                        r[f"{name}_sha256"]):
                    why.append(f"{name}.bin")
            with (d / "result.bin").open("rb") as fh:
                if digest(fh)[1] != r["result_sha256"]:
                    why.append("result.bin")
            checked += 1
            if why:
                bad += 1
                print(f"MISMATCH {d}: {', '.join(why)}")
    print(f"checked {checked} cases, {bad} mismatches")
    return 1 if bad or not checked else 0


if __name__ == "__main__":
    sys.exit(main())
