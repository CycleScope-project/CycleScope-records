# CycleScope execution records

[中文](README.zh-CN.md)

The per-instruction execution records of the [CycleScope] evaluation
suites. For each of the 1,000 suite kernels, 500 on NVIDIA H100 and 500
on NVIDIA B300, this repository provides what the simulator wrote:

- the **issue journal**: one row per issued instruction, with its issue
  cycle, how long it waited, and what it waited for;
- the **wait ledger**: per warp and cause, the total wait;
- the **result**: the totals of the run.

Together they hold 7.35 billion issued instructions. The simulated
cycle counts are the ones reported in the CycleScope repository; every
`result.bin` here is byte-identical to the one listed in its
`suites/<arch>/full_gpu/results/result_bin.sha256`.

## Download

The archives are attached to the
[latest release](../../releases/latest). The repository itself holds
only the manifests, the checksums and a checker.

### NVIDIA H100 (Hopper)

| Archive | Library | Kernels | Instructions | Size | Uncompressed |
|---|---|---|---|---|---|
| `ledgers-h100-cublas.tar` | cuBLAS | 50 | 392 M | 1.10 GiB | 17.5 GiB |
| `ledgers-h100-cutlass-1.tar` | CUTLASS | 46 | 401 M | 1.20 GiB | 17.9 GiB |
| `ledgers-h100-cutlass-2.tar` | CUTLASS | 54 | 399 M | 1.18 GiB | 17.8 GiB |
| `ledgers-h100-deepgemm-1.tar` | DeepGEMM | 78 | 422 M | 1.22 GiB | 18.9 GiB |
| `ledgers-h100-deepgemm-2.tar` | DeepGEMM | 22 | 423 M | 1.35 GiB | 18.9 GiB |
| `ledgers-h100-flashattention-1.tar` | FlashAttention | 38 | 416 M | 1.32 GiB | 18.6 GiB |
| `ledgers-h100-flashattention-2.tar` | FlashAttention | 31 | 385 M | 1.28 GiB | 17.2 GiB |
| `ledgers-h100-flashattention-3.tar` | FlashAttention | 31 | 410 M | 1.35 GiB | 18.3 GiB |
| `ledgers-h100-flashinfer.tar` | FlashInfer | 50 | 496 M | 1.66 GiB | 22.2 GiB |
| `ledgers-h100-flashmla.tar` | FlashMLA | 50 | 487 M | 1.44 GiB | 21.8 GiB |
| `ledgers-h100-triton.tar` | Triton | 50 | 549 M | 1.84 GiB | 24.6 GiB |
| **total** | | **500** | **4.78 G** | **14.9 GiB** | **214 GiB** |

### NVIDIA B300 (Blackwell)

| Archive | Library | Kernels | Instructions | Size | Uncompressed |
|---|---|---|---|---|---|
| `ledgers-b300-cublas.tar` | cuBLAS | 50 | 263 M | 0.66 GiB | 11.8 GiB |
| `ledgers-b300-cutlass.tar` | CUTLASS | 100 | 422 M | 1.36 GiB | 18.9 GiB |
| `ledgers-b300-deepgemm.tar` | DeepGEMM | 100 | 300 M | 1.09 GiB | 13.4 GiB |
| `ledgers-b300-flashattention.tar` | FlashAttention | 100 | 533 M | 1.77 GiB | 23.8 GiB |
| `ledgers-b300-flashinfer.tar` | FlashInfer | 50 | 362 M | 1.20 GiB | 16.2 GiB |
| `ledgers-b300-flashmla.tar` | FlashMLA | 50 | 377 M | 1.27 GiB | 16.8 GiB |
| `ledgers-b300-triton.tar` | Triton | 50 | 309 M | 0.99 GiB | 13.8 GiB |
| **total** | | **500** | **2.57 G** | **8.3 GiB** | **115 GiB** |

A library that is split into numbered parts unpacks into one
directory. Check a download and unpack it:

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf ledgers-h100-cublas.tar
```

## What is inside

```
h100-cublas/
  <case>/
    journal.bin.xz   the issue journal, xz-compressed
    wait.bin.xz      the wait ledger, xz-compressed
    result.bin       the result totals
    ledgers.tsv      size and SHA-256 of both ledgers before compression
```

`manifest-<device>.tsv` lists every case: its library, its archive, the
number of issued instructions, the simulated cycles, and the size and
SHA-256 of each file. To check unpacked archives against it:

```bash
python3 check.py <directory you unpacked into>
```

## Reading a record

The byte layouts are described in the CycleScope documentation
(`docs/en/formats.md`, section 7, and `docs/en/results.md`). With a
CycleScope checkout on `PYTHONPATH`:

```python
import lzma
from pathlib import Path
from cyclescope.job.result import read_journal, read_result

case = Path("h100-deepgemm/deepgemm_einsum_05")
result = read_result((case / "result.bin").read_bytes())
journal = read_journal(lzma.open(case / "journal.bin.xz").read())["JRNL"]
print(result["cycles"], "cycles,", len(journal), "instructions")
print(journal[0])
```

`read_journal` builds one dictionary per row, which is slow for large
kernels: the largest journal has 65.5 million rows and is 3.1 GB
uncompressed. A journal is a 104-byte header followed by fixed 48-byte
rows, so NumPy can map it directly:

```python
import lzma
import numpy as np

JRNL = np.dtype([("warp", "<u4"), ("smsp", "<u4"), ("instance", "<u4"),
                 ("issue_cycle", "<u8"), ("operation_class", "<u4"),
                 ("template", "<u4"), ("wait_bucket", "<u4"),
                 ("wait_cycles", "<u8"), ("object", "<u8")])
raw = lzma.open("h100-deepgemm/deepgemm_einsum_05/journal.bin.xz").read()
rows = np.frombuffer(raw, dtype=JRNL, offset=104)
waited = rows[rows["wait_bucket"] != 0xFFFFFFFF]      # 0xFFFFFFFF: no wait
print(len(rows), "instructions,", int(waited["wait_cycles"].sum()), "warp-cycles waited")
```

## How the records were produced

Each kernel was simulated once from its baseline capture with the issue
journal and the wait ledger enabled. Before packing, every case was
checked: the run finished with all instructions retired, `result.bin`
equals the baseline result byte for byte, the journal holds exactly one
row per issued instruction, and each compressed ledger decompresses to
the size and SHA-256 recorded before compression.

## License

MIT, see [LICENSE](LICENSE).

[CycleScope]: https://github.com/CycleScope-project/CycleScope
