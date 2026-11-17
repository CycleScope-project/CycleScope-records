# CycleScope 执行记录

[English](README.md)

[CycleScope] 评估套件的逐指令执行记录。套件共 1,000 个 kernel，NVIDIA H100 与 NVIDIA B300 各 500 个；本仓库提供仿真器为每个 kernel 写出的：

- **发射日志**：每条发射的指令一行，含发射周期、等待时长与等待原因；
- **等待账本**：按 warp 与原因汇总的等待；
- **结果**：一次运行的总量。

合计 73.5 亿条发射指令。仿真周期数就是 CycleScope 仓库里报告的那些；这里的每个 `result.bin` 都与其 `suites/<arch>/full_gpu/results/result_bin.sha256` 所列的逐字节相同。

## 下载

归档作为附件挂在[最新 release](../../releases/latest) 上。仓库本身只放清单、校验和与一个检查脚本。

### NVIDIA H100 (Hopper)

| 归档 | 库 | kernel 数 | 指令数 | 大小 | 解压后 |
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
| **合计** | | **500** | **4.78 G** | **14.9 GiB** | **214 GiB** |

### NVIDIA B300 (Blackwell)

| 归档 | 库 | kernel 数 | 指令数 | 大小 | 解压后 |
|---|---|---|---|---|---|
| `ledgers-b300-cublas.tar` | cuBLAS | 50 | 263 M | 0.66 GiB | 11.8 GiB |
| `ledgers-b300-cutlass.tar` | CUTLASS | 100 | 422 M | 1.36 GiB | 18.9 GiB |
| `ledgers-b300-deepgemm.tar` | DeepGEMM | 100 | 300 M | 1.09 GiB | 13.4 GiB |
| `ledgers-b300-flashattention.tar` | FlashAttention | 100 | 533 M | 1.77 GiB | 23.8 GiB |
| `ledgers-b300-flashinfer.tar` | FlashInfer | 50 | 362 M | 1.20 GiB | 16.2 GiB |
| `ledgers-b300-flashmla.tar` | FlashMLA | 50 | 377 M | 1.27 GiB | 16.8 GiB |
| `ledgers-b300-triton.tar` | Triton | 50 | 309 M | 0.99 GiB | 13.8 GiB |
| **合计** | | **500** | **2.57 G** | **8.3 GiB** | **115 GiB** |

被拆成带编号的几个部分的库，解包后落在同一个目录里。校验并解包：

```bash
sha256sum -c SHA256SUMS --ignore-missing
tar -xf ledgers-h100-cublas.tar
```

## 里面有什么

```
h100-cublas/
  <case>/
    journal.bin.xz   发射日志，xz 压缩
    wait.bin.xz      等待账本，xz 压缩
    result.bin       结果总量
    ledgers.tsv      两个日志压缩前的大小与 SHA-256
```

`manifest-<device>.tsv` 逐例列出：所属库、所在归档、发射指令数、仿真周期，以及每个文件的大小与 SHA-256。用它核对解包后的内容：

```bash
python3 check.py <解包到的目录>
```

## 读一份记录

字节布局见 CycleScope 文档（`docs/zh/formats.md` 第 7 节与 `docs/zh/results.md`）。把 CycleScope 的 checkout 放进 `PYTHONPATH` 后：

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

`read_journal` 为每一行建一个字典，大 kernel 上会很慢：最大的发射日志有 6550 万行，解压后 3.1 GB。发射日志是 104 字节的文件头加定长 48 字节的行，所以可以用 NumPy 直接映射：

```python
import lzma
import numpy as np

JRNL = np.dtype([("warp", "<u4"), ("smsp", "<u4"), ("instance", "<u4"),
                 ("issue_cycle", "<u8"), ("operation_class", "<u4"),
                 ("template", "<u4"), ("wait_bucket", "<u4"),
                 ("wait_cycles", "<u8"), ("object", "<u8")])
raw = lzma.open("h100-deepgemm/deepgemm_einsum_05/journal.bin.xz").read()
rows = np.frombuffer(raw, dtype=JRNL, offset=104)
waited = rows[rows["wait_bucket"] != 0xFFFFFFFF]      # 0xFFFFFFFF：没有等待
print(len(rows), "instructions,", int(waited["wait_cycles"].sum()), "warp-cycles waited")
```

## 这些记录是怎么产生的

每个 kernel 用它的基线 capture 仿真一次，并打开发射日志与等待账本。打包前逐例核对过：运行结束且所有指令都退出，`result.bin` 与基线结果逐字节相同，发射日志恰好每条发射指令一行，每个压缩后的日志解压得到的大小与 SHA-256 与压缩前记录的一致。

## 许可证

MIT，见 [LICENSE](LICENSE)。

[CycleScope]: https://github.com/CycleScope-project/CycleScope
