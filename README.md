# Vector Engines Benchmark

One script, nine vector search configurations, identical vectors: index size, build cost, query
latency distribution and Recall@10 against exact FP32 brute force.

Engines compared: Raw TurboVec (4-bit), Raw TurboVec (2-bit), FAISS IndexFlatIP (FP32), FAISS
IndexPQFastScan (4-bit), HNSW (hnswlib FP32), LangChain InMemory (FP32), LangChain + TurboVec
(4-bit), LlamaIndex Simple (FP32), LlamaIndex + TurboVec (4-bit)

(4-bit TurboVec index tested raw and again behind LangChain and LlamaIndex)

BEIR `scidocs` passages embedded locally via Ollama using 'qwen3-embedding:0.6b'. Recall is measured
against the **exact FP32 brute-force top-10 of the same vectors**, not BEIR's relevance labels — that
isolates the *index* from the *embedder*, so a gap between two engines is a gap in the index. And
since BEIR ships `queries.jsonl` separately from `corpus.jsonl`, no query is a passage that was
indexed: a self-match would hand every engine a free exact hit and hide real gaps.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen3-embedding:0.6b

python benchmark_vector_engines.py --check                 # small run, asserts result invariants
python benchmark_vector_engines.py --n 25000 --queries 1000 --runs 10 --builds 3
```

The first run downloads BEIR `scidocs` (~30 MB) and embeds 25,000 passages, printing progress; both
matrices are then cached to `data/*.npy`, so every later run starts in seconds.

| Flag | Default | Meaning |
| :--- | :---: | :--- |
| `--beir-dataset` | `scidocs` | Any BEIR dataset — `scifact`, `nfcorpus`, `fiqa`, `trec-covid`, … |
| `--beir-dir` | `data/`, beside the script | Where the corpus and cached embeddings live |
| `--embed-model` | `qwen3-embedding:0.6b` | Any Ollama embedding model |
| `--n` | `10000` | Passages to index (capped by the corpus) |
| `--queries` | `100` | Test queries |
| `--k` | `10` | Top-k returned |
| `--bit-widths` | `4,2` | turbovec quantization bit widths |
| `--exact-queries` | `10` | Probes for engines that are exact by construction |
| `--runs` | `5` | Query-set repetitions per engine; p50 reports the spread across runs |
| `--builds` | `1` | Times to rebuild each index; Build reports the spread across builds |
| `--check` | off | Small N=500 run on the same path, with invariants asserted |

A missing optional dependency becomes a skipped row rather than a crash.

## Results

BEIR `scidocs`, `qwen3-embedding:0.6b`, **N = 25,000 passages, d = 1024, 1,000 queries**, k = 10, 
Raw FP32 payload is 97.7 MB. Verbatim output:
[`result-n25000-d1024-k1000.txt`](result-n25000-d1024-k1000.txt).

| Engine | Index (MB) | Compression | Py Heap (MB) | Recall@10 | Build (s) | Train (s) | p50 (ms) | p95 (ms) | p99 (ms) | QPS |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw TurboVec (4-bit) | 13.4 | 7.3x | 97.8 | 96.7% | 0.08 ± 0.06 | - | 0.822 ± 0.002 | 0.831 | 0.844 | 1,214 |
| Raw TurboVec (2-bit) | 6.7 | 14.5x | 97.7 | 88.8% | 0.05 ± 0.02 | - | 0.373 ± 0.001 | 0.378 | 0.385 | 2,674 |
| FAISS IndexFlatIP (FP32) * | 97.7 | 1.0x | 5.9 | 100.0% | 0.00 ± 0.00 | - | 1.230 ± 0.015 | 1.326 | 1.379 | 802 |
| FAISS IndexPQFastScan (4-bit) | 12.2 | 8.0x | 0.0 | 95.6% | 4.43 ± 0.07 | 4.26 | 0.911 ± 0.004 | 0.924 | 0.934 | 1,096 |
| HNSW (hnswlib FP32) | 100.7 | 1.0x | 0.4 | 99.7% | 14.75 ± 0.12 | - | 1.547 ± 0.004 | 1.816 | 1.896 | 656 |
| LangChain InMemory (FP32) * | 97.7 | 1.0x | 833.6 | 100.0% | 14.73 ± 0.07 | - | 748.101 ± 2.295 | 751.989 | 752.275 | 1 |
| LangChain + TurboVec (4-bit) | 13.6 | 7.2x | 208.1 | 96.7% | 0.40 ± 0.01 | - | 0.860 ± 0.001 | 0.870 | 0.883 | 1,161 |
| LlamaIndex Simple (FP32) * | 97.7 | 1.0x | 866.9 | 100.0% | 23.81 ± 0.14 | - | 854.198 ± 3.199 | 861.059 | 862.128 | 1 |
| LlamaIndex + TurboVec (4-bit) | 13.6 | 7.2x | 1,030.2 | 96.7% | 29.18 ± 0.22 | - | 0.988 ± 0.005 | 0.998 | 1.004 | 1,012 |

`*` marks engines that are exact by construction, so their recall must be 100%. Timings are the
median of 10 runs and 3 builds per engine; `p50` and `Build` carry the standard deviation.

## What the numbers say

- **At a matched 4-bit rate**, TurboVec reaches 96.7% recall@10 against FAISS `IndexPQFastScan`'s
  95.6%, and builds in 0.08 s against 4.43 s — 4.26 s of which is k-means training. On latency the
  two are 0.822 ms against 0.911 ms, a 10% gap at the edge of what this setup resolves: treat them
  as equivalent.
- **TurboVec is not the smallest at that rate.** FAISS PQ reports 12.2 MB against 13.4 MB, because
  TurboQuant pays ~1 MB for its per-coordinate Lloyd-Max codebook. The memory win is against FP32:
  13.4 MB against 97.7 MB, and 7.2x through LangChain or LlamaIndex.
- **The order-of-magnitude latency win is against interpreted Python, not C++.** Through the identical
  LangChain interface, swapping the store takes a query from 748 ms to 0.860 ms, the index from
  97.7 MB to 13.6 MB, and the Python heap from 833.6 MB to 208.1 MB.
- **Against exact brute force**, TurboVec 4-bit is 33% faster at the median (0.822 ms against
  1.230 ms) at 7.3x less memory.
- **HNSW buys +3.0 points of recall and charges for it**: 99.7% against 96.7%, at 7.5x the memory
  (100.7 MB), a slower query (1.547 ms) and a 14.75 s build.
- **2-bit is a real trade, not a free win**: 88.8% recall for 6.7 MB, and the fastest row here at
  0.373 ms / 2,674 QPS.

## Columns

- **Index (MB)** — the engine's own index. Exact for turbovec (`len(index.to_bytes())`), analytical
  for the others; each run prints the method it used per row.
- **Compression** — raw FP32 payload (`N*dim*4`) ÷ `Index (MB)`. The ratio rises with N because the
  codebook is a fixed cost; the asymptotes are 8x at 4-bit and 16x at 2-bit.
- **Build (s)** — time until the index can answer a query. **Train (s)** — the part of it spent
  fitting codebooks (`-` when there is none). They are *not* comparable across engines: FAISS flat
  copies floats, PQ trains codebooks, TurboVec quantizes, HNSW builds a graph.
- **Py Heap (MB)** — Python-side `tracemalloc` peak over a single build, measured identically for
  every engine: framework object overhead, not index size. Near `0.0` for FAISS and HNSW because
  they allocate outside CPython's allocator.

## Caveats

- **The spread shown is within-session.** Re-run on a quiet machine, TurboVec, FAISS flat and the
  two wrapped stores reproduce to ~1%, but HNSW, FAISS PQ and the FP32 baselines have drifted
  10–20% between sessions. Treat engine differences below ~10% as ties.
- **Build times vary more than latency.** FAISS PQ has built in 4.4 s and 7.4 s on two quiet runs
  and 17.8 s on a throttled one: k-means and HNSW graph construction are parallel and depend on how
  many cores are free. Read that column as order of magnitude.
- **A throttled machine inflates everything.** One run of this configuration had every engine ~1.8x
  slower, C++ included, with FAISS flat's stdev going from ±0.015 to ±1.055 ms. Sanity check: FAISS
  `IndexFlatIP` p50 should be ~1.2 ms.
- The **2-bit row has no like-for-like baseline** — FAISS PQ is only run at 4-bit here.
- Exact engines are probed with `--exact-queries` (10) rather than the full 1,000, since brute force
  cannot change its own answer, so their p95/p99 are coarse.
- **Recall measures fidelity to the FP32 oracle for the same embeddings**, not task accuracy against
  BEIR's relevance labels. A stronger embedder would lift the whole table together, not reorder it.

## Credit where it's due

The script was written with AI assistance (DeepSeek V4.1 Flash) under human direction and reviewed
before publishing.

Every number in the results table comes from a real run, not from the model.