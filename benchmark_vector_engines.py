#!/usr/bin/env python3
import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import tracemalloc
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Set, Tuple

import numpy as np

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

def mb(nbytes: float) -> float:
    """Convert bytes to MiB."""
    return nbytes / (1024 * 1024)


# Dataset loading
def read_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def beir_folder(dataset: str, root: Path) -> Path:
    """Return the folder holding `corpus.jsonl`, downloading the BEIR dataset if needed."""
    folder = root / dataset
    if (folder / "corpus.jsonl").exists():
        return folder

    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"{dataset}.zip"
    url = BEIR_URL.format(dataset)
    print(f"[*] Downloading {url} ...")
    with urllib.request.urlopen(url) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return folder


def load_beir_texts(
    dataset: str, root: Path, n_vectors: int, n_queries: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Return (passage_texts, query_texts) from a BEIR dataset, shuffled then capped.

    BEIR ships `queries.jsonl` separately from `corpus.jsonl`, so a query is never a passage we
    indexed - a literal self-match would hand every engine a free exact hit and hide real gaps.
    """
    folder = beir_folder(dataset, root)
    corpus = read_jsonl(folder / "corpus.jsonl")
    queries = read_jsonl(folder / "queries.jsonl")

    rng = np.random.default_rng(seed)
    rng.shuffle(corpus)
    rng.shuffle(queries)

    passages = [f"{doc.get('title', '')} {doc['text']}".strip() for doc in corpus[:n_vectors]]
    asks = [query["text"] for query in queries[:n_queries]]
    print(f"[*] BEIR {dataset}: {len(passages):,} passages, {len(asks):,} queries")
    return passages, asks


def fingerprint(array: np.ndarray) -> str:
    """Short hash of a vector matrix, so each run states which matrix it used."""
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()[:10]


def embed_ollama(model: str, texts: Sequence[str], cache: Path, batch: int = 128) -> np.ndarray:
    """Embed texts through Ollama's /api/embed, caching to .npy so re-runs are instant."""
    if cache.exists():
        cached = np.load(cache)
        if cached.shape[0] == len(texts):
            print(f"[*] Reusing {cache.name} {cached.shape} sha1={fingerprint(cached)}")
            return cached
        print(
            f"[!] {cache.name} holds {cached.shape[0]:,} rows but {len(texts):,} were selected; "
            "re-embedding. Delete the file to start clean."
        )

    print(f"[*] Embedding {len(texts):,} texts with {model} (this is the slow part) ...")
    vectors: List[List[float]] = []
    for start in range(0, len(texts), batch):
        payload = json.dumps({"model": model, "input": list(texts[start : start + batch])}).encode()
        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request) as response:
            vectors.extend(json.loads(response.read())["embeddings"])
        print(f"    {min(start + batch, len(texts)):,}/{len(texts):,}", flush=True)

    array = np.asarray(vectors, dtype=np.float32)
    np.save(cache, array)
    print(f"[*] Cached {cache.name} {array.shape} sha1={fingerprint(array)}")
    return array


def l2_normalize(array: np.ndarray) -> np.ndarray:
    """Unit-normalize rows, so a dot product is cosine similarity."""
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32)


def compute_ground_truth(queries: np.ndarray, database: np.ndarray, k: int = 10) -> List[Set[int]]:
    """Exact brute-force cosine ground truth (dot product of unit vectors)."""
    scores = queries @ database.T
    return [set(np.argsort(-row)[:k].tolist()) for row in scores]


def calculate_recall_at_k(predicted: List[List[int]], ground_truth: List[Set[int]], k: int = 10) -> float:
    """Fraction of the true nearest neighbors that were retrieved."""
    if not predicted or not ground_truth:
        return 0.0
    return float(np.mean([len(set(p).intersection(g)) / k for p, g in zip(predicted, ground_truth)])) * 100.0


def measure_latencies_and_ids(
    search_fn: Callable[[np.ndarray, int], List[int]],
    queries: np.ndarray,
    k: int = 10,
    warmup: bool = True,
    runs: int = 1,
) -> Tuple[float, float, float, float, float, List[List[int]]]:
    """(p50_ms, p95_ms, p99_ms, qps, p50_stdev_ms, predicted_ids) over `runs` repeats.

    Percentiles are medians across runs; the stdev is the spread of the per-run p50, so two engines
    can be compared against their own noise. A full untimed pass runs first - warming up on a
    handful of queries leaves first-pass page faults inside run 1, which measured as a ~25% p50
    spread on an otherwise deterministic engine.
    """
    if warmup:
        for q in queries:
            search_fn(q, k)

    run_p50: List[float] = []
    run_p95: List[float] = []
    run_p99: List[float] = []
    run_qps: List[float] = []
    predicted_ids: List[List[int]] = []

    for _ in range(max(1, runs)):
        latencies_us: List[float] = []
        predicted_ids = []
        start_total = time.perf_counter_ns()
        for q in queries:
            t0 = time.perf_counter_ns()
            ids = search_fn(q, k)
            t1 = time.perf_counter_ns()
            latencies_us.append((t1 - t0) / 1000.0)
            predicted_ids.append(ids)
        end_total = time.perf_counter_ns()

        total_sec = (end_total - start_total) / 1e9
        run_qps.append(len(queries) / total_sec if total_sec > 0 else 0.0)
        run_p50.append(float(np.percentile(latencies_us, 50)) / 1000.0)
        run_p95.append(float(np.percentile(latencies_us, 95)) / 1000.0)
        run_p99.append(float(np.percentile(latencies_us, 99)) / 1000.0)

    stdev = float(np.std(run_p50, ddof=1)) if len(run_p50) > 1 else 0.0
    return (
        float(np.median(run_p50)),
        float(np.median(run_p95)),
        float(np.median(run_p99)),
        float(np.median(run_qps)),
        stdev,
        predicted_ids,
    )


@dataclass
class Built:
    """What a builder hands back to `run_engine`."""

    search_fn: Callable[[np.ndarray, int], List[int]]
    ram_mb: float
    ram_method: str
    build_s: float
    train_s: float = 0.0
    exact: bool = False
    notes: str = ""


@dataclass
class BenchResult:
    """One row of the report. The defaults are the skipped-row shape."""

    engine: str
    ram_mb: float = 0.0
    ram_method: str = "-"
    compression: float = 0.0
    py_heap_mb: float = 0.0
    recall_k: float = 0.0
    build_s: float = 0.0
    train_s: float = 0.0
    build_stdev_s: float = 0.0
    p50_ms: float = 0.0
    p50_stdev_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    qps: float = 0.0
    exact: bool = False
    queries: int = 0
    runs: int = 0
    builds: int = 0
    notes: str = ""
    skipped: str = ""


def run_engine(
    name: str,
    builder: Callable[[np.ndarray], Built],
    vecs: np.ndarray,
    queries: np.ndarray,
    ground_truth: List[Set[int]],
    k: int,
    exact_queries: int,
    runs: int,
    builds: int,
) -> BenchResult:
    """Build and measure one engine. A missing dependency or a builder failure becomes a skipped
    row instead of aborting the comparison."""
    raw_fp32_mb = mb(vecs.shape[0] * vecs.shape[1] * 4)

    try:
        built: Optional[Built] = None
        build_times: List[float] = []
        peaks: List[int] = []
        for _ in range(max(1, builds)):
            # Drop the previous index before the next build and time each in its own tracemalloc
            # session - otherwise the outgoing store is still alive and Py Heap counts two indexes.
            built = None
            gc.collect()
            tracemalloc.start()
            try:
                built = builder(vecs)
                build_times.append(built.build_s)
            finally:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            peaks.append(peak)
    except ImportError as exc:  # optional dependency absent
        return BenchResult(engine=name, skipped=str(exc))
    except Exception as exc:  # one bad engine must not kill the whole run
        return BenchResult(engine=name, skipped=f"{type(exc).__name__}: {exc}")
    assert built is not None  # --builds is validated positive, so at least one build ran

    # Exact engines brute-force the same vectors the ground truth came from, so 10 probes prove
    # the id plumbing works; the full query set costs minutes and cannot change the answer.
    probes = queries[:exact_queries] if built.exact else queries
    p50, p95, p99, qps, p50_stdev, pred_ids = measure_latencies_and_ids(
        built.search_fn, probes, k, runs=runs
    )

    return BenchResult(
        engine=name,
        ram_mb=built.ram_mb,
        ram_method=built.ram_method,
        compression=raw_fp32_mb / built.ram_mb if built.ram_mb > 0 else 0.0,
        py_heap_mb=max(peaks) / (1024 * 1024),
        recall_k=calculate_recall_at_k(pred_ids, ground_truth[: len(probes)], k),
        build_s=float(np.median(build_times)),
        train_s=built.train_s,
        build_stdev_s=float(np.std(build_times, ddof=1)) if len(build_times) > 1 else 0.0,
        p50_ms=p50,
        p50_stdev_ms=p50_stdev,
        p95_ms=p95,
        p99_ms=p99,
        qps=qps,
        exact=built.exact,
        queries=len(probes),
        runs=max(1, runs),
        builds=len(build_times),
        notes=built.notes,
    )


# Engine builders. Each returns a `Built`; `run_engine` owns timing, memory and recall, and any
# exception (usually a missing optional dependency) becomes a skipped row.
def build_raw_turbovec(vecs: np.ndarray, bit_width: int) -> Built:
    from turbovec import TurboQuantIndex

    dim = vecs.shape[1]
    t0 = time.perf_counter()
    idx = TurboQuantIndex(dim=dim, bit_width=bit_width)
    idx.add(vecs)
    idx.prepare()
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        _, ids = idx.search(q[None, :], k=top_k)
        return [int(x) for x in np.asarray(ids).ravel()[:top_k]]

    # The index knows its own serialized size, so no bytes-per-vector formula.
    return Built(search_call, mb(len(idx.to_bytes())), "exact (len(index.to_bytes()))", ingest_s)


def build_faiss_flat(vecs: np.ndarray) -> Built:
    import faiss

    n, dim = vecs.shape
    t0 = time.perf_counter()
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        _, ids = index.search(q[None, :], top_k)
        return [int(x) for x in ids[0]]

    return Built(
        search_call,
        mb(n * dim * 4),
        "analytical (n*dim*4 bytes, exact flat layout)",
        ingest_s,
        exact=True,
    )


def build_faiss_pq(vecs: np.ndarray) -> Built:
    """Product quantization at TurboVec's 4-bit rate: m = dim sub-quantizers of 4 bits each."""
    import faiss

    n, dim = vecs.shape
    if dim % 32:
        # FastScan needs m a multiple of 32, and m == dim keeps the bit rate identical.
        raise ValueError(f"IndexPQFastScan needs dim to be a multiple of 32, got {dim}")
    # k-means wants ~39 points per centroid; below that FAISS warns once per sub-quantizer
    # (1536 lines at dim=1536) and the codebooks are not worth measuring.
    min_train = 39 * 16  # nbits=4
    if n < min_train:
        raise ValueError(
            f"PQ training needs >= {min_train} vectors for {dim} sub-quantizers, got {n}; "
            "drop --check or raise --n"
        )

    t0 = time.perf_counter()
    index = faiss.IndexPQFastScan(dim, dim, 4)
    t_train = time.perf_counter()
    index.train(vecs)  # PQ is trainable: it fits its codebooks to the dataset
    train_s = time.perf_counter() - t_train
    index.add(vecs)
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        _, ids = index.search(q[None, :], top_k)
        return [int(x) for x in ids[0]]

    return Built(
        search_call,
        mb(n * dim // 2),
        "analytical (m=dim x 4-bit codes + codebooks)",
        ingest_s,
        train_s=train_s,
        notes="approximate: codebooks are trained on the dataset",
    )


def build_hnsw(vecs: np.ndarray) -> Built:
    import hnswlib

    n, dim = vecs.shape
    neighbours = 16  # hnswlib M
    t0 = time.perf_counter()
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, ef_construction=200, M=neighbours)
    index.add_items(vecs, np.arange(n))
    # hnswlib defaults ef_search to 10, truncating candidates to ~k and tanking recall for
    # reasons unrelated to HNSW; 64 is the usual fair start.
    index.set_ef(64)
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        ids, _ = index.knn_query(q, k=top_k)
        return [int(x) for x in ids[0]]

    # One stored FP32 vector plus a bidirectional adjacency list of M*2 int32 links.
    return Built(
        search_call,
        mb(n * (dim * 4 + neighbours * 2 * 4)),
        f"analytical (n * (dim*4 + {neighbours * 2}*4) bytes)",
        ingest_s,
        notes="graph adds ~8% over raw FP32 vectors; ef_search=64",
    )


class DummyEmbeddings:
    """Placeholder embedder; the harness injects the vectors itself.

    `InMemoryVectorStore` requires one, and `add_documents` does run it - but the output is
    discarded and overwritten with `vecs`. That baseline's `Py Heap` is dominated by the store
    holding `list[float]` vectors, not by this stub's rows.
    """

    def __init__(self, dim: int):
        self.dim = dim
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * self.dim for _ in texts]
    def embed_query(self, text: str) -> List[float]:
        return [0.0] * self.dim


def build_langchain_baseline(vecs: np.ndarray) -> Built:
    from langchain_core.documents import Document
    from langchain_core.vectorstores.in_memory import InMemoryVectorStore

    n, dim = vecs.shape
    t0 = time.perf_counter()

    store = InMemoryVectorStore(embedding=DummyEmbeddings(dim))
    docs = [Document(page_content=f"doc_{i}", id=str(i)) for i in range(n)]
    store.add_documents(docs, ids=[str(i) for i in range(n)])

    for i in range(n):
        store.store[str(i)]["vector"] = vecs[i].tolist()

    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        results = store.similarity_search_by_vector(q.tolist(), k=top_k)
        return [int(doc.id) for doc in results]

    return Built(
        search_call,
        mb(n * dim * 4),
        "analytical (n*dim*4 bytes; Python object overhead is in Py Heap)",
        ingest_s,
        exact=True,
        notes="brute force in interpreted Python; vectors are stored as float lists",
    )


def build_langchain_turbovec(vecs: np.ndarray) -> Built:
    from turbovec import IdMapIndex
    from turbovec.langchain import TurboQuantVectorStore

    n, dim = vecs.shape
    index = IdMapIndex(dim=dim, bit_width=4)
    store = TurboQuantVectorStore(
        embedding=DummyEmbeddings(dim),
        index=index,
    )

    t0 = time.perf_counter()
    # `_store_texts_and_vectors` is private - there is no public ingest path for
    # precomputed vectors (add_texts re-embeds), and an embedding shim would mis-feed if LangChain
    # batched its calls. Upgrade path: implement VectorStore.add_embeddings, then swap this call.
    store._store_texts_and_vectors(
        texts_list=[f"doc_{i}" for i in range(n)],
        vectors=vecs,
        metadatas=[{} for _ in range(n)],
        ids=[str(i) for i in range(n)],
    )
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        results = store.similarity_search_by_vector(q, k=top_k)
        return [int(doc.id) for doc in results]

    return Built(
        search_call,
        mb(len(index.to_bytes())),
        "exact (len(index.to_bytes())); text/metadata side-car is in Py Heap",
        ingest_s,
    )

 
def build_llamaindex_baseline(vecs: np.ndarray) -> Built:
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores import SimpleVectorStore
    from llama_index.core.vector_stores.types import VectorStoreQuery

    n, dim = vecs.shape
    t0 = time.perf_counter()

    store = SimpleVectorStore()
    nodes = [TextNode(text=f"doc_{i}", id_=str(i), embedding=vecs[i].tolist()) for i in range(n)]
    store.add(nodes)

    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        query = VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=top_k)
        res = store.query(query)
        ids = res.ids if res.ids is not None else [node.id_ for node in (res.nodes or [])]
        return [int(node_id) for node_id in ids]

    return Built(
        search_call,
        mb(n * dim * 4),
        "analytical (n*dim*4 bytes; Python object overhead is in Py Heap)",
        ingest_s,
        exact=True,
        notes="brute force in interpreted Python; TextNode metadata is extra",
    )


def build_llamaindex_turbovec(vecs: np.ndarray) -> Built:
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery
    from turbovec import IdMapIndex
    from turbovec.llama_index import TurboQuantVectorStore

    n, dim = vecs.shape
    index = IdMapIndex(dim=dim, bit_width=4)
    store = TurboQuantVectorStore(index=index)

    t0 = time.perf_counter()
    store.add([TextNode(text=f"doc_{i}", id_=str(i), embedding=vecs[i].tolist()) for i in range(n)])
    ingest_s = time.perf_counter() - t0

    def search_call(q, top_k):
        query = VectorStoreQuery(query_embedding=q.tolist(), similarity_top_k=top_k)
        res = store.query(query)
        ids = res.ids if res.ids is not None else [node.id_ for node in (res.nodes or [])]
        return [int(node_id) for node_id in ids]

    return Built(
        search_call,
        mb(len(index.to_bytes())),
        "exact (len(index.to_bytes()))",
        ingest_s,
    )

 
# Formatting & Presentation
def derived_notes(results: Sequence[BenchResult], k: int) -> List[str]:
    """Summary lines computed from the measurements, never hardcoded."""
    live = [r for r in results if not r.skipped]
    turbo = [r for r in live if "TurboVec" in r.engine]
    frameworks = [r for r in live if r.exact and "TurboVec" not in r.engine]
    notes: List[str] = []

    if turbo and frameworks:
        slow = max(r.p50_ms for r in frameworks)
        fast = min(r.p50_ms for r in turbo)
        notes.append(
            f"- Measured p50 spread on this run: slowest FP32 framework {slow:,.1f} ms vs fastest "
            f"TurboVec {fast:.3f} ms ({slow / fast:,.0f}x)."
        )
    if turbo:
        notes.append(
            f"- Measured TurboVec Recall@{k}: {min(r.recall_k for r in turbo):.1f}% to "
            f"{max(r.recall_k for r in turbo):.1f}% against exact FP32 ground truth."
        )
    # Are the two closest approximate timings separable given their own run-to-run spread?
    timed = [r for r in live if not r.exact and r.runs > 1]
    if len(timed) > 1:
        gap, a, b = min(
            ((abs(x.p50_ms - y.p50_ms), x, y) for i, x in enumerate(timed) for y in timed[i + 1 :]),
            key=lambda item: item[0],
        )
        slack = max(a.p50_stdev_ms, b.p50_stdev_ms)
        verdict = "the gap holds" if gap >= slack else "not separable at this run count"
        notes.append(
            f"- Closest timings: {a.engine} ({a.p50_ms:,.3f} ms) and {b.engine} ({b.p50_ms:,.3f} ms) "
            f"differ by {gap:,.3f} ms against a spread of {slack:,.3f} ms - {verdict}."
        )

    capped = {r.queries for r in live if r.exact}
    if capped:
        notes.append(
            f"- Exact engines were probed with {max(capped)} queries: brute force over these same "
            "vectors is the ground truth, so more queries cannot change the answer."
        )
    skipped = [r.engine for r in results if r.skipped]
    if skipped:
        notes.append(f"- Skipped (missing dependency or engine error): {', '.join(skipped)}.")
    return notes


def format_table(
    results: Sequence[BenchResult], n: int, dim: int, k: int, source: str
) -> str:
    runs = next((r.runs for r in results if not r.skipped), 1)
    builds = next((r.builds for r in results if not r.skipped), 1)
    headers = [
        "Vector Store Engine",
        "Index (MB)",
        "Compression",
        "Py Heap (MB)",
        f"Recall@{k}",
        "Build (s)",
        "Train (s)",
        "p50 (ms)",
        "p95 (ms)",
        "p99 (ms)",
        "QPS",
    ]
    lines = [
        f"### Vector Store Benchmark Results: {source}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:"] * (len(headers) - 1)) + " |",
    ]

    for r in results:
        if r.skipped:
            lines.append(f"| {r.engine} | SKIPPED: {r.skipped} | - | - | - | - | - | - | - | - | - |")
            continue
        marker = " *" if r.exact else ""
        train_cell = f"{r.train_s:,.2f}" if r.train_s > 0 else "-"
        build_cell = f"{r.build_s:,.2f}"
        if r.builds > 1:
            build_cell += f" ± {r.build_stdev_s:.2f}"
        p50_cell = f"{r.p50_ms:,.3f}"
        if r.runs > 1:
            p50_cell += f" ± {r.p50_stdev_ms:.3f}"
        lines.append(
            f"| {r.engine}{marker} | {r.ram_mb:,.1f} | {r.compression:,.1f}x | "
            f"{r.py_heap_mb:,.1f} | {r.recall_k:.1f}% | {build_cell} | {train_cell} | "
            f"{p50_cell} | {r.p95_ms:,.3f} | {r.p99_ms:,.3f} | {r.qps:,.0f} |"
        )

    lines += ["", "How `Index (MB)` was obtained, per row:"]
    lines += [
        f"- {r.engine}: {r.ram_method}" + (f" - {r.notes}" if r.notes else "")
        for r in results
        if not r.skipped
    ]
    lines += [
        "",
        "Notes:",
        f"- Timings are the median of {runs} runs and {builds} builds per engine; `p50` and "
        "`Build` carry the stdev across them.",
        f"- Compression is raw FP32 payload ({mb(n * dim * 4):,.1f} MB = N*dim*4) divided by "
        "`Index (MB)`; 1.0x means the engine stores the vectors uncompressed.",
        f"- Recall@{k} is evaluated against exact brute-force Float32 cosine ground truth (Q @ D.T).",
        "- `Build (s)` is time until the index can answer a query and `Train (s)` is the part of it "
        "spent fitting codebooks. They are not comparable across engines: FAISS flat copies floats, "
        "PQ trains codebooks, TurboVec quantizes, HNSW builds a graph.",
        "- `Py Heap (MB)` is the Python-side `tracemalloc` peak over the whole build, measured the "
        "same way for every engine. It tracks framework object overhead, not index size, and for the "
        "two FP32 baselines it includes the rows returned by the embedding stub.",
        "- `*` marks engines that are exact by construction (brute force over the same vectors the "
        "ground truth came from), so their recall must be 100% if the plumbing is correct.",
    ]
    lines += derived_notes(results, k)
    lines += [""]
    return "\n".join(lines)


def check_results(results: Sequence[BenchResult]) -> List[str]:
    """Return every invariant violation found in `results`; an empty list means a healthy run."""
    problems: List[str] = []
    names = [r.engine for r in results]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"duplicate engine name: {name}")
    if not results:
        problems.append("no engines were run")
    elif all(r.skipped for r in results):
        problems.append("every engine was skipped - nothing was measured")

    for r in results:
        if r.skipped:
            continue
        if not 0.0 < r.recall_k <= 100.0:
            problems.append(f"{r.engine}: recall {r.recall_k} is outside (0, 100]")
        if r.qps <= 0.0:
            problems.append(f"{r.engine}: QPS is not positive")
        if r.ram_mb <= 0.0:
            problems.append(f"{r.engine}: index size is not positive")
        if r.queries <= 0:
            problems.append(f"{r.engine}: ran no queries")
        if not r.p50_ms <= r.p95_ms <= r.p99_ms:
            problems.append(f"{r.engine}: p50/p95/p99 are out of order")
        if r.exact and r.recall_k < 100.0 - 1e-6:
            problems.append(f"{r.engine}: exact engine recall {r.recall_k:.4f}% is not 100%")
    return problems


def parse_bit_widths(text: str) -> List[int]:
    widths: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if part not in ("2", "3", "4"):
            raise argparse.ArgumentTypeError(f"bit width must be 2, 3 or 4, got {part!r}")
        if int(part) not in widths:
            widths.append(int(part))
    if not widths:
        raise argparse.ArgumentTypeError("at least one bit width is required")
    return widths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vector stores on memory, ingestion speed, latency and recall.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10000, help="Passages to index (capped by the corpus)")
    parser.add_argument("--queries", type=int, default=100, help="Number of test queries")
    parser.add_argument("--k", type=int, default=10, help="Top-k nearest neighbors")
    parser.add_argument(
        "--bit-widths",
        type=parse_bit_widths,
        default=[4, 2],
        help="Comma-separated turbovec quantization bit widths",
    )
    parser.add_argument(
        "--exact-queries",
        type=int,
        default=10,
        help="Queries used to probe engines that are exact by construction",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling/sampling")
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Times to repeat the query set per engine; p50 reports the spread across runs",
    )
    parser.add_argument(
        "--builds",
        type=int,
        default=1,
        help="Times to rebuild each index; Build reports the spread across builds",
    )
    parser.add_argument("--beir-dataset", default="scidocs", help="BEIR dataset name")
    parser.add_argument(
        "--beir-dir",
        default=str(Path(__file__).resolve().parent / "data"),
        help="Where the dataset and cached embeddings live",
    )
    parser.add_argument(
        "--embed-model", default="qwen3-embedding:0.6b", help="Ollama embedding model"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Small N=500 run on the same path, with result invariants asserted",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    n = 500 if args.check else args.n
    n_queries = min(20 if args.check else args.queries, n)
    k = args.k

    if n <= 0 or k <= 0 or args.exact_queries <= 0 or args.runs <= 0 or args.builds <= 0:
        print("n, k, runs, builds and exact-queries must all be positive", file=sys.stderr)
        return 2

    data_dir = Path(args.beir_dir)
    passages, asks = load_beir_texts(args.beir_dataset, data_dir, n, n_queries, args.seed)
    tag = f"{args.beir_dataset}-{args.embed_model.replace(':', '-')}"
    vecs = l2_normalize(
        embed_ollama(args.embed_model, passages, data_dir / f"{tag}-docs.npy")
    )
    queries = l2_normalize(
        embed_ollama(args.embed_model, asks, data_dir / f"{tag}-queries.npy")
    )
    n, dim = vecs.shape[0], vecs.shape[1]
    n_queries = queries.shape[0]
    source = f"BEIR {args.beir_dataset} - {args.embed_model} - {n:,} passages - d={dim}, k={k}"

    if k > n:
        print(f"k={k} exceeds the {n} indexed vectors", file=sys.stderr)
        return 2

    print(
        f"[*] Dataset: {source}; Queries={n_queries}; "
        f"raw FP32 payload = {mb(n * dim * 4):,.2f} MB"
    )

    print(f"[*] Computing exact Float32 ground truth for Recall@{k} ...")
    ground_truth = compute_ground_truth(queries, vecs, k=k)

    engines: List[Tuple[str, Callable[[np.ndarray], Built]]] = [
        *[
            (f"Raw TurboVec ({bw}-bit)", lambda v, bw=bw: build_raw_turbovec(v, bw))
            for bw in args.bit_widths
        ],
        ("FAISS IndexFlatIP (FP32)", build_faiss_flat),
        ("FAISS IndexPQFastScan (4-bit)", build_faiss_pq),
        ("HNSW (hnswlib FP32)", build_hnsw),
        ("LangChain InMemory (FP32)", build_langchain_baseline),
        ("LangChain + TurboVec (4-bit)", build_langchain_turbovec),
        ("LlamaIndex Simple (FP32)", build_llamaindex_baseline),
        ("LlamaIndex + TurboVec (4-bit)", build_llamaindex_turbovec),
    ]

    results: List[BenchResult] = []
    for name, builder in engines:
        print(f"[*] {name} ...")
        result = run_engine(
            name, builder, vecs, queries, ground_truth, k, args.exact_queries, args.runs, args.builds
        )
        results.append(result)
        if result.skipped:
            print(f"    skipped: {result.skipped}")

    print()
    print(format_table(results, n, dim, k, source))

    if args.check:
        problems = check_results(results)
        if problems:
            print("self-check FAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  - {problem}", file=sys.stderr)
            return 1
        print(f"self-check passed: {len(results)} engines, all invariants hold")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
