"""A synthetic corpus with exact, known relevance structure.

Why synthetic rather than a public benchmark: this repo measures *index
internals*, and for that the ground truth must be exact and cheap to recompute.
Every recall number here is measured against an exhaustive flat scan of the
same vectors, so "recall" means precisely "what fraction of the true nearest
neighbours did the approximate index return" -- not "what fraction of some
third party's human labels". That makes the recall numbers portable: they are
a property of the vectors and the index, and reproduce bit-for-bit on any
machine.

The corpus carries structured metadata (``category``, ``year``) because the
filtered-ANN failure in :mod:`.filtering` needs filters of *known, tunable*
selectivity. A filter that keeps 0.5% of the corpus is the whole point; you
cannot arrange that reliably on a corpus you did not build.

Embeddings come from ``all-MiniLM-L6-v2`` when it is available and are cached
to a gitignored directory. When it is not, a deterministic hashed-bag-of-words
embedder stands in. The fallback is not a toy: it produces genuinely clustered,
anisotropic vectors, which is what makes ANN recall interesting. Tests use it
exclusively so that ``pytest`` needs no model download and no network.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Topic vocabulary. Distinct enough that documents on different topics are
# genuinely far apart in embedding space, which is what gives the ANN graph
# real cluster structure to navigate rather than uniform noise.
_TOPICS: dict[str, list[str]] = {
    "storage": ["write-ahead log", "page cache", "compaction", "durability", "checkpoint",
                "replication lag", "fsync", "tablespace", "vacuum", "bloom filter"],
    "networking": ["congestion window", "packet loss", "handshake", "round trip",
                   "load balancer", "keepalive", "backpressure", "multiplexing",
                   "retransmission", "jumbo frame"],
    "retrieval": ["inverted index", "term frequency", "posting list", "recall curve",
                  "nearest neighbour", "quantisation", "embedding drift", "re-ranking",
                  "candidate set", "query latency"],
    "scheduling": ["work stealing", "priority inversion", "context switch", "time slice",
                   "starvation", "affinity mask", "run queue", "preemption",
                   "deadline miss", "thread pool"],
    "security": ["certificate chain", "key rotation", "replay attack", "constant time",
                 "privilege escalation", "audit log", "token expiry", "side channel",
                 "sandbox escape", "nonce reuse"],
}

_FRAMES = [
    "The {a} interacts with the {b} whenever {c} is contended.",
    "Tuning the {a} without accounting for the {b} makes {c} worse under load.",
    "A postmortem traced the incident to the {a}, which had masked the {b}.",
    "Operators usually watch the {a}, but the leading indicator is the {b}.",
    "Under saturation the {a} degrades first, then the {b} follows.",
    "Documentation claims the {a} is independent of the {b}; measurement disagrees.",
]

_DETAILS = [
    "Reproduced on a three-node cluster.",
    "Only visible above the p99 threshold.",
    "The regression bisected to a config change.",
    "Mitigated by halving the batch size.",
    "Confirmed with a synthetic workload replay.",
    "Escalated after the second recurrence.",
    "Root cause remained unconfirmed.",
    "A rollback restored baseline behaviour.",
    "The metric had been sampled too coarsely.",
    "Load shedding hid the symptom for a week.",
    "Correlated with a dependency upgrade.",
    "Observed only on the read replicas.",
]

CATEGORIES = tuple(_TOPICS)
YEARS = (2019, 2020, 2021, 2022, 2023, 2024)


@dataclass(frozen=True)
class Document:
    """One corpus passage and its metadata."""

    doc_id: int
    text: str
    category: str
    year: int


def build_corpus(n_docs: int = 20_000, *, seed: int = 0) -> list[Document]:
    """Generate ``n_docs`` passages deterministically.

    Category is assigned round-robin rather than sampled so that the class
    balance is exact and does not wobble with ``n_docs``. Filter selectivity in
    :mod:`.filtering` is computed from these counts, and a filter whose true
    selectivity drifts between runs would make the collapse magnitude
    unreproducible.
    """
    if n_docs <= 0:
        raise ValueError(f"n_docs must be positive, got {n_docs}")

    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    docs: list[Document] = []

    for i in range(n_docs):
        category = cats[i % len(cats)]
        vocab = _TOPICS[category]
        # A minority of documents borrow a term from a neighbouring category.
        # Perfectly separated clusters would make every ANN index trivially
        # perfect; the leakage is what creates hard queries near boundaries.
        other = cats[(i // len(cats)) % len(cats)]
        pool = vocab + (_TOPICS[other][:3] if i % 7 == 0 else [])

        a, b, c = rng.choice(pool, size=3, replace=False)
        frame = _FRAMES[i % len(_FRAMES)]
        text = frame.format(a=a, b=b, c=c)
        text += f" Observed in the {category} subsystem during {YEARS[i % len(YEARS)]}."

        # Every document ends with a unique detail clause. Without it the
        # frame-times-vocabulary space is far smaller than n_docs and the
        # corpus contains thousands of *exact* duplicates -- which put
        # identical vectors in the index, make the true top-k arbitrary among
        # tied neighbours, and pin measured recall below 1.0 no matter how
        # large ef_search grows. That ceiling is an artefact of the corpus, not
        # a property of HNSW, and it would have silently corrupted every recall
        # number in this repo. The clause also gives BM25 discriminative terms.
        detail = rng.choice(_DETAILS)
        docs.append(Document(doc_id=i, text=f"{text} {detail} Case {i:06d}.",
                             category=category, year=YEARS[i % len(YEARS)]))
    return docs


def build_queries(docs: list[Document], n_queries: int = 200, *, seed: int = 1) -> list[str]:
    """Queries phrased as questions over corpus terminology.

    Deliberately *not* verbatim document text. A query that is a copy of a
    document has a trivial nearest neighbour at distance zero, and every index
    finds it; recall would be pinned at 1.0 and the sweep would show nothing.
    These recombine terms across frames so the true neighbour set is a genuine
    contest between several similar passages.
    """
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    out: list[str] = []
    for i in range(n_queries):
        category = cats[i % len(cats)]
        vocab = _TOPICS[category]
        a, b = rng.choice(vocab, size=2, replace=False)
        out.append(f"why does the {a} affect the {b} in production")
    return out


# --- embedding ---------------------------------------------------------


def _hashed_embed(texts: list[str], dim: int = 384) -> np.ndarray:
    """Deterministic bag-of-words embedding, used when MiniLM is unavailable.

    Each token is hashed to a stable coordinate and accumulated with a signed
    weight. This is the classic hashing trick: it has no semantics, but it does
    produce clustered vectors (documents sharing vocabulary land near each
    other), which is the only property the index sweep actually needs.

    Stable across machines and Python builds because it hashes with blake2b
    rather than ``hash()``, whose salt varies per process.
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            out[row, value % dim] += 1.0 if (value >> 63) & 1 else -1.0
    return out


def embed(texts: list[str], *, use_model: bool = True) -> np.ndarray:
    """Embed and L2-normalise. Returns ``float32`` of shape ``(len(texts), dim)``.

    Normalised so that inner product equals cosine similarity, which lets the
    flat ground-truth index and every ANN index agree on what "nearest" means.
    A recall comparison between indexes using different metrics is meaningless.
    """
    if not texts:
        raise ValueError("cannot embed an empty list")

    vectors: np.ndarray | None = None
    if use_model and os.environ.get("RIT_FORCE_FALLBACK_EMBED") != "1":
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME)
            vectors = np.asarray(
                model.encode(texts, batch_size=128, show_progress_bar=False),
                dtype=np.float32,
            )
        except Exception:
            # Any failure -- no network, no package, no disk -- falls back
            # rather than aborting. The fallback is documented in the results
            # header so a reader always knows which embedder produced a number.
            vectors = None

    if vectors is None:
        vectors = _hashed_embed(texts)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


def embedder_name(*, use_model: bool = True) -> str:
    """Which embedder :func:`embed` would actually use, for results provenance."""
    if not use_model or os.environ.get("RIT_FORCE_FALLBACK_EMBED") == "1":
        return "hashed-bow-384 (fallback)"
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(MODEL_NAME)
        return MODEL_NAME
    except Exception:
        return "hashed-bow-384 (fallback)"


def cached_embeddings(
    docs: list[Document],
    queries: list[str],
    *,
    cache_dir: Path,
    use_model: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed corpus and queries, caching to ``cache_dir``.

    The cache key includes the corpus size and the embedder name, so switching
    embedders or corpus size cannot silently reuse stale vectors -- a mistake
    that would produce recall numbers for a corpus that no longer exists.
    """
    name = embedder_name(use_model=use_model)
    key = hashlib.blake2b(
        f"{name}|{len(docs)}|{len(queries)}".encode(), digest_size=8
    ).hexdigest()
    path = cache_dir / f"emb-{key}.npz"

    if path.exists():
        blob = np.load(path)
        return blob["docs"], blob["queries"]

    doc_vecs = embed([d.text for d in docs], use_model=use_model)
    query_vecs = embed(queries, use_model=use_model)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, docs=doc_vecs, queries=query_vecs)
    return doc_vecs, query_vecs
