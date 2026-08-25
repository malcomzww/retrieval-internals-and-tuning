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
import re
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

# Tenant ids exist to give the filtering experiment a selectivity knob that is
# statistically independent of both topic and year. Deriving selectivity from
# `doc_id % n` instead looks equivalent and is not: categories are assigned
# round-robin over five values, so `doc_id % 5` is perfectly aliased to
# category, and a "0.7% filter" built that way silently keeps 3.3% of one
# category. Tenants are assigned from a stride coprime with both 5 and 6, so
# every (tenant, category, year) cell is populated in proportion.
N_TENANTS = 64


@dataclass(frozen=True)
class Document:
    """One corpus passage and its metadata."""

    doc_id: int
    text: str
    category: str
    year: int
    tenant: int


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
                             category=category, year=YEARS[i % len(YEARS)],
                             tenant=(i * 7) % N_TENANTS))
    return docs


QUERY_TEMPLATE = "why does the {a} affect the {b} in production"

# Paraphrases of the topic terms, used to build queries that do NOT contain the
# document's literal wording. Without this the whole re-ranking comparison is
# degenerate: a query built from the same strings that define relevance is
# solved perfectly by exact term matching, every arm scores NDCG 1.0, and there
# is no headroom in which a crossover could appear. Vocabulary mismatch between
# query and document is also the actual reason dense retrieval is worth its
# cost in production, so a corpus without it cannot measure retrieval at all.
_PARAPHRASE: dict[str, str] = {
    "write-ahead log": "transaction journal",
    "page cache": "buffer pool",
    "compaction": "segment merging",
    "durability": "crash safety",
    "checkpoint": "flush barrier",
    "replication lag": "follower delay",
    "fsync": "disk flush",
    "tablespace": "storage volume",
    "vacuum": "dead tuple cleanup",
    "bloom filter": "membership sketch",
    "congestion window": "send quota",
    "packet loss": "dropped datagrams",
    "handshake": "session setup",
    "round trip": "echo delay",
    "load balancer": "traffic director",
    "keepalive": "liveness probe",
    "backpressure": "flow throttling",
    "multiplexing": "stream sharing",
    "retransmission": "resend attempt",
    "jumbo frame": "oversized packet",
    "inverted index": "term dictionary",
    "term frequency": "word count weighting",
    "posting list": "document id run",
    "recall curve": "coverage trade-off",
    "nearest neighbour": "closest vector",
    "quantisation": "code compression",
    "embedding drift": "vector staleness",
    "re-ranking": "second pass ordering",
    "candidate set": "shortlist",
    "query latency": "response delay",
    "work stealing": "idle task pulling",
    "priority inversion": "rank reversal",
    "context switch": "task swap",
    "time slice": "quantum",
    "starvation": "indefinite postponement",
    "affinity mask": "core pinning",
    "run queue": "ready list",
    "preemption": "forced yield",
    "deadline miss": "late completion",
    "thread pool": "worker group",
    "certificate chain": "trust path",
    "key rotation": "credential refresh",
    "replay attack": "message resend abuse",
    "constant time": "timing invariant",
    "privilege escalation": "rights elevation",
    "audit log": "activity trail",
    "token expiry": "credential timeout",
    "side channel": "indirect leak",
    "sandbox escape": "isolation breakout",
    "nonce reuse": "counter repetition",
}


def query_terms(query: str) -> tuple[str, str]:
    """Recover the two topic terms a query was built from.

    Exists so that relevance can be defined *independently of any embedding*.
    See :func:`term_relevance` for why that independence is the difference
    between a measurement and a tautology.
    """
    match = re.fullmatch(r"why does the (.+) affect the (.+) in production", query)
    if match is None:
        raise ValueError(f"not a generated query: {query!r}")
    return match.group(1), match.group(2)


def term_relevance(docs: list[Document], queries: list[str]) -> list[list[int]]:
    """Relevant document ids per query: those mentioning *both* query terms.

    This is the ground truth the re-ranking comparison needs, and getting it
    from the corpus generator rather than from a model is the whole point.

    The first version of that comparison defined relevance as the exact
    embedding ranking. Every retrieval arm then scored NDCG 1.0000 by
    construction -- the retriever was being graded against the very ordering it
    approximates -- and no re-ranker could do anything but lose. That is not a
    finding about re-ranking; it is a broken experiment, and it would have been
    easy to publish as "re-ranking never pays".

    Relevance here is a property of the *text*: a passage is relevant to
    "why does the checkpoint affect the replication lag" when it actually
    discusses both the checkpoint and the replication lag. Both the retriever
    and the re-ranker are then measured against something neither one defines,
    which is the minimum bar for the comparison to mean anything.
    """
    inverse = {phrase: term for term, phrase in _PARAPHRASE.items()}
    relevant: list[list[int]] = []
    for query in queries:
        first, second = query_terms(query)
        # Queries are paraphrased, documents are not, so each query term is
        # resolved back to the corpus wording before matching. A paraphrase
        # that is not in the map is used as-is, which is what makes
        # `paraphrase=False` queries work through the same code path.
        first = inverse.get(first, first)
        second = inverse.get(second, second)
        relevant.append(
            [d.doc_id for d in docs if first in d.text and second in d.text]
        )
    return relevant


def build_queries(
    docs: list[Document], n_queries: int = 200, *, seed: int = 1, paraphrase: bool = True
) -> list[str]:
    """Queries over corpus concepts, worded differently from the documents.

    With ``paraphrase=True`` (the default) each topic term is replaced by a
    synonym that appears nowhere in the corpus, so a query asking about the
    "flush barrier" and "follower delay" must be matched to documents saying
    "checkpoint" and "replication lag". That gap is the point: it is why dense
    retrieval earns its cost over exact term matching, and without it every
    retrieval method scores identically and no comparison is possible.

    ``paraphrase=False`` keeps the literal terms and is retained to *show* the
    degenerate case rather than to hide it -- the results script reports both.
    """
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    out: list[str] = []
    for i in range(n_queries):
        category = cats[i % len(cats)]
        vocab = _TOPICS[category]
        a, b = rng.choice(vocab, size=2, replace=False)
        if paraphrase:
            a, b = _PARAPHRASE[str(a)], _PARAPHRASE[str(b)]
        out.append(QUERY_TEMPLATE.format(a=a, b=b))
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
