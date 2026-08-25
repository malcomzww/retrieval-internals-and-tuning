# ADR 0001: When Postgres with pgvector is enough

- **Status:** accepted
- **Date:** 2026-08-25
- **Context corpus:** 20,000 passages, 384-dimensional MiniLM embeddings
- **Hardware:** 24-core CPU, 32 GB RAM, no GPU

## Context

The default architectural move when a project needs vector search is to add a
dedicated vector database. That decision imports a new service to operate, a
second copy of the data, a synchronisation path between the two, and a new
failure mode where the index and the source of truth disagree.

This repo measured the two things that decision is usually justified by --
query latency and filtered-query recall -- and the measurements do not support
the default at this scale.

## Decision

**Use Postgres with `pgvector` until at least one of these is true:**

1. The corpus exceeds roughly a million vectors, *or*
2. p99 latency must stay under a few milliseconds at high concurrency, *or*
3. The workload is dominated by unfiltered top-k over the whole corpus.

Below those thresholds a relational database holding the vectors alongside the
rows they describe is not a compromise. It is the better engineering choice,
and two measurements in this repo say so.

## Evidence

### 1. An exact scan is a few milliseconds at this scale

A brute-force inner-product scan over 20,000 x 384 float32 vectors, pinned to
a single core, costs roughly 4 ms per query — measured, not estimated; the
absolute figure for this machine is in the gitignored raw results file.

That is *already inside* most RAG latency budgets, where an LLM generation
step costs hundreds of milliseconds to seconds. Approximate search on a corpus
this size optimises a component that is not the bottleneck, and it pays for
that optimisation with recall below 1.0 and a graph to keep in memory.

The ordering that matters, and that holds independent of machine: **exact scan
latency scales linearly with corpus size, and at 20k vectors it is small
compared with a single LLM call.** The question is not whether ANN is faster —
it is, by roughly 20x here — but whether the difference is visible to a user
behind a generation step. At this scale it is not.

### 2. Selective metadata filters *invert* the comparison

This is the finding that most strongly favours Postgres, and it is the one
usually missed.

Vector indexes handle metadata filters badly. The graph is built over the whole
corpus, so a filtered query must either post-filter its results — which
collapses recall — or over-fetch enough candidates that some survive. Measured
on this corpus at k=10 (full table in `results/filtering.md`):

| filter keeps | post-filter recall | queries returning nothing |
|---|---|---|
| 50% | 0.495 | 0% |
| 12.5% | 0.127 | 27% |
| 3.1% | 0.038 | 68% |
| 1.6% | 0.008 | 92% |

At 1.6% selectivity naive post-filtering returns **an empty result set for 92%
of queries**. Nothing errors; the API returns `200 OK` with an empty list.

A relational database does the opposite thing by default. It applies the
`WHERE` clause first and scans only the surviving rows — and because that
subset is small, the exact scan over it is *fast*. Measured here, at 3.1%
selectivity and below, **pre-filtered exact search is both perfectly accurate
and lower-latency than the filtered ANN search it replaces.**

That is a crossover, and it is the machine-independent part of the claim: as a
filter becomes more selective, the exact-scan cost falls linearly with the
surviving row count while the ANN index keeps paying for a graph built over
rows the query has already excluded. Below some selectivity the exact scan must
win on both axes. Where the crossover sits depends on the machine; that it
exists does not.

`pgvector` gets this behaviour from the query planner without anyone designing
it. A dedicated vector store generally requires you to know about the filtered-
ANN problem in advance and to configure around it.

## Consequences

**Accepted:**

- Higher unfiltered query latency — roughly 20x an HNSW lookup here, and still
  a small fraction of an end-to-end RAG request.
- No horizontal read scaling beyond what Postgres replication provides.

**Gained:**

- One datastore, one backup, one consistency model. No index-vs-truth skew,
  which is a correctness property rather than an operational convenience.
- Transactional writes: a document and its embedding become visible together.
  Freshness is free rather than a rebuild schedule.
- Metadata filters that are *correct by default*, including selective ones.
- `JOIN`. Retrieval is frequently "documents matching this vector **and**
  belonging to this account **and** not soft-deleted", which is a relational
  query with a vector predicate rather than a vector query with filters bolted
  on.

## What would change this decision

- Corpus growth past ~1M vectors, where exact scan latency stops being
  negligible and the HNSW memory cost is justified.
- A workload that is genuinely unfiltered top-k at high QPS — the one case
  where the ANN index's advantage is not eroded by filtering.
- A latency budget with no LLM call in it, where a few milliseconds is the
  whole budget rather than 1% of it.

## Limitations of this evidence

Measured on one corpus at one size on one machine, with no `pgvector` instance
in the loop: the exact-scan baseline here is FAISS `IndexFlatIP` and numpy, not
Postgres, so it excludes SQL parsing, planning, tuple deserialisation and
network round-trip. Those add a real and unmeasured constant. The argument rests
on the *ordering* of costs and on the filtering crossover, both of which have
margins wide enough to survive that constant — but this ADR does not measure
Postgres itself, and should not be read as though it did.
