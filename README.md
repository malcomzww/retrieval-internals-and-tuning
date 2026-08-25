# retrieval-internals-and-tuning

HNSW, IVF-PQ, filtering and fusion measured on a 20,000-document corpus, built
around one question:

> **At what latency budget does re-ranking stop being worth it?**

The sibling repo
[`rag-eval-retrieval-vs-generation`](https://github.com/malcomzww/rag-eval-retrieval-vs-generation)
established *that* retrieval and generation must be scored separately. This one
is the index-internals half: what the knobs actually cost.

## The headline is a negative result

Charging each arm its **total** per-query latency — so a budget spent entirely
on retrieval competes against one split between cheap retrieval and
re-ranking — re-ranking never won at any budget tested:

| budget | winner | NDCG@10 | re-ranked? |
|---|---|---|---|
| 0.2 ms | `retrieve-only ef=32` | 0.1204 | no |
| 0.5 ms | `retrieve-only ef=32` | 0.1204 | no |

A cross-encoder on CPU costs more than simply raising `ef_search`, and raising
`ef_search` buys the same ranking quality. **This is a finding about CPU
re-ranking economics, not about re-ranking** — a GPU cross-encoder changes the
cost side entirely, and the file says so.

One thing worth reading in `results/reranking.md`: an earlier version of this
experiment scored relevance *by the embedding's own ranking*, which made every
retrieval arm score NDCG 1.0000 and made "re-ranking never pays" true by
construction. The setup was measuring itself. That is recorded rather than
quietly corrected.

## The result I would actually act on

Naive post-filtering collapses when the filter is selective — and it collapses
by *returning nothing*, which is worse than returning something wrong because
it looks like an empty result set rather than a bug:

| filter keeps | post-filter recall | over-fetch 10× | pre-filter exact | queries returning nothing |
|---|---|---|---|---|
| 50.0% | 0.490 | 0.965 | 1.000 | 0% |
| 12.5% | 0.120 | 0.922 | 1.000 | **27%** |
| 6.2% | 0.052 | 0.615 | 1.000 | **58%** |
| 3.1% | **0.020** | 0.313 | 1.000 | **82%** |

At a 3.1% filter, recall is 0.020 and four queries in five come back empty.
Over-fetching 10× recovers much of it; exact pre-filtering recovers all of it.
The failure mode is silent, which is what makes it dangerous.

## The fusion pitfall

Two retrievers, one relevant document. Both naive fusion methods pick the wrong
winner; only reciprocal rank fusion picks the relevant one:

| method | winner | relevant? |
|---|---|---|
| min-max score fusion | 3 | **no** |
| raw score fusion | 3 | **no** |
| reciprocal rank fusion | **7** | **yes** |

BM25 scores live on an unbounded scale and cosine scores on [0, 1]. Normalising
them into comparability is the step that quietly discards the signal; ranking
positions survive the change of units.

## What a recall target costs

| recall target | cheapest configuration | recall | relative cost |
|---|---|---|---|
| 0.90 | `M=8,efc=200,ef=50` | 0.913 | 3.4× |
| 0.95 | `M=16,efc=200,ef=50` | 0.982 | 4.6× |
| 0.99 | `M=16,efc=200,ef=200` | 0.997 | 12.1× |
| 1.00 | `M=16,efc=200,ef=400` | 1.000 | 27.4× |

The last two percentage points cost 6× what the first ninety do.

## Quickstart

```bash
uv sync --extra dev
uv run pytest -q                                     # 93 tests
uv run python scripts/generate_results.py --quick    # 60 queries
uv run python scripts/generate_results.py            # full sweep, slower
```

## Limitations

- **Latency is machine-dependent.** Only orderings, crossovers and relative
  costs are committed; absolute milliseconds go to the gitignored raw files.
  The re-ranking verdict is a *CPU* verdict.
- **Absolute NDCG is low (~0.12)** because relevance is judged against
  hand-labelled targets on paraphrased queries, not against the embedding's own
  ranking. That makes the comparison honest and the absolute number
  uninformative — only the arm ordering is claimed.
- **20,000 documents is small.** HNSW's cost curve bends where the graph stops
  fitting in cache, and that point is not established here.
- Without `sentence-transformers` installed the corpus falls back to a hashed
  bag-of-words embedder, which ties on short documents and caps recall. The
  scripts detect this and relax their assertions, reporting the embedder in use.

## License

MIT
