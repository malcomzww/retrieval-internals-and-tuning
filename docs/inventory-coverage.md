# Inventory coverage

Anchored to `Bucket_Concept_Inventory.md`, bucket B2 (Agentic & multimodal).

- 2B vector DB internals: HNSW (M, ef_construction, ef_search), IVF-PQ, flat
- 2B chunking: fixed, recursive, semantic, late; overlap; parent-document; small-to-big
- 2B BM25 and sparse retrieval
- 2B hybrid search and fusion: RRF, weighted score fusion, normalization pitfalls
- 2B re-ranking: cross-encoders, ColBERT late-interaction, latency cost vs quality gain, when to skip
- 2B query transformation: rewriting, HyDE, multi-query, decomposition, routing
- 2B metadata filtering, pre- vs post-filtering, the filtered-ANN problem, index rebuild and freshness
- 2B pgvector vs Pinecone, when Postgres is enough
- 2B graph RAG, structured/SQL retrieval (TRY-J)
