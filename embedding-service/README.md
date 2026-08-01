# Embedding service

Phase 1 adds the dedicated, CPU-hosted BGE-M3 embedding and reranking service
here. It will have no external network route.

## Internal adapter contract

The backend exposes provider-neutral `EmbeddingPort` and `RerankerPort`
interfaces. `EmbeddingServiceAdapter` and `RerankerServiceAdapter` translate
those values to this service's internal HTTP contract. Domain and API code do
not depend on BGE request or response formats.

The configured base URL must use HTTP(S) and name only a loopback address,
private address, or single-label deployment host. Redirects and environment
proxy settings are disabled. The Compose processing network remains the
enforcement boundary that prevents runtime egress.

### Embeddings

`POST /v1/embeddings`

```json
{"model":"bge-m3","texts":["text"],"kind":"query"}
```

```json
{"model":"bge-m3","data":[{"index":0,"embedding":[0.1,0.2]}]}
```

The adapter restores input order from the response indexes and rejects missing,
duplicate, non-finite, empty, or inconsistent-dimension vectors.

### Reranking

`POST /v1/rerank`

```json
{"model":"bge-reranker-v2-m3","query":"text","passages":[{"identifier":"p1","text":"text"}],"top_k":1}
```

```json
{"model":"bge-reranker-v2-m3","data":[{"identifier":"p1","score":0.9}]}
```

Passage metadata never leaves the backend. The service receives only the
opaque identifier and text, and the adapter joins results back to the original
trusted metadata before returning descending scores.

Both contracts are bounded to 256 inputs of at most 32,768 characters each.
Model identifiers must match configuration exactly. Transport and protocol
errors are generic and never include input text or service response bodies.

The next work item packages the checksum-pinned BGE-M3 and
bge-reranker-v2-m3 runtime behind this contract. Until that service is added,
these adapters are contract-tested with an in-memory HTTP transport and are not
wired into application request paths.
