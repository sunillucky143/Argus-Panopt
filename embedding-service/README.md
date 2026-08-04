# Embedding service

Phase 1 provides a dedicated local BGE-M3 embedding and reranking runtime.
All three retrieval containers have only the internal processing-network route;
the service never sends document text to an external endpoint.

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

## Local runtime topology

The Compose `cpu` and `gpu` profiles start this FastAPI contract gateway and
two Text Embeddings Inference 1.9.1 CPU workers. The gateway translates only to
the workers' local `/embed` and `/rerank` endpoints. It disables redirects,
environment proxies, API documentation, and access logs; it caps inputs,
responses, vector dimensions, and timeouts; and it replaces validation,
transport, and protocol failures with content-safe messages.

Both workers use one digest-pinned TEI image and run as UID/GID 10001 with a
read-only root filesystem, all capabilities dropped, no-new-privileges, bounded
tmpfs/PIDs/CPU/RAM, no host ports, and only the internal processing network.
`HF_HUB_OFFLINE=1` prevents registry access. Each worker receives exactly one
checksum-verified bundle directory at `/models/model` as a read-only mount.

`GET /health/live` reports gateway process liveness. `GET /health/ready`
returns HTTP 200 only when both TEI workers report healthy; otherwise it returns
HTTP 503 without exposing worker details. TEI's supported CPU image is pinned to
`linux/amd64`; deploy this profile on an AMD64 host.
