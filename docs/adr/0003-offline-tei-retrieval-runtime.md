# ADR-0003: Offline TEI retrieval workers behind an internal gateway

- Status: Accepted
- Date: 2026-08-03

## Context

Argus Panopt needs BGE-M3 embeddings and bge-reranker-v2-m3 scoring without
exposing PHI or coupling the backend to an engine-specific API. The two models
have different runtime tasks and independently pinned artifact bundles. Runtime
model downloads, public network routes, mutable image tags, and provider SDKs
are prohibited.

## Decision

Run two Text Embeddings Inference 1.9.1 CPU workers: one for embeddings and one
for sequence-classification reranking. Pin the TEI image by digest and mount one
checksum-verified local bundle into each worker at `/models/model`. Force
offline mode and attach the workers only to the internal processing network.

Place a small FastAPI gateway at the existing `embedding-service` endpoint.
It preserves the provider-neutral Argus `/v1/embeddings` and `/v1/rerank`
contracts and translates them to TEI's local `/embed` and `/rerank`
contracts. It treats worker responses as untrusted, applies input and response
bounds, rejects malformed or non-finite results, disables redirects and proxy
inheritance, and returns only content-safe failures.

All three services run non-root with read-only root filesystems, dropped
capabilities, no-new-privileges, bounded writable tmpfs, and CPU/RAM/PID
ceilings. Preflight re-verifies both bundles before a real-model profile starts,
and CI validates the rendered policy without requiring model weights.

## Consequences

- Backend retrieval ports remain independent of TEI request and response types.
- Embedding and reranking failures and scaling can be observed independently.
- No retrieval container needs an internet route after operator provisioning.
- A real deployment requires both verified multi-gigabyte bundles and an AMD64
  host because the pinned TEI CPU image is platform-specific.
- The gateway adds one internal hop and must remain contract-tested alongside
  backend adapters.
