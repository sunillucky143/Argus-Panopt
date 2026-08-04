# Architecture

## System context

```mermaid
flowchart LR
  User["Clinic or back-office user"] -->|"TLS / browser"| Web["Argus web application"]
  Operator["Deployment operator"] -->|"Local administration"| Web
  Web -->|"Project-scoped API + SSE"| API["FastAPI gateway"]
  API -->|"OIDC validation"| Identity["Self-hosted Keycloak"]
  API -->|"Jobs"| Worker["Ingestion worker"]
  API -->|"Retrieval and audit"| Postgres["PostgreSQL + pgvector"]
  API -->|"Cache and rate limits"| Redis["Redis"]
  Worker --> Parser["Sandboxed Docling parser"]
  Worker --> Embedding["Local embedding + reranker"]
  API --> Model["Local model adapter"]
  Metrics["Prometheus + Grafana"] -.-> API
  Tracing["Self-hosted Langfuse"] -.-> API
```

There are no external AI services or processing-container egress paths.
Keycloak, model serving, parsing, observability, and storage all run within the
operator-controlled deployment.

## Backend containers and dependency direction

```mermaid
flowchart TB
  API["API adapter"] --> UseCases["Domain use cases"]
  Workers["Worker adapter"] --> UseCases
  UseCases --> Ports["Ports / protocols"]
  Infra["Postgres, Redis, model, parser, identity, tracing adapters"] --> Ports
  Domain["Domain entities and policies"] --> Ports
```

The arrows represent compile-time dependencies. Domain code never imports
adapters, web frameworks, database clients, or model engines.

## Runtime networks

```mermaid
flowchart LR
  Browser -->|"published port"| Proxy["Web / reverse proxy"]
  subgraph Internal["Internal processing network (no egress)"]
    API
    DB["PostgreSQL"]
    Cache["Redis"]
    Parser
    Model
    Embed["Embedding service"]
  end
  Proxy --> API
  API --> DB
  API --> Cache
  API --> Model
  API --> Embed
  Parser --> Embed
```

Phase 0 provisions the web, API, PostgreSQL, and Redis boundaries. Phase 1 adds
isolated llama.cpp CPU and vLLM GPU runtimes behind the provider-neutral model
adapter, the internal BGE retrieval gateway and workers, and provider-neutral
inference instrumentation. Later phases fill the remaining reserved service
locations.

## Inference telemetry boundary

```mermaid
flowchart LR
  API["API and future RAG use cases"] --> Decorator["Instrumented ChatModelPort"]
  Decorator --> Adapter["Fake, llama.cpp, or vLLM adapter"]
  Decorator --> MetricsPort["InferenceMetricsPort"]
  MetricsPort --> Prometheus["Private Prometheus registry"]
  Scraper["Internal scraper (Phase 6)"] -->|"GET /internal/metrics"| Prometheus
```

The decorator measures every configured chat provider through one domain
contract. Terminal chunks may carry normalized input/output token usage, while
provider payloads remain inside infrastructure adapters. Only the fixed
provider set, a validated configured model name, bounded outcomes, and bounded
finish reasons can become metric labels. Prompts, outputs, request metadata,
document identifiers, user identifiers, and exception messages never enter the
metrics port.

The API owns a private Prometheus registry and exposes it only at the backend's
`/internal/metrics` path. The backend has no published host port, and the web
proxy denies the entire `/api/internal/` namespace. Prometheus/Grafana service
provisioning and dashboards remain Phase 6 work.

## Data lifecycle

Project identity is the mandatory partition key for documents, chunks, vectors,
and caches. Persistent projects use encrypted operator-managed volumes.
Ephemeral projects parse raw files in tmpfs and do not persist the upload. The
idempotent deletion workflow introduced in Phase 5 removes every content-bearing
artifact before recording completion in the metadata-only audit log.
