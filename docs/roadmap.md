# Delivery roadmap

| Phase | Scope | Exit evidence |
| --- | --- | --- |
| 0 | Monorepo, API/web shell, Compose profiles, CI, ADR and threat model | CPU profile health, lint/type/unit/build/security green |
| 1 | Local model adapters, embeddings/reranking, 20-case eval seed | Streaming debug endpoint, adapter contracts, and seed evaluation report |
| 2 | Sandboxed ingestion and status stream | PDF/DOCX/XLSX and malicious-file integration suite |
| 3 | Hybrid RAG, citations, refusal, cache, model bake-off | 50+ case eval report and scoped-cache tests |
| 4 | Streaming project/chat/document UI | CPU-profile Playwright and accessibility suite |
| 5 | Keycloak, RBAC, audit, retention/deletion | Cross-tenant negatives and zero-residue cascade |
| 6 | Prometheus/Grafana, Langfuse, admin status | Populated dashboards, masking and trace deletion |
| 7 | Hardening, performance, backup/restore, packaging | Tier-M load target, restore rehearsal, pilot release |

Later phases must not begin until the prior exit evidence is green and reviewed.
