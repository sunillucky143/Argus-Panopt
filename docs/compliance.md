# Compliance control mapping

Argus Panopt is designed to be **HIPAA-ready and SOC 2-aligned**. This is a
technical-control map, not a certification or legal conclusion. The operator is
responsible for policies, workforce training, risk analysis, vendor agreements,
physical safeguards, and correct configuration.

| Framework control | Technical implementation | Verification / status |
| --- | --- | --- |
| HIPAA access control, 45 CFR 164.312(a)(1) | OIDC, MFA-capable identity, per-project RBAC, short sessions, least privilege | Phase 5 negative authorization tests |
| HIPAA unique user identification | Keycloak subject propagated to audit and project grants | Phase 5 |
| HIPAA emergency access procedure | Operator-run documented recovery process; no application backdoor | Operations review, Phase 7 |
| HIPAA automatic logoff | Configurable access and idle timeout | Phase 5 browser/API tests |
| HIPAA encryption/decryption | TLS, encrypted operator volumes, secrets outside images | Deployment inspection, Phase 7 |
| HIPAA audit controls, 164.312(b) | Append-only metadata audit for auth, content, query, admin, retention events | Phase 5 audit completeness tests |
| HIPAA integrity, 164.312(c)(1) | File hashes, immutable artifact identity, database constraints, verified deletion | Phases 2 and 5 integration tests |
| HIPAA person/entity authentication, 164.312(d) | Standards-based OIDC validation through Keycloak | Phase 5 token tests |
| HIPAA transmission security, 164.312(e)(1) | TLS/HSTS at ingress; isolated internal networks; processing egress denial | TLS review and egress test |
| SOC 2 CC6 logical access | RBAC, secrets handling, network segmentation, reviewed administration | Phases 5 and 7 |
| SOC 2 CC7 monitoring and response | Content-free inference metrics, structured security events, Prometheus alerts, incident runbook | Phase 1 metric-contract tests; Phase 6 dashboards and drills |
| SOC 2 CC8 change management | Protected main, small PRs, required CI, ADRs, release SBOM | Repository rules and CI evidence |
| SOC 2 CC9 risk mitigation | STRIDE model, dependency/image scans, backup/restore rehearsal | CI and Phase 7 evidence |

## Phase 0 evidence

- Baseline security headers: `backend/tests/test_health.py`
- No external AI clients/hosts: `scripts/check_no_external_ai.py`
- Runtime processing egress denial: `scripts/check_processing_egress.sh`
- Reproducible dependencies: `backend/uv.lock`, `frontend/package-lock.json`
- Security gates: `.github/workflows/ci.yml`
- Threat analysis: `docs/security.md`

## Phase 1 telemetry evidence

- Provider-neutral lifecycle instrumentation:
  `backend/tests/test_inference_metrics.py`
- Content-free bounded labels and private scrape route:
  `backend/tests/test_inference_metrics.py`
- Local-engine usage normalization and safe protocol failures:
  `backend/tests/test_openai_compatible_adapters.py`
- Telemetry architecture decision:
  `docs/adr/0004-provider-neutral-inference-metrics.md`

Mappings must be updated in the same PR as any control-affecting feature.
