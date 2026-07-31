# Security and threat model

## Scope and assumptions

Argus Panopt treats every uploaded file, prompt, retrieved chunk, answer, and
trace as PHI. The operator controls the host, local network, encryption keys,
identity configuration, backups, physical access, and incident response. This
document covers technical controls in the application and Compose deployment;
it does not certify an organization as HIPAA or SOC 2 compliant.

Trust boundaries are the browser/reverse proxy, authenticated API, worker queue,
untrusted parser sandbox, model services, data stores, observability stack, and
operator administration plane. Only the reverse proxy is externally reachable.

## STRIDE analysis

| Threat | Category | Primary controls | Verification |
| --- | --- | --- | --- |
| Stolen or replayed access token | Spoofing | Keycloak OIDC, short access-token TTL, refresh rotation, MFA capability, TLS, idle timeout | Negative auth tests in Phase 5 |
| Cross-project object identifier | Spoofing / elevation | Project role checks and tenant predicate in every query; deny by default | User A cannot access project B integration tests |
| Malicious PDF/Office payload or zip bomb | Tampering / DoS | MIME plus magic validation, size and expansion limits, ClamAV, isolated non-root parser with read-only root, tmpfs, no network, CPU/RAM ceilings | Malicious upload suite and runtime egress test |
| Document text attempts prompt injection | Tampering / elevation | Retrieved text is untrusted data, versioned system templates, instruction neutralization, no model-triggered tools, structured citation metadata | Adversarial golden cases in Phase 3 |
| User denies upload, query, or deletion | Repudiation | Append-only metadata audit log with actor, IP, outcome, request ID, masked or hashed query | Audit completeness tests in Phase 5 |
| Raw content appears in logs or traces | Information disclosure | Structured logs, metadata-only INFO events, error redaction, PII masking, configurable trace retention | Log-capture and trace-masking tests |
| Cache or vector leaks across tenants | Information disclosure | Project and user scoped cache keys, project predicates, corpus-version invalidation, no shared semantic cache | Negative cache and retrieval tests |
| Token or secret exposed in image/source | Information disclosure | Docker secrets support, ignored local env, gitleaks, least-privilege service identities | CI secret scan and image inspection |
| Container reaches an outside host | Information disclosure | Internal processing network, no published processing ports, egress test | `scripts/check_processing_egress.sh` |
| Parser or model container escape | Elevation | Non-root, dropped capabilities, no-new-privileges, read-only rootfs, pids and resource limits; parser seccomp in Phase 2 | Compose policy and container inspection tests |
| Oversized workload exhausts services | Denial of service | Upload/context ceilings, rate limits, queue backpressure, per-container resource ceilings, request timeout | Property, load, and soak tests |
| Compromised dependency or image | Tampering / elevation | Lockfiles, Dependabot, audit/SAST/secret/image scans, SBOM releases, reviewed upgrades | Required CI security job |
| Replaced or corrupted model weights | Tampering | Immutable upstream revision, exact size and SHA-256 manifest, restricted HTTPS source/redirects, atomic verified install, read-only runtime mount | Offline provisioner tests and pre-deployment verification |
| Privileged insider reads content | Information disclosure | Least-privilege RBAC, content access audit, encrypted volumes, trace masking/deletion, operator key custody | Access review and audit procedures |

## Baseline controls present in Phase 0

- API inputs and outputs use strict typed contracts.
- Correlation IDs are validated; invalid caller values are replaced.
- Browser responses set CSP, frame denial, MIME sniffing denial, permissions,
  and referrer controls.
- Processing services are placed on an internal Docker network.
- Application containers run non-root with read-only filesystems, dropped
  capabilities, no-new-privileges, pids limits, and CPU/RAM ceilings.
- Database/cache credentials enter through Docker secrets.
- CI blocks known external AI clients and hosts, secrets, critical dependency
  vulnerabilities, and common static-analysis findings.
- Model adapter configuration accepts only loopback, private-address, or
  single-label deployment endpoints, blocking explicit public model hosts; the
  internal processing network remains the egress enforcement boundary.
- Model weights are acquired only as an explicit operator provisioning action.
  Version-controlled manifests pin the immutable source revision, byte size,
  SHA-256 digest, and license; verified artifacts are atomically installed
  before being mounted read-only into an isolated runtime service.
- INFO logging of document text and prompts is prohibited by policy and review.

## Required design rules for future phases

1. Authorization is enforced in storage queries, never after fetching rows.
2. Uploaded bytes enter only the parser sandbox after AV/type/size checks.
3. Model output is inert text: it cannot invoke tools or create trusted links.
4. Citations are rendered from server-owned structured metadata.
5. HTML is never rendered from model output without sanitization.
6. Cache keys include user, project, normalized query, and corpus version.
7. Content deletion is idempotent, ordered, audited, and residue-tested.
8. No processing service receives a route to the public internet.

## Security reporting

Do not open a public issue containing a vulnerability, PHI, credentials, or
exploit details. Contact the repository owner privately. Rotate any exposed
credential immediately and follow the incident procedure in `operations.md`.
