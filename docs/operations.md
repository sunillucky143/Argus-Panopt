# Operations runbook

## Model provisioning

Model provisioning is a controlled pre-deployment action and is not performed
by a running processing service.

1. Review the selected manifest under `inference/manifests/` and accept its
   linked model license.
2. On an approved staging host with temporary access to the manifest's source,
   run:

   ```sh
   python -m inference.download_model \
     --manifest inference/manifests/tier-s-gemma-3-4b-it-q4_k_m.json \
     --output-dir models
   ```

For the Tier M Qwen bundle, run:

```sh
python -m inference.download_bundle \
  --manifest inference/manifests/tier-m-qwen2.5-vl-7b-instruct-awq.json \
  --output-dir models
```

For the two BGE retrieval bundles, run:

```sh
python -m inference.download_bundle --manifest inference/manifests/bge-m3-onnx-fp32.json --output-dir models
python -m inference.download_bundle --manifest inference/manifests/bge-reranker-v2-m3-safetensors-fp32.json --output-dir models
```

Bundle directories become visible only after every required file passes its
size and checksum checks. A valid existing bundle skips the network; an
invalid existing bundle is preserved and fails closed.

3. Record each manifest revision and SHA-256 value in the change record. Do not
   copy signed download URLs, credentials, or model contents into the record.
4. Remove external network access before starting processing services. Runtime
   services receive only verified, read-only model mounts.
5. Rerun the relevant command after moving an artifact or bundle. Existing
   content skips the network only when every exact size and SHA-256 still
   matches.

If provisioning fails, preserve the previous verified artifact. The tool does
not replace the destination until verification succeeds, removes temporary
artifacts after ordinary failures, and reports any cleanup failure.

## Deployment

1. Review `docs/security.md` and size the host for tier S, M, or L.
2. Copy `deploy/.env.example` to `.env`, replace all example credentials, and
   restrict file permissions to the deployment account.
   For Tier M, also replace `ARGUS_TIER` and the four `ARGUS_MODEL_*` values
   with the GPU values documented in the example file.
3. Terminate TLS at the deployment reverse proxy using an operator-managed
   certificate and enable HSTS.
4. Run `./deploy/preflight.sh cpu` or `gpu`. Preflight fails closed when
   any required bundle is absent or fails checksum verification. CPU verifies
   the pinned llama.cpp artifact; GPU verifies Qwen, the adapter/profile
   alignment, NVIDIA Container Toolkit, and at least 24 GB VRAM. The vLLM and
   TEI images require AMD64.
5. Run `docker compose --profile <tier> up -d --build`.
6. Confirm `/api/health/live` and `/api/health/ready` through the TLS endpoint.
7. Confirm the selected inference service, `embedding-service`,
   `embedding-engine`, and `reranker-engine` are healthy with
   `docker compose ps`.
8. Review container health, resource ceilings, and processing network isolation.

Never place PHI or secrets in command history, tickets, screenshots, or support
bundles.

## Backup (implemented in Phase 7)

The supported backup set will contain an encrypted PostgreSQL logical backup,
encrypted persistent document volume snapshot, configuration excluding secret
values, and a manifest with version/checksums. Redis is disposable and is not a
source of record. Backups must be encrypted before leaving the host and access
must be audited.

Until Phase 7 scripts and restore rehearsal are complete, the deployment is not
approved for pilot production data.

## Restore (implemented and rehearsed in Phase 7)

Restore into an isolated environment with no user traffic, verify manifest
checksums, restore volumes and PostgreSQL, run migrations, then execute
referential-integrity and deletion-residue checks before reopening traffic.
Record operator, timestamps, backup ID, software version, and verification
outcome without recording content.

## Upgrade

1. Read release notes and ADRs; confirm a rollback-compatible backup.
2. Pull immutable release images and verify signatures/SBOM.
3. Enter maintenance mode and drain ingestion work.
4. Run database migrations as a one-shot least-privilege task.
5. Start services and validate health, auth, upload, query, citation, and audit.
6. Roll back images only when the migration notes explicitly permit it.

## Incident response

1. **Contain:** remove external access, preserve logs and audit metadata, revoke
   affected sessions, and rotate suspected credentials.
2. **Assess:** identify tenants, data categories, time range, and control failure.
   Do not copy raw PHI into the incident record.
3. **Eradicate:** patch the cause, rebuild from trusted artifacts, and scan.
4. **Recover:** restore or redeploy, run security/integrity checks, and monitor.
5. **Notify and learn:** the operator follows contractual and regulatory
   notification duties, then records a blameless post-incident review.

## Routine checks

- Daily: health, disk capacity, backup outcome, failed authentication, queue age.
- Weekly: vulnerability alerts, certificate lifetime, resource headroom.
- Monthly: restore test, access review, deletion job completion, audit retention.
- Per release: SBOM, image scan, upgrade rehearsal, and rollback decision.
