# Argus Panopt

Argus Panopt is a self-hosted, privacy-first document intelligence platform for
healthcare and other sensitive workloads. All document processing and model
inference are designed to remain inside the operator's deployment.

This repository contains the Phase 0 production foundation and active **Phase 1**
inference work: provider-neutral model ports, local streaming adapters, model-aware
readiness, hardened Compose profiles, and CI/security gates.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2
- 10 GB of free disk for the Phase 0 images; an additional 2.5 GB for the
  optional Tier S model artifact
- For development: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 22
- For the future `gpu` inference profile: NVIDIA driver and Container Toolkit

## Quickstart (CPU)

1. Create local configuration:

   ```sh
   cp deploy/.env.example .env
   sh deploy/init-secrets.sh
   ```

2. Replace every value marked `replace-before-production`, then rerun
   `sh deploy/init-secrets.sh --force`. The example values are suitable only for
   a local Phase 0 smoke test.

3. Run the preflight and start the stack:

   ```sh
   sh deploy/preflight.sh cpu
   docker compose --profile cpu up --build
   ```

4. Open:

   - App: <http://localhost:8080>
   - API docs: <http://localhost:8080/api/docs>
   - API readiness: <http://localhost:8080/api/health/ready>

The current Compose default uses the deterministic smoke adapter. The Tier S
weights can now be provisioned with the checksum-verified tool documented in
[`inference/README.md`](inference/README.md); the next Phase 1 runtime work item
switches the CPU profile to llama.cpp after verified weights are installed.

Stop the stack with `docker compose --profile cpu down`.

## Local development

Backend:

```sh
cd backend
uv sync --all-groups
uv run ruff format --check .
uv run ruff check .
uv run mypy app tests
uv run pytest
```

Frontend:

```sh
cd frontend
npm ci
npm run check
```

Validate Compose and the no-external-AI policy:

```sh
docker compose --env-file deploy/.env.example --profile cpu config --quiet
python scripts/check_no_external_ai.py .
```

## Repository map

- `backend/` — FastAPI service, organized around domain ports and adapters
- `frontend/` — strict TypeScript React/Vite client
- `inference/` — local inference manifests and configuration (Phase 1)
- `embedding-service/` — local embeddings and reranking service (Phase 1)
- `deploy/` — Compose, preflight, observability, and identity configuration
- `evals/` — model evaluation data, harness, and generated reports
- `docs/` — architecture, security, operations, compliance, and ADRs

## Security posture

Treat all files, prompts, and responses as PHI. Do not place real secrets in
`.env`, logs, issues, test fixtures, or commits. See
[`docs/security.md`](docs/security.md) before changing an input boundary or
network path.

The software is intended to be **HIPAA-ready and SOC 2-aligned** at the
technical-control level. Deployment and organizational compliance remain the
operator's responsibility.

## Delivery status

The phased roadmap is tracked in [`docs/roadmap.md`](docs/roadmap.md). A phase
does not begin until the previous phase's Definition of Done is met.
