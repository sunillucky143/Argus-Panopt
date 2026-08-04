# Argus Panopt

Argus Panopt is a self-hosted, privacy-first document intelligence platform for
healthcare and other sensitive workloads. All document processing and model
inference are designed to remain inside the operator's deployment.

This repository contains the Phase 0 production foundation and active **Phase 1**
inference work: provider-neutral model ports, local streaming adapters, model-aware
readiness, a 20-case synthetic evaluation harness, hardened Compose profiles,
and CI/security gates.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2
- Tier S CPU: 15 GB of free disk
- Tier M GPU: 30 GB free disk, AMD64, one 24 GB NVIDIA GPU, driver, and
  NVIDIA Container Toolkit
- For development: Python 3.12, [uv](https://docs.astral.sh/uv/), and Node.js 22

## Quickstart (CPU)

1. Create local configuration:

   ```sh
   cp deploy/.env.example .env
   sh deploy/init-secrets.sh
   ```

2. Replace every value marked `replace-before-production`, then rerun
   `sh deploy/init-secrets.sh --force`. The example values are suitable only for
   a local Phase 0 smoke test.

3. Provision the pinned Tier S model from the repository root:

   ```sh
   python -m inference.download_model \
     --manifest inference/manifests/tier-s-gemma-3-4b-it-q4_k_m.json \
     --output-dir models
   ```

4. Run the preflight and start the stack:

   ```sh
   sh deploy/preflight.sh cpu
   docker compose --profile cpu up --build
   ```

5. Open:

   - App: <http://localhost:8080>
   - API docs: <http://localhost:8080/api/docs>
   - API readiness: <http://localhost:8080/api/health/ready>

The `cpu` profile starts the digest-pinned llama.cpp server with the verified
model directory mounted read-only. It has no published port and runs only
on the internal processing network. See
[`inference/README.md`](inference/README.md) for runtime details.

Stop the stack with `docker compose --profile cpu down`.

## Quickstart (GPU)

Tier M requires an AMD64 host, at least 30 GB of free disk, one NVIDIA GPU with
at least 24 GB VRAM, a working NVIDIA driver, and NVIDIA Container Toolkit.

1. Complete steps 1 and 2 from the CPU quickstart, then replace these `.env`
   values:

   ```dotenv
   ARGUS_TIER=M
   ARGUS_MODEL_PROVIDER=vllm
   ARGUS_MODEL_NAME=qwen2.5-vl-7b-instruct-awq
   ARGUS_MODEL_ENDPOINT=http://inference-gpu:8000/v1
   ARGUS_MODEL_CONTEXT_CEILING=65536
   ```

2. Provision the pinned Qwen and both BGE bundles documented in
   [`inference/README.md`](inference/README.md).

3. Run the GPU preflight and start Tier M:

   ```sh
   sh deploy/preflight.sh gpu
   docker compose --profile gpu up --build
   ```

The vLLM port is not published to the host. Use the app and readiness URLs
listed above, and stop with `docker compose --profile gpu down`.

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

Run the synthetic Phase 1 evaluation seed and produce metadata-only reports:

```sh
uv run --project backend python -m evals.run --runner fixture
```

See [`evals/README.md`](evals/README.md) for local-adapter evaluation options.

Validate Compose and the no-external-AI policy:

```sh
docker compose --env-file deploy/.env.example --profile cpu config --quiet
docker compose --env-file deploy/.env.example --profile gpu config --quiet
python scripts/check_model_runtime_policy.py
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
