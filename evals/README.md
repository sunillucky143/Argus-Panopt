# Evaluations

Phase 1 provides an adapter-independent evaluation CLI. It exercises local
model adapters through the backend's normalized streaming endpoint.

The checked-in `datasets/phase1-seed.json` contains 20 fully synthetic
healthcare document questions spanning PDF, DOCX, and XLSX source scenarios.

## Run the seed

Validate harness and report plumbing without starting a model:

```sh
uv run --project backend python -m evals.run --runner fixture
```

Run against the adapter configured in a local backend:

```sh
ARGUS_DEBUG_INFERENCE_ENABLED=true \
  uv run --project backend python -m evals.run --runner http
```

The HTTP runner accepts only an explicit loopback, RFC 1918/ULA, or single-label
deployment host and one of the internal debug-generation paths. It uses finite
limits, verifies TLS by default, and disables redirects and inherited proxies.

## Report safety

Reports are metadata-only: case identifiers, scores, timings, finish reasons,
errors, and response SHA-256 values. They do not store contexts, questions,
expected answers, or generated text. Generated files stay under the ignored
`evals/reports/` directory and are written with owner-only permissions.

Exit code `0` means every case passed, `1` means a case failed or errored, and
`2` means the dataset, configuration, or report output was invalid.
