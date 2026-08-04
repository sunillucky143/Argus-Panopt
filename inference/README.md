# Inference

Phase 1 stores checksum-verified model manifests and local vLLM/llama.cpp
configuration here. Application code must access those engines only through
provider-agnostic ports.

## Tier S model provisioning

The Tier S CPU baseline is Gemma 3 4B Instruct in GGUF Q4_K_M format. Its
manifest pins the upstream repository commit, exact filename, byte count,
SHA-256 digest, quantization, and license:

```sh
python -m inference.download_model \
  --manifest inference/manifests/tier-s-gemma-3-4b-it-q4_k_m.json \
  --output-dir models
```

Run provisioning from an operator workstation or controlled staging host after
reviewing and accepting the linked Gemma license. The tool accepts only the
approved HTTPS source, permits redirects only to approved artifact hosts,
streams into a temporary file, verifies the exact byte count and SHA-256, and
atomically installs the artifact. It stages an existing artifact through one
open file descriptor and atomically refreshes it only after verification,
skipping the network without a check/use gap. Model weights are never committed.

Provisioning is the only step that needs access to the upstream artifact host.
The runtime model service receives a read-only local model mount and remains on
the internal processing network with no download path.

## BGE model bundle provisioning

BGE-M3 embeddings and bge-reranker-v2-m3 are pinned as versioned,
multi-file bundles for the dedicated CPU retrieval runtime:

```sh
python -m inference.download_bundle \
  --manifest inference/manifests/bge-m3-onnx-fp32.json \
  --output-dir models
```

```sh
python -m inference.download_bundle \
  --manifest inference/manifests/bge-reranker-v2-m3-safetensors-fp32.json \
  --output-dir models
```

Each manifest restricts the approved BAAI repository, immutable commit,
runtime format, license, destination filenames, byte counts, and SHA-256
digests. The installer downloads into a private temporary directory and makes
the bundle visible only after every file passes a final verification. An
existing invalid bundle is preserved for investigation instead of being
overwritten; publish a new bundle ID when model content changes.

Run the offline provisioner tests without downloading model weights:

```sh
python -m unittest discover -s inference/tests -v
```

When updating a model, add a new manifest after separately confirming the
immutable upstream revision, exact byte size, license, and SHA-256. Do not edit
an existing manifest in place for a different artifact.
## Local retrieval runtime

The Compose `cpu` and `gpu` profiles mount the exact BGE bundle IDs above
into separate, digest-pinned Text Embeddings Inference 1.9.1 CPU workers.
Workers run in forced offline mode and receive only a read-only
`/models/model` mount on the internal processing network. The
`embedding-service` gateway preserves the provider-neutral Argus contracts
while translating requests to TEI's `/embed` and `/rerank` endpoints.

Before either real-model profile starts, `deploy/preflight.sh` verifies every
bundle file's name, byte count, and SHA-256. CI renders the real profile and
enforces the image digest, local command, exact mount source, offline
environment, network isolation, non-root identity, read-only filesystem, and
resource ceilings. The pinned TEI CPU image currently requires `linux/amd64`.



## llama.cpp CPU runtime

The Compose `cpu` profile starts the official llama.cpp server build `b9445`
from a pinned multi-architecture image digest. The service:

- runs as UID/GID 10001 with all capabilities dropped, no-new-privileges, a
  read-only root filesystem, PID/CPU/RAM ceilings, and a bounded tmpfs;
- mounts the host `models/` directory read-only and starts only from the exact
  pinned filename after preflight checksum verification;
- receives no model URL or registry option, publishes no host port, and joins
  only the internal processing network; and
- reports ready only when llama.cpp's `/health` endpoint confirms the model is
  loaded.

`deploy/preflight.sh cpu` re-verifies the installed artifact without network
access before validating Compose. CI renders and policy-checks the real CPU
profile, pulls the pinned runtime image, and runs application smoke tests under
the separate model-free `smoke` profile.

The API adapters use the engines' local OpenAI-compatible
`/v1/chat/completions` endpoints directly through `httpx`; no third-party
model-provider SDK is used. Model endpoint configuration is restricted to
loopback, private IP, or single-label Compose service hosts.

The API readiness endpoint probes each engine's root `/health` route and reports
the configured model name, quantization, context ceiling, and capabilities.
llama.cpp is ready only after it returns its loaded-model `{"status":"ok"}`
payload; vLLM readiness follows its status-only HTTP 200 contract.
