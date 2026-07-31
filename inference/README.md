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

Run the offline provisioner tests without downloading model weights:

```sh
python -m unittest discover -s inference/tests -v
```

When updating a model, add a new manifest after separately confirming the
immutable upstream revision, exact byte size, license, and SHA-256. Do not edit
an existing manifest in place for a different artifact.

The API adapters use the engines' local OpenAI-compatible
`/v1/chat/completions` endpoints directly through `httpx`; no third-party
model-provider SDK is used. Model endpoint configuration is restricted to
loopback, private IP, or single-label Compose service hosts.

The API readiness endpoint probes each engine's root `/health` route and reports
the configured model name, quantization, context ceiling, and capabilities.
llama.cpp is ready only after it returns its loaded-model `{"status":"ok"}`
payload; vLLM readiness follows its status-only HTTP 200 contract.
