# Inference

Phase 1 adds checksum-verified model manifests and local vLLM/llama.cpp
configurations here. Application code must access those engines only through
provider-agnostic ports.

The API adapters use the engines' local OpenAI-compatible
`/v1/chat/completions` endpoints directly through `httpx`; no third-party
model-provider SDK is used. Model endpoint configuration is restricted to
loopback, private IP, or single-label Compose service hosts.

The API readiness endpoint probes each engine's root `/health` route and reports
the configured model name, quantization, context ceiling, and capabilities.
llama.cpp is ready only after it returns its loaded-model `{"status":"ok"}`
payload; vLLM readiness follows its status-only HTTP 200 contract.
