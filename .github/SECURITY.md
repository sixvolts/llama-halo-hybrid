# Security policy (llama-halo-hybrid)

llama-halo-hybrid is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) maintained by one person.
The root [SECURITY.md](../SECURITY.md) is upstream's and still describes how to run llama.cpp safely; this file
says where to report problems in this fork.

## Reporting a vulnerability

Report it privately through this repository's **Security** tab (**Report a vulnerability**). Do not open a public
issue. This is a best-effort project; please allow time for a fix before disclosing.

- Problems in code this fork adds or changes (the RPC protocol extensions, new ggml ops, the scheduler and
  prefill-lane code, the server changes) belong here.
- Problems that also exist in upstream llama.cpp should be reported upstream as well, following their policy.

## Running the server safely

- `llama-server` has no authentication unless you pass `--api-key`. The cookbook launch lines bind
  `--host 0.0.0.0` so the web UI is reachable from the LAN; on any network you do not fully trust, bind to
  `127.0.0.1` or set `--api-key`.
- `ggml-rpc-server` has no authentication and executes graphs sent to it. Bind it only to a dedicated link
  address between your own machines and firewall the port; never expose it to an untrusted network.
- Only load model files you trust.
