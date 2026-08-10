# Security

This repository is an experimental agent-execution and evaluation harness. It
is not hardened for production or multi-tenant use.

## Safe operating assumptions

- Run agent-authored commands only on a disposable, non-production Linux host.
- Use a dedicated least-privilege SSH identity and operating-system account.
- Do not expose secrets, personal files, cloud credentials, production data, or
  privileged sockets to the controller, remote worker, or task workspace.
- Keep Bubblewrap and the configured resource limits enabled. Do not add an
  unsafe host-execution fallback when sandbox setup fails.
- Keep model inference and task execution serial unless you have independently
  tested memory limits and interference.
- Review generated policy prompts and tool permissions before execution.

The harness restricts its model-facing tool surface, but this does not make
arbitrary generated code trustworthy. SSH configuration, the remote account,
Pi provider extensions, the operating system, and the model endpoint remain
part of the trusted computing base.

## Fixed-page Unbrowser smoke boundary

The optional Unbrowser smoke is intentionally not a general browsing tool:

- Model-authored Bash remains inside Bubblewrap with its network namespace
  unshared.
- A separate `unbrowser` child runs as the least-privilege user on the
  disposable SSH runner, outside Bubblewrap, because it requires outbound
  network access.
- The model cannot provide a URL. The adapter accepts only the exact public
  smoke page `https://example.com/`, checks the final reported URL, and exposes
  only `navigate`, `query`, `text`, and `blockmap`.
- Cookies, credentials, authentication, clicks, form submission, JavaScript
  evaluation, downloads, and raw JSON-RPC are not exposed.
- Every attempt receives a fresh process and temporary home with a minimal
  environment, bounded calls, timeouts, JSON-line limits, and result-size
  limits. The process group is killed on protocol failure or timeout.
- Page content is untrusted data. Policy prompts must never follow instructions
  found in page text.

Exact-URL pinning and post-navigation validation do not constitute a complete
SSRF or redirect defense because the outbound request occurs before the final
URL can be checked. Do not change the URL, enable redirects to untrusted
targets, add credentials, or expose mutating/advanced methods without a new
security review. Run this smoke only on a disposable runner that contains no
secrets.

## Interactive Unbrowser plumbing spike

The interactive Unbrowser path (`unbrowser_interactive` family) is a
**disposable-host plumbing spike, not a security boundary**. It adds
`click`, `type`, and `submit` to the adapter for the Wikipedia search smoke:

- The initial URL is controller-fixed to `https://en.wikipedia.org/wiki/Main_Page`.
- After navigation, the adapter checks `status` and `challenge` fields in the
  response. Non-200 status or a present challenge blocks all further
  read/interactive actions until a fresh `navigate` succeeds.
- After `click` or `submit`, the adapter enforces same-origin: the final URL
  must start with `https://en.wikipedia.org/`. Off-origin URLs kill the
  process group immediately.
- Element refs from `query`/`text`/`blockmap` results are session-scoped and
  become stale after navigation. Unknown or stale refs fail closed with a
  clear error.
- `type` values are bounded (max 1024 chars, no NUL bytes). Ref strings are
  bounded (max 256 chars) with no control characters.
- **No** cookies, authentication, JavaScript evaluation (`eval`/QuickJS),
  downloads, `POST` requests, arbitrary URL navigation, or raw JSON-RPC
  are exposed.

**Explicit residual risks:**

1. This is **NOT** SSRF-safe. The outbound HTTP request occurs before the
   final URL can be checked. Redirects, DNS rebinding, or connection-level
   attacks are not prevented.
2. The adapter runs outside Bubblewrap on the disposable SSH runner. Do not
   run on a host containing secrets, credentials, or production data.
3. Only `GET`-based form submission is supported. Any form requiring `POST`
   should fail.
4. The same-origin enforcement is a post-navigation check, not a connection-
   level filter. A malicious Wikipedia article could theoretically link to
   an off-origin page, though this would be caught and kill the session.

This path shares the base Unbrowser security constraints (fresh process per
attempt, bounded JSON-RPC lines, timeouts, result-size limits, minimal
environment, process group kill on failure). Use only with a dedicated
least-privilege SSH identity and disposable runner.

## Sensitive outputs

`.runs/` may contain prompts, trajectories, model outputs, generated files,
absolute paths, benchmark data, and learned artifacts. It is ignored by Git but
is not encrypted or automatically deleted. Inspect it before copying or
archiving the repository.

Do not load untrusted PyTorch artifact files. Treat `.pt`, `.pth`, pickle, and
similar model files as executable/untrusted input even when a loading API
offers safer modes.

## Reporting a vulnerability

For sensitive reports, use GitHub private vulnerability reporting if it is
enabled for the repository. Do not include credentials, private datasets, or
exploit details in a public issue. Non-sensitive hardening suggestions may use
the normal issue tracker.

## Support status

Only the current default branch is maintained. This project is research
software and currently provides no security SLA.
