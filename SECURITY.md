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
