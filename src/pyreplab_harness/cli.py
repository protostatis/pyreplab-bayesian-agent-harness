from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

from .artifact_gym import (
    generate_artifact_task,
    prepare_attempt,
    record_pi_events,
    verify_artifact_attempt,
)
from .events import normalize_pi_events
from .fixture_server import FixtureServer
from .fixture_templates import TEMPLATES as _FIXTURE_TEMPLATES
from .gym_registry import FAMILIES, generate_task, verify_attempt
from .worker import add_worker_arguments, run_from_args
from .structural_probe import structural_probe
from .unbrowser_rpc import validate_interactive_url


def _emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--family", choices=FAMILIES, required=True)
    generate.add_argument("--root", required=True)
    generate.add_argument("--seed", required=True, type=int)
    generate.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    generate.add_argument(
        "--fixture-template",
        choices=_FIXTURE_TEMPLATES,
        default="single_page_extraction",
        help="fixture page template (only used when --family is unbrowser_fixture)",
    )
    generate.add_argument(
        "--task-role",
        default=None,
        help="frozen protocol role (only supported by unbrowser_fixture and routing_fixture)",
    )

    generate_artifact = subparsers.add_parser("generate-artifact")
    generate_artifact.add_argument("--root", required=True)
    generate_artifact.add_argument("--seed", required=True, type=int)
    generate_artifact.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")

    prepare = subparsers.add_parser("prepare-attempt")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--task-id", required=True)
    prepare.add_argument("--attempt-id", required=True)
    prepare.add_argument("--policy-id", required=True)
    prepare.add_argument("--policy-version", default="1")
    prepare.add_argument("--treatment-bundle-hash", default=None)
    prepare.add_argument("--treatment-registry-hash", default=None)
    prepare.add_argument("--rollout-replica", type=int, default=None)
    prepare.add_argument("--sampling-seed", type=int, default=None)
    prepare.add_argument("--pilot-manifest-hash", default=None)
    prepare.add_argument("--pilot-panel-id", default=None)

    record = subparsers.add_parser("record-events")
    record.add_argument("--root", required=True)
    record.add_argument("--attempt-id", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--family", choices=FAMILIES, required=True)
    verify.add_argument("--root", required=True)
    verify.add_argument("--task-id", required=True)
    verify.add_argument("--attempt-id", required=True)

    verify_artifact = subparsers.add_parser("verify-artifact")
    verify_artifact.add_argument("--root", required=True)
    verify_artifact.add_argument("--task-id", required=True)
    verify_artifact.add_argument("--attempt-id", required=True)

    normalize = subparsers.add_parser("normalize-events")
    normalize.add_argument("path", nargs="?", default="-")

    commitment = subparsers.add_parser("routing-commitment")
    commitment.add_argument("--url", required=True)

    worker = subparsers.add_parser("serve-worker")
    add_worker_arguments(worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        _emit(
            generate_task(
                args.family, args.root, args.seed, args.difficulty,
                fixture_template=getattr(args, "fixture_template", "single_page_extraction"),
                task_role=getattr(args, "task_role", None),
            ).to_dict()
        )
    elif args.command == "verify":
        _emit(verify_attempt(args.family, args.root, args.task_id, args.attempt_id).to_dict())
    elif args.command == "generate-artifact":
        _emit(generate_artifact_task(args.root, args.seed, args.difficulty).to_dict())
    elif args.command == "prepare-attempt":
        _emit(
            prepare_attempt(
                args.root,
                args.task_id,
                args.attempt_id,
                args.policy_id,
                args.policy_version,
                args.treatment_bundle_hash,
                args.treatment_registry_hash,
                args.rollout_replica,
                args.sampling_seed,
                args.pilot_manifest_hash,
                args.pilot_panel_id,
            ).to_dict()
        )
    elif args.command == "record-events":
        raw = sys.stdin.read()
        normalized = normalize_pi_events(raw)
        _emit(record_pi_events(args.root, args.attempt_id, raw, normalized).to_dict())
    elif args.command == "verify-artifact":
        _emit(verify_artifact_attempt(args.root, args.task_id, args.attempt_id).to_dict())
    elif args.command == "normalize-events":
        raw = sys.stdin.read() if args.path == "-" else Path(args.path).read_text(encoding="utf-8")
        _emit(normalize_pi_events(raw))
    elif args.command == "routing-commitment":
        url = validate_interactive_url(args.url, allow_fixture=True)
        if "/routing/" not in url:
            raise ValueError("routing-commitment requires an opaque routing fixture URL")
        server = FixtureServer(port=18090)
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                source = response.read(262145)
                final_url = response.geturl()
                status = response.status
        finally:
            server.stop()
        if final_url != url or status != 200:
            raise ValueError("routing fixture commitment fetch changed URL or status")
        if len(source) > 262144:
            raise ValueError("routing fixture source exceeds 262144 bytes")
        html = source.decode("utf-8")
        probe = structural_probe(html)
        _emit(
            {
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "probe_features_sha256": hashlib.sha256(
                    json.dumps(
                        probe["features"],
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "probe_receipt_sha256": hashlib.sha256(
                    json.dumps(
                        probe["receipt"],
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
    elif args.command == "serve-worker":
        return run_from_args(args)
    else:  # pragma: no cover - argparse enforces the command set.
        parser.error(f"unsupported command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
