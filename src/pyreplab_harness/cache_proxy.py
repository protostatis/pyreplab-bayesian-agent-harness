"""Loopback-only instrumentation proxy for the isolated cache canary.

The proxy never persists request or response bodies. It hashes the incoming
request, injects only the frozen cache mode and slot identity, streams the
upstream response, and records server-originated cache/timing mechanics.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import ipaddress
import json
import math
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cache_mechanics import canonical_receipt_hash

CACHE_PROXY_RECEIPT_SCHEMA_VERSION = "pyreplab-cache-proxy-turn-receipt-v1"

_ALLOWED_PATHS = frozenset(
    {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/responses",
    }
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


def _require_loopback(host: str, label: str) -> None:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError(f"{label} must be a numeric loopback address") from error
    if not address.is_loopback:
        raise ValueError(f"{label} must be loopback-only")


def _non_negative_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    )


def _non_negative_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _response_payloads(body: bytes) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    stripped = body.strip()
    if not stripped:
        return payloads
    try:
        value = json.loads(stripped)
    except (json.JSONDecodeError, UnicodeDecodeError):
        value = None
    if isinstance(value, Mapping):
        payloads.append(value)
    for line in body.splitlines():
        if not line.startswith(b"data:"):
            continue
        data = line[len(b"data:") :].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(value, Mapping):
            payloads.append(value)
    return payloads


def _server_mechanics(body: bytes, cache_mode: str) -> dict[str, Any]:
    timings: Mapping[str, Any] | None = None
    usage: Mapping[str, Any] | None = None
    for payload in _response_payloads(body):
        candidate = payload.get("timings")
        if isinstance(candidate, Mapping):
            timings = candidate
        candidate = payload.get("usage")
        if isinstance(candidate, Mapping):
            usage = candidate

    required = {
        "cache_n": _non_negative_integer,
        "prompt_n": _non_negative_integer,
        "prompt_ms": _non_negative_number,
        "predicted_n": _non_negative_integer,
        "predicted_ms": _non_negative_number,
    }
    invalidation_codes: list[str] = []
    observed: dict[str, Any] = {}
    for field, validator in required.items():
        value = timings.get(field) if isinstance(timings, Mapping) else None
        if validator(value):
            observed[field] = value
        else:
            observed[field] = None
            invalidation_codes.append(f"server_timing_missing_or_invalid:{field}")
    if cache_mode == "off" and observed.get("cache_n") not in {None, 0}:
        invalidation_codes.append("cache_off_reported_reused_prefix")

    cached_tokens = None
    if isinstance(usage, Mapping):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, Mapping):
            value = details.get("cached_tokens")
            if _non_negative_integer(value):
                cached_tokens = value
            elif "cached_tokens" in details:
                invalidation_codes.append("server_usage_cached_tokens_invalid")
    if cached_tokens is None:
        invalidation_codes.append("server_usage_cached_tokens_missing_or_invalid")
    elif observed.get("cache_n") is not None and cached_tokens != observed["cache_n"]:
        invalidation_codes.append("server_cache_counter_mismatch")

    return {
        "timings": observed,
        "usage_cached_tokens": cached_tokens,
        "mechanics_valid": not invalidation_codes,
        "invalidation_codes": sorted(set(invalidation_codes)),
    }


@dataclass(frozen=True)
class CacheProxyContext:
    attempt_id: str
    panel_id: str
    pair_id: str
    sampling_seed: int
    cache_runtime_receipt_hash: str

    def __post_init__(self) -> None:
        for label, value in (
            ("attempt_id", self.attempt_id),
            ("panel_id", self.panel_id),
            ("pair_id", self.pair_id),
        ):
            if not value or len(value) > 256 or any(ord(char) < 32 for char in value):
                raise ValueError(f"invalid proxy context {label}")
        if isinstance(self.sampling_seed, bool) or not isinstance(
            self.sampling_seed, int
        ):
            raise ValueError("sampling_seed must be an integer")
        if len(self.cache_runtime_receipt_hash) != 64:
            raise ValueError("cache runtime receipt hash is invalid")
        if any(char not in "0123456789abcdef" for char in self.cache_runtime_receipt_hash):
            raise ValueError("cache runtime receipt hash must be lowercase hex")


@dataclass(frozen=True)
class CacheProxyConfig:
    bind_host: str
    bind_port: int
    upstream_host: str
    upstream_port: int
    cache_mode: str
    slot_id: int = 0
    max_requests: int = 32
    max_request_bytes: int = 8 * 1024 * 1024
    max_capture_bytes: int = 16 * 1024 * 1024
    upstream_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        _require_loopback(self.bind_host, "bind_host")
        _require_loopback(self.upstream_host, "upstream_host")
        for label, value in (
            ("bind_port", self.bind_port),
            ("upstream_port", self.upstream_port),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 65535:
                raise ValueError(f"{label} must be a valid port")
        if self.cache_mode not in {"off", "on"}:
            raise ValueError("cache_mode must be off or on")
        if self.slot_id != 0:
            raise ValueError("the Stage 1 single-slot canary requires slot_id=0")
        if self.max_requests < 1 or self.max_request_bytes < 1 or self.max_capture_bytes < 1:
            raise ValueError("proxy limits must be positive")
        if self.upstream_timeout_seconds <= 0:
            raise ValueError("upstream timeout must be positive")


class CacheProxyRecorder:
    def __init__(
        self,
        path: Path,
        context: CacheProxyContext,
        cache_mode: str,
        max_requests: int,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.context = context
        self.cache_mode = cache_mode
        self.max_requests = max_requests
        self._lock = threading.Lock()
        self._next_index = 1
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("x", encoding="utf-8") as handle:
            handle.flush()
            os.fsync(handle.fileno())

    def reserve(self) -> int | None:
        with self._lock:
            if self._next_index > self.max_requests:
                return None
            value = self._next_index
            self._next_index += 1
            return value

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        receipt = {
            **payload,
            "receipt_hash": canonical_receipt_hash(payload),
        }
        serialized = json.dumps(
            receipt,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        return receipt


def validate_cache_proxy_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("schema_version") != CACHE_PROXY_RECEIPT_SCHEMA_VERSION:
        raise ValueError("unsupported cache proxy receipt schema")
    observed = receipt.get("receipt_hash")
    if not isinstance(observed, str) or len(observed) != 64:
        raise ValueError("cache proxy receipt hash is invalid")
    payload = {key: value for key, value in receipt.items() if key != "receipt_hash"}
    if canonical_receipt_hash(payload) != observed:
        raise ValueError("cache proxy receipt hash mismatch")
    if receipt.get("raw_request_persisted") is not False:
        raise ValueError("cache proxy receipt persisted a raw request")
    if receipt.get("raw_response_persisted") is not False:
        raise ValueError("cache proxy receipt persisted a raw response")
    if receipt.get("authorization_header_persisted") is not False:
        raise ValueError("cache proxy receipt persisted an authorization header")
    codes = receipt.get("invalidation_codes")
    if not isinstance(codes, list) or any(not isinstance(code, str) for code in codes):
        raise ValueError("cache proxy invalidation codes must be strings")
    expected_valid = not codes and receipt.get("response_status") == 200
    if receipt.get("mechanics_valid") is not expected_valid:
        raise ValueError("cache proxy mechanics validity contradicts receipt state")
    mechanics = receipt.get("server_mechanics")
    if not isinstance(mechanics, Mapping):
        raise ValueError("cache proxy server mechanics must be an object")
    timings = mechanics.get("timings")
    if not isinstance(timings, Mapping):
        raise ValueError("cache proxy timing fields must be an object")
    for field, validator in {
        "cache_n": _non_negative_integer,
        "prompt_n": _non_negative_integer,
        "prompt_ms": _non_negative_number,
        "predicted_n": _non_negative_integer,
        "predicted_ms": _non_negative_number,
    }.items():
        value = timings.get(field)
        if receipt.get("mechanics_valid") is True and not validator(value):
            raise ValueError(f"cache proxy valid receipt omitted {field}")


class _CacheProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        config: CacheProxyConfig,
        recorder: CacheProxyRecorder,
    ) -> None:
        self.cache_config = config
        self.cache_recorder = recorder
        super().__init__((config.bind_host, config.bind_port), CacheProxyHandler)


class CacheProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    server_version = "pyreplab-cache-proxy/1"

    @property
    def proxy_server(self) -> _CacheProxyServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _reject(self, status: int, code: str) -> None:
        body = json.dumps({"error": code}, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self) -> None:
        self._reject(405, "model_proxy_post_only")

    def do_POST(self) -> None:
        config = self.proxy_server.cache_config
        recorder = self.proxy_server.cache_recorder
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        invalidation_codes: list[str] = []
        parsed_path = urlsplit(self.path)
        if parsed_path.path not in _ALLOWED_PATHS or parsed_path.query:
            self._reject(404, "unsupported_model_endpoint")
            return
        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length or "")
        except ValueError:
            body_length = -1
        if body_length < 0 or body_length > config.max_request_bytes:
            self._reject(413, "invalid_request_size")
            return
        incoming = self.rfile.read(body_length)
        if len(incoming) != body_length:
            self._reject(400, "truncated_request_body")
            return
        try:
            request_json = json.loads(incoming)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._reject(400, "request_body_not_json")
            return
        if not isinstance(request_json, dict):
            self._reject(400, "request_body_not_object")
            return
        if "cache_prompt" in request_json or "id_slot" in request_json:
            self._reject(409, "proxy_controlled_field_present")
            return
        request_index = recorder.reserve()
        if request_index is None:
            self._reject(429, "proxy_request_budget_exhausted")
            return

        incoming_hash = hashlib.sha256(incoming).hexdigest()
        logical_request_hash = canonical_receipt_hash(request_json)
        forwarded_json = {
            **request_json,
            "cache_prompt": config.cache_mode == "on",
            "id_slot": config.slot_id,
        }
        forwarded = json.dumps(
            forwarded_json,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        forwarded_hash = hashlib.sha256(forwarded).hexdigest()
        headers = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Accept": self.headers.get("Accept", "application/json"),
            "Content-Length": str(len(forwarded)),
        }
        response_status: int | None = None
        response_started = False
        response_hash: str | None = None
        response_bytes = 0
        first_byte_seconds: float | None = None
        captured = bytearray()
        mechanics = {
            "timings": {},
            "usage_cached_tokens": None,
            "mechanics_valid": False,
            "invalidation_codes": ["upstream_response_unavailable"],
        }
        connection = http.client.HTTPConnection(
            config.upstream_host,
            config.upstream_port,
            timeout=config.upstream_timeout_seconds,
        )
        response_digest = hashlib.sha256()
        try:
            connection.request("POST", parsed_path.path, body=forwarded, headers=headers)
            response = connection.getresponse()
            response_status = response.status
            self.send_response(response.status, response.reason)
            response_started = True
            for name, value in response.getheaders():
                if name.casefold() in _HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(name, value)
            self.send_header("Connection", "close")
            self.end_headers()
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                if first_byte_seconds is None:
                    first_byte_seconds = time.monotonic() - started
                response_digest.update(chunk)
                response_bytes += len(chunk)
                if len(captured) < config.max_capture_bytes:
                    remaining = config.max_capture_bytes - len(captured)
                    captured.extend(chunk[:remaining])
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    invalidation_codes.append("downstream_disconnected")
            self.close_connection = True
            response_hash = response_digest.hexdigest()
            if response_bytes > config.max_capture_bytes:
                invalidation_codes.append("response_exceeded_mechanics_capture_limit")
            else:
                mechanics = _server_mechanics(bytes(captured), config.cache_mode)
        except (OSError, http.client.HTTPException) as error:
            invalidation_codes.append(f"upstream_transport_error:{type(error).__name__}")
            if not response_started and not self.wfile.closed:
                try:
                    self._reject(502, "upstream_transport_error")
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self.close_connection = True
        finally:
            connection.close()

        invalidation_codes.extend(mechanics["invalidation_codes"])
        context = recorder.context
        payload = {
            "schema_version": CACHE_PROXY_RECEIPT_SCHEMA_VERSION,
            "attempt_id": context.attempt_id,
            "panel_id": context.panel_id,
            "pair_id": context.pair_id,
            "sampling_seed": context.sampling_seed,
            "cache_runtime_receipt_hash": context.cache_runtime_receipt_hash,
            "provider_turn": request_index,
            "cache_mode": config.cache_mode,
            "slot_identity": config.slot_id,
            "request_path": parsed_path.path,
            "incoming_request_sha256": incoming_hash,
            "logical_request_sha256": logical_request_hash,
            "forwarded_request_sha256": forwarded_hash,
            "cache_prompt_injected": config.cache_mode == "on",
            "slot_identity_injected": True,
            "response_status": response_status,
            "response_sha256": response_hash,
            "response_bytes": response_bytes,
            "transport_first_byte_seconds": round(first_byte_seconds, 6)
            if first_byte_seconds is not None
            else None,
            "transport_total_seconds": round(time.monotonic() - started, 6),
            "server_mechanics": mechanics,
            "mechanics_valid": mechanics["mechanics_valid"]
            and not invalidation_codes
            and response_status == 200,
            "invalidation_codes": sorted(set(invalidation_codes)),
            "raw_request_persisted": False,
            "raw_response_persisted": False,
            "authorization_header_persisted": False,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
        recorder.append(payload)


def build_cache_proxy_server(
    config: CacheProxyConfig,
    context: CacheProxyContext,
    receipt_path: str | Path,
) -> ThreadingHTTPServer:
    recorder = CacheProxyRecorder(
        Path(receipt_path), context, config.cache_mode, config.max_requests
    )
    return _CacheProxyServer(config, recorder)


def proxy_mechanics_for_provider_receipt(
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    validate_cache_proxy_receipt(receipt)
    if receipt.get("mechanics_valid") is not True:
        raise ValueError("cache proxy mechanics are invalid")
    mechanics = receipt.get("server_mechanics")
    if not isinstance(mechanics, Mapping):
        raise ValueError("cache proxy server mechanics are missing")
    timings = mechanics.get("timings")
    if not isinstance(timings, Mapping):
        raise ValueError("cache proxy timing receipt is missing")
    return {
        "exact_serialized_request_sha256": receipt["incoming_request_sha256"],
        "reused_prefix_tokens": timings["cache_n"],
        "prompt_evaluation_seconds": float(timings["prompt_ms"]) / 1000.0,
        "generation_seconds": float(timings["predicted_ms"]) / 1000.0,
        "slot_identity": receipt["slot_identity"],
        "server_prompt_tokens": timings["prompt_n"],
        "server_predicted_tokens": timings["predicted_n"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pyreplab-cache-proxy")
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument("--bind-port", type=int, required=True)
    parser.add_argument("--upstream-host", default="127.0.0.1")
    parser.add_argument("--upstream-port", type=int, required=True)
    parser.add_argument("--cache-mode", choices=("off", "on"), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--panel-id", required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--cache-runtime-receipt-hash", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--max-requests", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = CacheProxyConfig(
        bind_host=args.bind_host,
        bind_port=args.bind_port,
        upstream_host=args.upstream_host,
        upstream_port=args.upstream_port,
        cache_mode=args.cache_mode,
        max_requests=args.max_requests,
    )
    context = CacheProxyContext(
        attempt_id=args.attempt_id,
        panel_id=args.panel_id,
        pair_id=args.pair_id,
        sampling_seed=args.sampling_seed,
        cache_runtime_receipt_hash=args.cache_runtime_receipt_hash,
    )
    server = build_cache_proxy_server(config, context, args.receipt_output)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE_PROXY_RECEIPT_SCHEMA_VERSION",
    "CacheProxyConfig",
    "CacheProxyContext",
    "build_cache_proxy_server",
    "proxy_mechanics_for_provider_receipt",
    "validate_cache_proxy_receipt",
]
