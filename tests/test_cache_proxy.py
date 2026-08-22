from __future__ import annotations

import http.client
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pyreplab_harness.cache_proxy import (
    CacheProxyConfig,
    CacheProxyContext,
    build_cache_proxy_server,
    proxy_mechanics_for_provider_receipt,
    validate_cache_proxy_receipt,
)


class _UpstreamHandler(BaseHTTPRequestHandler):
    received: list[dict] = []
    cache_n = 90
    include_usage = True

    def log_message(self, format: str, *args) -> None:
        return

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        self.__class__.received.append(json.loads(self.rfile.read(length)))
        payload = {
            "id": "chatcmpl-test",
            "choices": [],
            "timings": {
                "cache_n": self.__class__.cache_n,
                "prompt_n": 10,
                "prompt_ms": 125.0,
                "predicted_n": 2,
                "predicted_ms": 40.0,
            },
        }
        if self.__class__.include_usage:
            payload["usage"] = {
                "completion_tokens": 2,
                "prompt_tokens": 100,
                "total_tokens": 102,
                "prompt_tokens_details": {
                    "cached_tokens": self.__class__.cache_n
                },
            }
        body = (
            "data: "
            + json.dumps(payload, separators=(",", ":"))
            + "\n\ndata: [DONE]\n\n"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class CacheProxyTest(unittest.TestCase):
    def setUp(self) -> None:
        _UpstreamHandler.received = []
        _UpstreamHandler.cache_n = 90
        _UpstreamHandler.include_usage = True
        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
        self.upstream_thread = threading.Thread(
            target=self.upstream.serve_forever, daemon=True
        )
        self.upstream_thread.start()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(timeout=2)
        self._tmp.cleanup()

    def _run_request(
        self,
        mode: str,
        *,
        body: dict | None = None,
        max_requests: int = 4,
    ) -> tuple[int, bytes, dict, str]:
        receipt_path = self.root / f"{mode}-receipts.jsonl"
        config = CacheProxyConfig(
            bind_host="127.0.0.1",
            bind_port=0,
            upstream_host="127.0.0.1",
            upstream_port=self.upstream.server_port,
            cache_mode=mode,
            max_requests=max_requests,
        )
        context = CacheProxyContext(
            attempt_id=f"attempt-{mode}",
            panel_id="panel-1",
            pair_id="pair-1",
            sampling_seed=7,
            cache_runtime_receipt_hash="a" * 64,
        )
        proxy = build_cache_proxy_server(config, context, receipt_path)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        request_body = body or {
            "model": "gemma",
            "prompt": "SENSITIVE PROMPT",
            "seed": 7,
            "stream": True,
        }
        encoded = json.dumps(request_body).encode("utf-8")
        connection = http.client.HTTPConnection(
            "127.0.0.1", proxy.server_port, timeout=5
        )
        try:
            connection.request(
                "POST",
                "/v1/completions",
                body=encoded,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            response_body = response.read()
            status = response.status
        finally:
            connection.close()
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=2)
        text = receipt_path.read_text(encoding="utf-8")
        receipt = json.loads(text.splitlines()[0]) if text.strip() else {}
        return status, response_body, receipt, text

    def test_proxy_injects_controls_and_records_server_mechanics(self) -> None:
        status, response_body, receipt, receipt_text = self._run_request("on")
        self.assertEqual(status, 200)
        self.assertIn(b"[DONE]", response_body)
        self.assertEqual(len(_UpstreamHandler.received), 1)
        forwarded = _UpstreamHandler.received[0]
        self.assertIs(forwarded["cache_prompt"], True)
        self.assertEqual(forwarded["id_slot"], 0)
        self.assertTrue(receipt["mechanics_valid"])
        self.assertEqual(receipt["server_mechanics"]["timings"]["cache_n"], 90)
        self.assertEqual(receipt["slot_identity"], 0)
        self.assertNotIn("SENSITIVE PROMPT", receipt_text)
        self.assertNotIn("chatcmpl-test", receipt_text)
        validate_cache_proxy_receipt(receipt)

        mechanics = proxy_mechanics_for_provider_receipt(receipt)
        self.assertEqual(mechanics["reused_prefix_tokens"], 90)
        self.assertEqual(mechanics["prompt_evaluation_seconds"], 0.125)
        self.assertEqual(mechanics["generation_seconds"], 0.04)
        self.assertEqual(mechanics["server_prompt_tokens"], 10)
        self.assertEqual(mechanics["server_predicted_tokens"], 2)
        self.assertEqual(
            mechanics["exact_serialized_request_sha256"],
            receipt["incoming_request_sha256"],
        )

    def test_cache_off_rejects_server_reported_reuse(self) -> None:
        _UpstreamHandler.cache_n = 1
        _, _, receipt, _ = self._run_request("off")
        self.assertFalse(receipt["mechanics_valid"])
        self.assertIn(
            "cache_off_reported_reused_prefix", receipt["invalidation_codes"]
        )
        with self.assertRaisesRegex(ValueError, "mechanics are invalid"):
            proxy_mechanics_for_provider_receipt(receipt)

    def test_cache_off_accepts_zero_reuse(self) -> None:
        _UpstreamHandler.cache_n = 0
        _, _, receipt, _ = self._run_request("off")
        self.assertTrue(receipt["mechanics_valid"])
        self.assertIs(_UpstreamHandler.received[0]["cache_prompt"], False)

    def test_missing_openai_cached_token_usage_is_invalid(self) -> None:
        _UpstreamHandler.include_usage = False
        _, _, receipt, _ = self._run_request("on")
        self.assertFalse(receipt["mechanics_valid"])
        self.assertIn(
            "server_usage_cached_tokens_missing_or_invalid",
            receipt["invalidation_codes"],
        )

    def test_proxy_rejects_caller_controlled_cache_fields(self) -> None:
        status, _, receipt, _ = self._run_request(
            "on", body={"prompt": "x", "cache_prompt": False}
        )
        self.assertEqual(status, 409)
        self.assertEqual(receipt, {})
        self.assertEqual(_UpstreamHandler.received, [])

    def test_receipt_tampering_is_detected(self) -> None:
        _, _, receipt, _ = self._run_request("on")
        receipt["slot_identity"] = 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_cache_proxy_receipt(receipt)

    def test_non_loopback_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback-only"):
            CacheProxyConfig(
                bind_host="0.0.0.0",
                bind_port=18083,
                upstream_host="127.0.0.1",
                upstream_port=18084,
                cache_mode="on",
            )


if __name__ == "__main__":
    unittest.main()
