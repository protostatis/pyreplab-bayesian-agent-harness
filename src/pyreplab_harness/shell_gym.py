"""Deterministic shell/filesystem transformation task family.

The generated initial workspace contains a flat ``incoming/`` directory whose
files must be classified into per-category directories, renamed by an explicit
rule, deduplicated by content hash, and summarized in a ``manifest.json`` at the
workspace root. The private oracle (expected semantic tree, content hashes, and
duplicate mapping) lives under ``<task>/private/`` and is never mounted in an
attempt workspace.

The verifier measures only the final submitted workspace: the set of relative
paths, per-file content hashes, required absence of source files, and the
semantic content of the manifest. Command history is irrelevant.

Layout compatibility: tasks and attempts are written in the exact locations the
generic ``artifact_gym`` helpers expect, so ``artifact_gym.prepare_attempt``,
``artifact_gym.load_task``, and ``artifact_gym.load_attempt`` work unchanged.
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from datetime import UTC
except ImportError:  # Python < 3.11 compatibility
    UTC = timezone.utc

from .artifact_gym import _attempt_dir, _root, _task_dir, load_attempt, load_task
from .contracts import AttemptRecord, TaskSpec, VerificationResult
from .io_utils import read_json, write_json

GENERATOR_VERSION = "shell-file-classify-v2"
TEMPLATE_ID = "file-classify-v2"
VERIFIER_ID = "shell-tree-semantic"
VERIFIER_VERSION = "1"
DIFFICULTIES = {"easy", "medium", "hard"}
MANIFEST_SCHEMA_VERSION = 1

CATEGORIES = ("image", "note", "data", "script")

CATEGORY_META: dict[str, dict[str, str]] = {
    "image": {"dir": "images", "stem": "image", "ext": ".img"},
    "note": {"dir": "notes", "stem": "note", "ext": ".txt"},
    "data": {"dir": "data", "stem": "data", "ext": ".csv"},
    "script": {"dir": "scripts", "stem": "script", "ext": ".sh"},
}

_NAME_WORDS: dict[str, tuple[str, ...]] = {
    "image": ("photo", "pic", "shot", "frame", "still", "render", "view", "snap", "grab", "take"),
    "note": ("memo", "jot", "note", "draft", "scrap", "blurb", "scribble", "remark", "aside", "margin"),
    "data": ("table", "sheet", "series", "ledger", "roster", "index", "set", "grid", "log", "tally"),
    "script": ("run", "task", "job", "step", "phase", "stage", "pipe", "flow", "init", "sync"),
}

_NOTE_WORDS = (
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
    "mike",
    "november",
    "oscar",
    "papa",
)

_ECHO_WORDS = ("ok", "done", "start", "finish", "retry", "wait")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Files the verifier tolerates at the workspace root: the required output
# manifest and the task handout copied into every attempt workspace.
_IGNORED_FILES = {"manifest.json", "TASK.md"}


def _difficulty_shape(difficulty: str) -> dict[str, Any]:
    if difficulty == "easy":
        return {
            "base": 2,
            "extra_p": 0.0,
            "dup_groups": 1,
            "misfiled": 0,
            "data_rows": 3,
            "image_min": 32,
            "image_max": 64,
            "note_lines": 2,
        }
    if difficulty == "medium":
        return {
            "base": 3,
            "extra_p": 0.4,
            "dup_groups": 2,
            "misfiled": 2,
            "data_rows": 8,
            "image_min": 128,
            "image_max": 256,
            "note_lines": 4,
        }
    if difficulty == "hard":
        return {
            "base": 5,
            "extra_p": 0.8,
            "dup_groups": 3,
            "misfiled": 4,
            "data_rows": 16,
            "image_min": 512,
            "image_max": 1024,
            "note_lines": 8,
        }
    raise ValueError(f"unsupported difficulty: {difficulty!r}")


def _classify_content(content: bytes) -> str:
    first_line = content.split(b"\n", 1)[0]
    if first_line.startswith(b"IMGDATA"):
        return "image"
    if first_line.startswith(b"NOTE:"):
        return "note"
    if first_line == b"key,value":
        return "data"
    if first_line == b"#!/bin/sh":
        return "script"
    raise ValueError("content has no recognized signature")


def _make_content(
    rng: random.Random,
    category: str,
    shape: dict[str, Any],
    nonce_text: str,
) -> bytes:
    nonce = f"#nonce {nonce_text}"
    if category == "image":
        size = rng.randint(shape["image_min"], shape["image_max"])
        return b"IMGDATA " + nonce.encode("utf-8") + b"\n" + rng.randbytes(size)
    if category == "note":
        title = " ".join(rng.choice(_NOTE_WORDS) for _ in range(3))
        lines = [nonce]
        for _ in range(shape["note_lines"]):
            lines.append(" ".join(rng.choice(_NOTE_WORDS) for _ in range(rng.randint(3, 8))))
        return (f"NOTE: {title}\n" + "\n".join(lines) + "\n").encode("utf-8")
    if category == "data":
        rows = [nonce]
        for index in range(shape["data_rows"]):
            rows.append(f"key_{index + 1:03d},{rng.randint(0, 99999)}")
        return ("key,value\n" + "\n".join(rows) + "\n").encode("utf-8")
    if category == "script":
        body = [nonce]
        for _ in range(rng.randint(2, 4)):
            body.append(f"echo {rng.choice(_ECHO_WORDS)}")
        return ("#!/bin/sh\n" + "\n".join(body) + "\n").encode("utf-8")
    raise ValueError(f"unsupported category: {category!r}")


def _fresh_name(rng: random.Random, category: str, used: set[str], wrong: bool) -> str:
    pool = _NAME_WORDS[category]
    ext = CATEGORY_META[category]["ext"]
    if wrong:
        ext = CATEGORY_META[rng.choice([c for c in CATEGORIES if c != category])]["ext"]
    for _ in range(100):
        first, second = rng.sample(pool, 2)
        name = f"{first}_{second}{ext}"
        if name not in used:
            used.add(name)
            return name
    raise RuntimeError(f"name pool exhausted for category {category!r}")


def _expected_semantics(incoming: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in incoming:
        digest = hashlib.sha256(item["content"]).hexdigest()
        groups.setdefault(digest, []).append(item)

    kept_by_category: dict[str, list[dict[str, Any]]] = {}
    duplicate_groups: list[dict[str, Any]] = []
    for digest, items in groups.items():
        items.sort(key=lambda item: item["name"])
        keeper = items[0]
        category = _classify_content(keeper["content"])
        kept_by_category.setdefault(category, []).append(keeper)
        if len(items) > 1:
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "original_names": sorted(item["name"] for item in items),
                    "_keeper": keeper["name"],
                }
            )

    expected_files: dict[str, dict[str, Any]] = {}
    path_by_original: dict[str, str] = {}
    counts: dict[str, int] = {}
    for category, keepers in kept_by_category.items():
        keepers.sort(key=lambda item: item["name"])
        counts[category] = len(keepers)
        meta = CATEGORY_META[category]
        for index, keeper in enumerate(keepers, start=1):
            rel = f"{meta['dir']}/{meta['stem']}_{index:02d}{meta['ext']}"
            expected_files[rel] = {
                "category": category,
                "sha256": hashlib.sha256(keeper["content"]).hexdigest(),
                "size": len(keeper["content"]),
                "original_name": keeper["name"],
                "content_b64": base64.b64encode(keeper["content"]).decode("ascii"),
            }
            path_by_original[keeper["name"]] = rel

    duplicates = [
        {
            "kept": path_by_original[group["_keeper"]],
            "sha256": group["sha256"],
            "original_names": group["original_names"],
        }
        for group in sorted(duplicate_groups, key=lambda group: group["_keeper"])
    ]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "files": expected_files,
        "duplicates": duplicates,
        "counts": counts,
    }


def _build_dataset(
    seed: int,
    difficulty: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shape = _difficulty_shape(difficulty)
    rng = random.Random(seed)
    used_names: set[str] = set()
    incoming: list[dict[str, Any]] = []
    nonce_counter = 0

    def next_nonce(category: str) -> str:
        nonlocal nonce_counter
        nonce_counter += 1
        return f"{seed}-{category}-{nonce_counter:04d}"

    for category in CATEGORIES:
        count = shape["base"] + (1 if rng.random() < shape["extra_p"] else 0)
        for _ in range(count):
            content = _make_content(rng, category, shape, next_nonce(category))
            incoming.append({"name": _fresh_name(rng, category, used_names, wrong=False), "content": content})

    for _ in range(shape["dup_groups"]):
        category = rng.choice(CATEGORIES)
        members = rng.randint(2, 3)
        content = _make_content(rng, category, shape, next_nonce(category))
        for _ in range(members):
            incoming.append({"name": _fresh_name(rng, category, used_names, wrong=False), "content": content})

    for _ in range(shape["misfiled"]):
        category = rng.choice(CATEGORIES)
        content = _make_content(rng, category, shape, next_nonce(category))
        incoming.append({"name": _fresh_name(rng, category, used_names, wrong=True), "content": content})

    return incoming, _expected_semantics(incoming)


def generate_shell_task(root: str | Path, seed: int, difficulty: str = "medium") -> TaskSpec:
    """Generate a deterministic filesystem transformation task under ``root``."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {sorted(DIFFICULTIES)}")
    root_path = _root(root)
    task_id = f"shell-{TEMPLATE_ID}-{difficulty}-{seed}"
    task_path = _task_dir(root_path, task_id)
    manifest_path = task_path / "task.json"
    if manifest_path.exists():
        return TaskSpec.from_dict(read_json(manifest_path))

    initial = task_path / "initial"
    private = task_path / "private"
    initial.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)

    incoming, oracle = _build_dataset(seed, difficulty)

    incoming_dir = initial / "incoming"
    incoming_dir.mkdir(parents=True, exist_ok=False)
    for item in incoming:
        (incoming_dir / item["name"]).write_bytes(item["content"])

    contract = (
        "Classify every file under incoming/ into exactly one of four categories. "
        "By extension: .img is image, .txt is note, .csv is data, .sh is script. "
        "By content signature: image files start with IMGDATA, notes start with NOTE:, "
        "data files start with a key,value header line, scripts start with #!/bin/sh. "
        "Some files are misfiled (extension disagrees with content); classify those by content.",
        "Move every file into its category directory at the workspace root: "
        "images/, notes/, data/, scripts/.",
        "Deduplicate by exact content (same sha256): keep exactly one copy per duplicate group - "
        "the copy whose original filename sorts first (byte ascending) - and remove every other copy.",
        "Rename each kept file within its category to {category}_{NN}{ext} where NN is its "
        "position 01, 02, ... when kept files are ordered by original filename (byte ascending). "
        "Use the category's canonical extension regardless of the original filename: image=.img, "
        "note=.txt, data=.csv, script=.sh. Category stems are image, note, data, script.",
        "Leave incoming/ empty or remove it.",
        "Write manifest.json at the workspace root: schema_version 1; files: one entry per kept "
        "file with path, category, sha256, size, original_name; duplicates: one entry per duplicate "
        "group with kept, sha256, original_names; category_counts: {category: count} for all four "
        "categories.",
        "Do not leave temporary or helper files. The final workspace may contain only TASK.md, "
        "manifest.json, the four category directories, and their expected kept files.",
    )
    prompt = (
        "Complete the filesystem transformation task in the isolated /workspace directory.\n\n"
        + "\n".join(f"- {item}" for item in contract)
        + "\n\nThe task is complete only when /workspace/manifest.json exists, incoming/ contains "
        "no files, the transformed tree satisfies every rule, and manifest.json accurately "
        "describes the final tree with correct content hashes."
    )
    (initial / "TASK.md").write_text(prompt + "\n", encoding="utf-8")
    write_json(private / "oracle.json", oracle)

    spec = TaskSpec(
        id=task_id,
        family="shell",
        template_id=TEMPLATE_ID,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        difficulty=difficulty,
        prompt=prompt,
        contract=contract,
        public_metadata={
            "input_dir": "incoming",
            "categories": list(CATEGORIES),
            "category_directories": {category: meta["dir"] for category, meta in CATEGORY_META.items()},
            "file_count": len(incoming),
            "kept_file_count": len(oracle["files"]),
            "duplicate_groups": len(oracle["duplicates"]),
            "misfiled_count": _difficulty_shape(difficulty)["misfiled"],
            "required_output": "manifest.json",
        },
        workspace_ref=str(initial),
        verifier_ref=str(private / "oracle.json"),
    )
    write_json(manifest_path, spec.to_dict())
    return spec


def _verdict(success: bool, failure_code: str | None, **diagnostics: Any) -> VerificationResult:
    return VerificationResult(
        success=success,
        verifier_id=VERIFIER_ID,
        verifier_version=VERIFIER_VERSION,
        failure_code=failure_code,
        diagnostics=diagnostics,
    )


def _file_entry_schema_error(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return "files entry is not an object"
    for field in ("path", "category", "sha256", "size", "original_name"):
        if field not in entry:
            return f"files entry missing field {field!r}"
    if not all(isinstance(entry[field], str) for field in ("path", "category", "sha256", "original_name")):
        return "files entry has a non-string field"
    if not isinstance(entry["size"], int) or isinstance(entry["size"], bool):
        return "files entry size must be an integer"
    if entry["category"] not in CATEGORIES:
        return f"files entry has unknown category {entry['category']!r}"
    if not _SHA256_RE.fullmatch(entry["sha256"]):
        return "files entry sha256 must be a 64-character lowercase hex digest"
    return None


def _duplicate_entry_schema_error(entry: Any) -> str | None:
    if not isinstance(entry, dict):
        return "duplicates entry is not an object"
    for field in ("kept", "sha256", "original_names"):
        if field not in entry:
            return f"duplicates entry missing field {field!r}"
    if not isinstance(entry["kept"], str):
        return "duplicates entry kept must be a string"
    if not isinstance(entry["sha256"], str) or not _SHA256_RE.fullmatch(entry["sha256"]):
        return "duplicates entry sha256 must be a 64-character lowercase hex digest"
    if not isinstance(entry["original_names"], list) or not all(
        isinstance(name, str) for name in entry["original_names"]
    ):
        return "duplicates entry original_names must be a list of strings"
    return None


def _manifest_schema_error(manifest: Any) -> str | None:
    if not isinstance(manifest, dict):
        return "manifest is not a JSON object"
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        return "schema_version must be 1"
    for key in ("files", "duplicates", "category_counts"):
        if key not in manifest:
            return f"missing required key {key!r}"

    files = manifest["files"]
    if not isinstance(files, list):
        return "files must be a list"
    for entry in files:
        error = _file_entry_schema_error(entry)
        if error is not None:
            return error
    if len(files) != len({entry["path"] for entry in files}):
        return "files contains duplicate paths"

    duplicates = manifest["duplicates"]
    if not isinstance(duplicates, list):
        return "duplicates must be a list"
    for entry in duplicates:
        error = _duplicate_entry_schema_error(entry)
        if error is not None:
            return error
    if len(duplicates) != len({entry["kept"] for entry in duplicates}):
        return "duplicates contains duplicate kept paths"

    counts = manifest["category_counts"]
    if not isinstance(counts, dict):
        return "category_counts must be an object"
    for category, value in counts.items():
        if category not in CATEGORIES or not isinstance(value, int) or isinstance(value, bool):
            return f"invalid category count entry {category!r}: {value!r}"
    return None


def _manifest_semantics_error(manifest: dict[str, Any], oracle: dict[str, Any]) -> str | None:
    expected_files = oracle["files"]
    agent_files = {entry["path"]: entry for entry in manifest["files"]}

    if set(agent_files) != set(expected_files):
        return "files paths do not match the expected set"
    for rel, expected in expected_files.items():
        entry = agent_files[rel]
        if entry["category"] != expected["category"]:
            return f"{rel}: category mismatch"
        if entry["sha256"] != expected["sha256"]:
            return f"{rel}: sha256 mismatch"
        if entry["size"] != expected["size"]:
            return f"{rel}: size mismatch"
        if entry["original_name"] != expected["original_name"]:
            return f"{rel}: original_name mismatch"

    expected_dups = sorted(oracle["duplicates"], key=lambda group: group["kept"])
    agent_dups = sorted(manifest["duplicates"], key=lambda group: group["kept"])
    if len(agent_dups) != len(expected_dups):
        return "duplicates count mismatch"
    for agent_group, expected_group in zip(agent_dups, expected_dups):
        if agent_group["kept"] != expected_group["kept"]:
            return f"duplicates kept path mismatch: {agent_group['kept']!r}"
        if agent_group["sha256"] != expected_group["sha256"]:
            return f"duplicates sha256 mismatch for {expected_group['kept']}"
        if sorted(agent_group["original_names"]) != sorted(expected_group["original_names"]):
            return f"duplicates original_names mismatch for {expected_group['kept']}"

    if manifest["category_counts"] != oracle["counts"]:
        return "category_counts mismatch"
    return None


def _evaluate(workspace: Path, oracle: dict[str, Any]) -> VerificationResult:
    expected_files = oracle["files"]
    expected_paths = set(expected_files)

    actual: dict[str, Path] = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_file():
            rel = path.relative_to(workspace).as_posix()
            if rel not in _IGNORED_FILES:
                actual[rel] = path
    actual_paths = set(actual)

    missing = sorted(expected_paths - actual_paths)
    extra = sorted(actual_paths - expected_paths)
    if missing or extra:
        return _verdict(False, "file_set_mismatch", missing=missing, extra=extra)

    for rel in sorted(expected_paths):
        digest = hashlib.sha256(actual[rel].read_bytes()).hexdigest()
        if digest != expected_files[rel]["sha256"]:
            return _verdict(
                False,
                "content_mismatch",
                path=rel,
                expected=expected_files[rel]["sha256"],
                actual=digest,
            )

    manifest_path = workspace / "manifest.json"
    if not manifest_path.exists():
        return _verdict(False, "missing_manifest", required="manifest.json")
    try:
        manifest = read_json(manifest_path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        return _verdict(False, "invalid_manifest", error=str(error))

    schema_error = _manifest_schema_error(manifest)
    if schema_error is not None:
        return _verdict(False, "invalid_manifest", error=schema_error)

    semantic_error = _manifest_semantics_error(manifest, oracle)
    if semantic_error is not None:
        return _verdict(False, "manifest_mismatch", error=semantic_error)

    return _verdict(
        True,
        None,
        file_count=len(expected_files),
        duplicate_groups=len(oracle["duplicates"]),
    )


def _write_verification(root_path: Path, attempt: AttemptRecord, result: VerificationResult) -> None:
    attempt_path = _attempt_dir(root_path, attempt.attempt_id)
    verification_path = attempt_path / "verification.json"
    write_json(verification_path, result.to_dict())
    updated = replace(
        attempt,
        status="verified",
        verification_ref=str(verification_path),
    )
    write_json(attempt_path / "attempt.json", updated.to_dict())


def verify_shell_attempt(root: str | Path, task_id: str, attempt_id: str) -> VerificationResult:
    """Verify a submitted shell attempt against the task's private oracle."""
    root_path = _root(root)
    spec = load_task(root_path, task_id)
    attempt = load_attempt(root_path, attempt_id)
    if attempt.task_id != spec.id:
        raise ValueError("attempt does not belong to task")

    workspace = Path(attempt.workspace_ref)
    oracle = read_json(Path(spec.verifier_ref))
    result = _evaluate(workspace, oracle)
    _write_verification(root_path, attempt, result)
    return result


__all__ = [
    "CATEGORIES",
    "CATEGORY_META",
    "DIFFICULTIES",
    "GENERATOR_VERSION",
    "MANIFEST_SCHEMA_VERSION",
    "TEMPLATE_ID",
    "VERIFIER_ID",
    "VERIFIER_VERSION",
    "generate_shell_task",
    "verify_shell_attempt",
]
