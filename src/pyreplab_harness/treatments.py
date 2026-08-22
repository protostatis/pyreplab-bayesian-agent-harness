"""Generalised treatment registry and controlled random-policy generator.

A ``TreatmentSpec`` is an immutable, validated description of an agent policy.
Every spec carries a deterministic SHA-256 ``bundle_hash`` (from a canonical
JSON payload) and a composite ``bundle_id`` (``id@version-<hash-prefix>``).

``TreatmentRegistry`` stores a collection of specs with unique id/version and
hash constraints, provides deterministic lookup by id, id@version, bundle_id
or hash, and supports JSON serialisation with integrity verification.

The controlled random generator builds treatments from a finite, interpretable
policy grammar that varies planning, verification, execution style, and budget
while keeping a fixed safety/workspace suffix.  It samples without replacement
under a seed and rejects counts beyond the grammar size.

CLI
---
``python -m pyreplab_harness.treatments generate OUTPUT --count N --seed S``
``python -m pyreplab_harness.treatments inspect REGISTRY``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

# ---------------------------------------------------------------------------
# Policy grammar dimensions
# ---------------------------------------------------------------------------

_PLANNING = [
    (
        "direct",
        "Solve the task directly without pre-planning. Start working immediately.",
    ),
    (
        "deliberate",
        "Analyse the task requirements carefully before executing. "
        "Think through the approach, consider edge cases, then act.",
    ),
    (
        "decompose",
        "Break the task into sub-problems, solve each independently, "
        "then combine the results into a final answer.",
    ),
]

_VERIFICATION = [
    ("final", "Verify the result once at the end of execution."),
    ("incremental", "Verify each intermediate step before proceeding to the next."),
]

_EXECUTION = [
    ("single-pass", "Execute each step exactly once. Do not retry on failure."),
    (
        "retry-on-failure",
        "If a step fails, retry it once with a corrected approach before moving on.",
    ),
]

_BUDGETS: list[tuple[str, dict[str, int]]] = [
    (
        "tight",
        {
            "max_output_tokens": 1024,
            "tool_call_limit": 4,
            "command_timeout_seconds": 30,
            "wall_time_limit_seconds": 180,
        },
    ),
    (
        "moderate",
        {
            "max_output_tokens": 2048,
            "tool_call_limit": 8,
            "command_timeout_seconds": 45,
            "wall_time_limit_seconds": 360,
        },
    ),
    (
        "generous",
        {
            "max_output_tokens": 4096,
            "tool_call_limit": 12,
            "command_timeout_seconds": 60,
            "wall_time_limit_seconds": 600,
        },
    ),
]

_SAFETY_SUFFIX = (
    "\n\n---\n"
    "Safety: Always work in the assigned workspace directory. "
    "Clean up temporary files after completion. "
    "Do not modify system files or access locations outside the workspace."
)

# Grammar size = 3 × 2 × 2 × 3 = 36
_GRAMMAR_SIZE = len(_PLANNING) * len(_VERIFICATION) * len(_EXECUTION) * len(_BUDGETS)

_DEFAULT_ALLOWED_TOOLS = ("bash",)
_DEFAULT_TOOL_INTERFACE = "native_bash"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _canonical_payload(treatment: "TreatmentSpec") -> str:
    """Deterministic sorted-key JSON string of the treatment fields (no hash)."""
    return json.dumps(
        {
            "id": treatment.id,
            "version": treatment.version,
            "system_prompt": treatment.system_prompt,
            "allowed_tools": sorted(treatment.allowed_tools),
            "max_output_tokens": treatment.max_output_tokens,
            "tool_call_limit": treatment.tool_call_limit,
            "command_timeout_seconds": treatment.command_timeout_seconds,
            "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
            "tool_interface": treatment.tool_interface,
            "generator_metadata": _thaw_json(treatment.generator_metadata),
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _compute_bundle_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_bundle_id(treatment_id: str, version: str, bundle_hash: str) -> str:
    return f"{treatment_id}@{version}-{bundle_hash[:8]}"


def _validate_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"id must be str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError("id must be a non-empty string")
    stripped = value.strip()
    if len(stripped) > 128:
        raise ValueError("id must be at most 128 characters")
    if not all(c.isalnum() or c in "-_.@" for c in stripped):
        raise ValueError(f"id contains unsafe characters: {stripped!r}")
    return stripped


def _validate_version(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"version must be str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError("version must be a non-empty string")
    stripped = value.strip()
    if len(stripped) > 128:
        raise ValueError("version must be at most 128 characters")
    if not all(c.isalnum() or c in "-_.@" for c in stripped):
        raise ValueError(f"version contains unsafe characters: {stripped!r}")
    return stripped


def _validate_system_prompt(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"system_prompt must be str, got {type(value).__name__}")
    if value == "":
        return value
    if not value.strip():
        raise ValueError("system_prompt must be empty or contain non-whitespace text")
    return value


def _validate_allowed_tools(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"allowed_tools must be list or tuple, got {type(value).__name__}"
        )
    tools = tuple(str(item).strip() for item in value)
    if not tools:
        raise ValueError("allowed_tools must contain at least one tool name")
    for tool in tools:
        if not tool:
            raise ValueError("allowed_tools must not contain empty tool names")
    return tools


def _validate_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate_tool_interface(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(
            f"tool_interface must be str, got {type(value).__name__}"
        )
    if not value.strip():
        raise ValueError("tool_interface must be a non-empty string")
    return value.strip()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_generator_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(
            f"generator_metadata must be dict, got {type(value).__name__}"
        )
    try:
        # JSON round-tripping both validates and severs aliases to caller-owned
        # nested containers. NaN/Infinity are forbidden so hashes are portable.
        payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
        copied = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("generator_metadata must be finite JSON data") from error
    if not isinstance(copied, dict):  # Defensive; the input check already guarantees this.
        raise TypeError("generator_metadata must encode a JSON object")
    return _freeze_json(copied)


def _tools_signature(allowed_tools: tuple[str, ...]) -> str:
    """Deterministic comma-separated sorted tool signature for categorical use."""
    return ",".join(sorted(allowed_tools))


# ---------------------------------------------------------------------------
# TreatmentSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentSpec:
    """Immutable, validated agent-policy descriptor with a content hash.

    Fields match ``PolicySpec`` in :mod:`pyreplab_harness.contracts` with two
    additions: ``tool_interface`` (default ``"native_bash"``) and
    ``generator_metadata``.  Use :func:`to_policy_spec_kwargs` to obtain a
    dict suitable for constructing a ``PolicySpec`` without modifying
    ``contracts.py``. An exact empty ``system_prompt`` represents a treatment
    with no appended prompt overlay; whitespace-only prompts are invalid.
    """

    id: str
    version: str
    system_prompt: str
    allowed_tools: tuple[str, ...]
    max_output_tokens: int
    tool_call_limit: int
    command_timeout_seconds: int
    wall_time_limit_seconds: int
    tool_interface: str = _DEFAULT_TOOL_INTERFACE
    generator_metadata: Mapping[str, Any] = field(default_factory=dict)

    # Computed at init — set via object.__setattr__ in __post_init__.
    _bundle_hash: str = field(init=False, repr=True)
    _bundle_id: str = field(init=False, repr=True)

    def __post_init__(self) -> None:
        # Validate every field.
        object.__setattr__(self, "id", _validate_id(self.id))
        object.__setattr__(self, "version", _validate_version(self.version))
        _validate_system_prompt(self.system_prompt)
        object.__setattr__(self, "allowed_tools", _validate_allowed_tools(self.allowed_tools))
        _validate_positive_int(self.max_output_tokens, "max_output_tokens")
        _validate_positive_int(self.tool_call_limit, "tool_call_limit")
        _validate_positive_int(
            self.command_timeout_seconds, "command_timeout_seconds"
        )
        _validate_positive_int(
            self.wall_time_limit_seconds, "wall_time_limit_seconds"
        )
        object.__setattr__(
            self, "tool_interface", _validate_tool_interface(self.tool_interface)
        )
        object.__setattr__(
            self,
            "generator_metadata",
            _validate_generator_metadata(self.generator_metadata),
        )

        # Compute deterministic hash from canonical payload (excluding hash
        # itself so we can verify round-trips).
        payload = _canonical_payload(self)
        bundle_hash = _compute_bundle_hash(payload)
        bundle_id = _build_bundle_id(self.id, self.version, bundle_hash)

        object.__setattr__(self, "_bundle_hash", bundle_hash)
        object.__setattr__(self, "_bundle_id", bundle_id)

    @property
    def bundle_hash(self) -> str:
        """SHA-256 hex digest of the canonical sorted-key JSON payload."""
        return self._bundle_hash

    @property
    def bundle_id(self) -> str:
        """Composite identifier: ``id@version-<hash-prefix-8>``."""
        return self._bundle_id

    @property
    def allowed_tools_signature(self) -> str:
        """Deterministic comma-separated sorted tool labels for categorical use."""
        return _tools_signature(self.allowed_tools)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a dict that includes ``bundle_hash`` and ``bundle_id``."""
        return {
            "id": self.id,
            "version": self.version,
            "system_prompt": self.system_prompt,
            "allowed_tools": list(self.allowed_tools),
            "max_output_tokens": self.max_output_tokens,
            "tool_call_limit": self.tool_call_limit,
            "command_timeout_seconds": self.command_timeout_seconds,
            "wall_time_limit_seconds": self.wall_time_limit_seconds,
            "tool_interface": self.tool_interface,
            "generator_metadata": _thaw_json(self.generator_metadata),
            "bundle_hash": self.bundle_hash,
            "bundle_id": self.bundle_id,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, verify_hash: bool = True
    ) -> "TreatmentSpec":
        """Reconstruct from a dict, optionally verifying the content hash.

        When ``verify_hash`` is ``True`` (the default) and the dict carries a
        ``bundle_hash`` key, the reconstructed payload is re-hashed and must
        match.  A mismatch raises :class:`ValueError`.
        """
        supplied_hash = value.get("bundle_hash")
        spec = cls(
            id=str(value["id"]),
            version=str(value["version"]),
            system_prompt=str(value["system_prompt"]),
            allowed_tools=tuple(str(item) for item in value["allowed_tools"]),
            max_output_tokens=int(value["max_output_tokens"]),
            tool_call_limit=int(value["tool_call_limit"]),
            command_timeout_seconds=int(value["command_timeout_seconds"]),
            wall_time_limit_seconds=int(value["wall_time_limit_seconds"]),
            tool_interface=str(value.get("tool_interface", _DEFAULT_TOOL_INTERFACE)),
            generator_metadata=dict(value.get("generator_metadata", {})),
        )
        if verify_hash and supplied_hash is not None:
            if spec.bundle_hash != str(supplied_hash):
                raise ValueError(
                    f"bundle_hash mismatch for {spec.id}@{spec.version}: "
                    f"expected {supplied_hash!r}, computed {spec.bundle_hash!r}"
                )
        return spec


# ---------------------------------------------------------------------------
# Conversion helper (caller can later build PolicySpec without touching
# contracts.py).
# ---------------------------------------------------------------------------


def to_policy_spec_kwargs(treatment: TreatmentSpec) -> dict[str, Any]:
    """Return keyword arguments suitable for constructing a ``PolicySpec``.

    This is intentionally a plain dict so callers can pass ``**kwargs`` to
    ``PolicySpec(**to_policy_spec_kwargs(t))`` without this module importing
    ``contracts.py``.
    """
    return {
        "id": treatment.id,
        "version": treatment.version,
        "system_prompt": treatment.system_prompt,
        "allowed_tools": treatment.allowed_tools,
        "max_output_tokens": treatment.max_output_tokens,
        "tool_call_limit": treatment.tool_call_limit,
        "command_timeout_seconds": treatment.command_timeout_seconds,
        "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
    }


# ---------------------------------------------------------------------------
# model_input descriptor (treatment-specific fields for outcome_model.py)
# ---------------------------------------------------------------------------


def treatment_model_input_descriptor(
    treatment: TreatmentSpec,
    *,
    task_text: str = "",
) -> dict[str, Any]:
    """Return the treatment-specific model-input fields for the outcome model.

    The returned dict carries:
    * ``text`` — task text (optional) concatenated with system_prompt,
    * numeric fields: ``max_output_tokens``, ``tool_call_limit``,
      ``command_timeout_seconds``, ``wall_time_limit_seconds``,
    * categorical fields: ``tool_interface``, ``allowed_tools_signature``,
      ``bundle_id``,
    * ``policy_id`` and ``policy_version`` for compatibility with the existing
      outcome-model feature set.
    """
    text_parts = []
    if task_text:
        text_parts.append(task_text)
    if treatment.system_prompt:
        text_parts.append(treatment.system_prompt)
    return {
        "text": "\n\n".join(text_parts),
        "max_output_tokens": treatment.max_output_tokens,
        "tool_call_limit": treatment.tool_call_limit,
        "command_timeout_seconds": treatment.command_timeout_seconds,
        "wall_time_limit_seconds": treatment.wall_time_limit_seconds,
        "tool_interface": treatment.tool_interface,
        "allowed_tools_signature": treatment.allowed_tools_signature,
        "bundle_id": treatment.bundle_id,
        "policy_id": treatment.id,
        "policy_version": treatment.version,
    }


# ---------------------------------------------------------------------------
# TreatmentRegistry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreatmentRegistry:
    """Immutable ordered collection of ``TreatmentSpec`` with integrity checks.

    Construction validates that no two specs share the same (id, version) or
    bundle_hash.  The registry hash is a deterministic SHA-256 over the sorted
    canonical payloads of every contained spec.
    """

    treatments: tuple[TreatmentSpec, ...]

    _id_index: dict[str, list[int]] = field(init=False, repr=False)
    _id_version_index: dict[tuple[str, str], int] = field(init=False, repr=False)
    _bundle_id_index: dict[str, int] = field(init=False, repr=False)
    _hash_index: dict[str, int] = field(init=False, repr=False)
    _registry_hash: str = field(init=False, repr=True)

    def __post_init__(self) -> None:
        id_index: dict[str, list[int]] = {}
        id_version_index: dict[tuple[str, str], int] = {}
        bundle_id_index: dict[str, int] = {}
        hash_index: dict[str, int] = {}

        for idx, spec in enumerate(self.treatments):
            key = (spec.id, spec.version)
            if key in id_version_index:
                raise ValueError(
                    f"duplicate id@version in registry: {spec.id}@{spec.version}"
                )
            id_version_index[key] = idx

            if spec.bundle_hash in hash_index:
                raise ValueError(
                    f"duplicate bundle_hash in registry: {spec.bundle_hash} "
                    f"(specs {spec.id}@{spec.version} and "
                    f"{self.treatments[hash_index[spec.bundle_hash]].id}"
                    f"@{self.treatments[hash_index[spec.bundle_hash]].version})"
                )
            hash_index[spec.bundle_hash] = idx

            if spec.bundle_id in bundle_id_index:
                raise ValueError(
                    f"duplicate bundle_id in registry: {spec.bundle_id}"
                )
            bundle_id_index[spec.bundle_id] = idx

            id_index.setdefault(spec.id, []).append(idx)

        object.__setattr__(self, "_id_index", id_index)
        object.__setattr__(self, "_id_version_index", id_version_index)
        object.__setattr__(self, "_bundle_id_index", bundle_id_index)
        object.__setattr__(self, "_hash_index", hash_index)

        # Deterministic registry hash from sorted canonical payloads.
        sorted_payloads = sorted(
            _canonical_payload(spec) for spec in self.treatments
        )
        registry_hash = hashlib.sha256(
            "\n".join(sorted_payloads).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "_registry_hash", registry_hash)

    @property
    def registry_hash(self) -> str:
        """SHA-256 over the sorted canonical payloads of all treatments."""
        return self._registry_hash

    # -- lookup ---------------------------------------------------------------

    def by_id(self, treatment_id: str) -> TreatmentSpec:
        """Look up by id alone; raises when ambiguous (multiple versions)."""
        indices = self._id_index.get(str(treatment_id))
        if indices is None:
            raise KeyError(f"no treatment with id {treatment_id!r}")
        if len(indices) > 1:
            versions = sorted(
                self.treatments[i].version for i in indices
            )
            raise KeyError(
                f"ambiguous id {treatment_id!r}: found versions {versions}. "
                f"Use by_id_version() or specify id@version."
            )
        return self.treatments[indices[0]]

    def by_id_version(self, treatment_id: str, version: str) -> TreatmentSpec:
        """Look up by exact id and version."""
        key = (str(treatment_id), str(version))
        idx = self._id_version_index.get(key)
        if idx is None:
            raise KeyError(f"no treatment with id {treatment_id!r} version {version!r}")
        return self.treatments[idx]

    def by_bundle_id(self, bundle_id: str) -> TreatmentSpec:
        """Look up by composite ``bundle_id``."""
        idx = self._bundle_id_index.get(str(bundle_id))
        if idx is None:
            raise KeyError(f"no treatment with bundle_id {bundle_id!r}")
        return self.treatments[idx]

    def by_hash(self, bundle_hash: str) -> TreatmentSpec:
        """Look up by full SHA-256 ``bundle_hash``."""
        idx = self._hash_index.get(str(bundle_hash))
        if idx is None:
            raise KeyError(f"no treatment with bundle_hash {bundle_hash!r}")
        return self.treatments[idx]

    def __len__(self) -> int:
        return len(self.treatments)

    def __iter__(self):
        return iter(self.treatments)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, TreatmentSpec):
            return item.bundle_hash in self._hash_index
        if isinstance(item, str):
            return (
                item in self._hash_index
                or item in self._bundle_id_index
                or item in self._id_index
                or any(item in self._id_version_index for item in [item])
            )
        return False

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_hash": self.registry_hash,
            "treatments": [spec.to_dict() for spec in self.treatments],
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, verify_hashes: bool = True
    ) -> "TreatmentRegistry":
        treatments = tuple(
            TreatmentSpec.from_dict(item, verify_hash=verify_hashes)
            for item in value["treatments"]
        )
        registry = cls(treatments)
        supplied_hash = value.get("registry_hash")
        if verify_hashes and supplied_hash is not None:
            if registry.registry_hash != str(supplied_hash):
                raise ValueError(
                    f"registry_hash mismatch: expected {supplied_hash!r}, "
                    f"computed {registry.registry_hash!r}"
                )
        return registry

    @classmethod
    def load(cls, path: str | Path) -> "TreatmentRegistry":
        """Read a JSON registry file with full hash verification."""
        file_path = Path(path).expanduser().resolve()
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data, verify_hashes=True)

    def save(self, path: str | Path) -> None:
        """Write the registry as deterministic JSON."""
        file_path = Path(path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False
        ) + "\n"
        file_path.write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Controlled random generator: finite policy grammar
# ---------------------------------------------------------------------------


def _enumerate_grammar() -> list[dict[str, Any]]:
    """Produce every grammar combination in a deterministic order.

    Returns a list of metadata dicts, one per combination, ordered by
    (planning, verification, execution, budget).
    """
    combinations: list[dict[str, Any]] = []
    for planning_key, planning_desc in _PLANNING:
        for verification_key, verification_desc in _VERIFICATION:
            for execution_key, execution_desc in _EXECUTION:
                for budget_key, budget_params in _BUDGETS:
                    combinations.append({
                        "planning": planning_key,
                        "planning_desc": planning_desc,
                        "verification": verification_key,
                        "verification_desc": verification_desc,
                        "execution": execution_key,
                        "execution_desc": execution_desc,
                        "budget": budget_key,
                        "budget_params": dict(budget_params),
                    })
    return combinations


def _build_treatment_from_combination(
    combo: dict[str, Any], index: int, version: str
) -> TreatmentSpec:
    """Build one ``TreatmentSpec`` from grammar combination metadata."""
    treatment_id = (
        f"{combo['planning']}-{combo['verification']}-"
        f"{combo['execution']}-{combo['budget']}"
    )

    prompt_parts = [
        f"Planning: {combo['planning_desc']}",
        f"Verification: {combo['verification_desc']}",
        f"Execution: {combo['execution_desc']}",
        _SAFETY_SUFFIX,
    ]
    system_prompt = "\n\n".join(prompt_parts)

    budget = combo["budget_params"]

    return TreatmentSpec(
        id=treatment_id,
        version=version,
        system_prompt=system_prompt,
        allowed_tools=_DEFAULT_ALLOWED_TOOLS,
        max_output_tokens=budget["max_output_tokens"],
        tool_call_limit=budget["tool_call_limit"],
        command_timeout_seconds=budget["command_timeout_seconds"],
        wall_time_limit_seconds=budget["wall_time_limit_seconds"],
        tool_interface=_DEFAULT_TOOL_INTERFACE,
        generator_metadata={
            "grammar_version": "1",
            "grammar_size": _GRAMMAR_SIZE,
            "index": index,
            "planning": combo["planning"],
            "verification": combo["verification"],
            "execution": combo["execution"],
            "budget": combo["budget"],
        },
    )


def generate_treatments(
    count: int,
    seed: int,
    *,
    version: str = "1",
) -> list[TreatmentSpec]:
    """Sample ``count`` treatments without replacement from the policy grammar.

    The full grammar is enumerated in a deterministic order, then shuffled
    with ``random.Random(seed)`` to produce a stable permutation.  The first
    ``count`` entries are returned.  Calls with the same (count, seed) are
    reproducible; calls with different ``count`` values and the same seed
    return prefixes of the same full permutation.

    Raises :class:`ValueError` when ``count`` exceeds the grammar size of
    :data:`_GRAMMAR_SIZE` (``36``).
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > _GRAMMAR_SIZE:
        raise ValueError(
            f"count {count} exceeds grammar size {_GRAMMAR_SIZE}"
        )

    combinations = _enumerate_grammar()
    rng = random.Random(seed)
    rng.shuffle(combinations)

    return [
        _build_treatment_from_combination(combinations[i], i, version)
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyreplab-harness-treatments",
        description="Treatment registry and controlled random-policy generator.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate treatment registry")
    generate_parser.add_argument("output", help="output JSON path for the registry")
    generate_parser.add_argument(
        "--count", "-n", type=int, required=True,
        help="number of treatments to sample (max %d)" % _GRAMMAR_SIZE,
    )
    generate_parser.add_argument(
        "--seed", "-s", type=int, default=42, help="random seed (default: 42)"
    )
    generate_parser.add_argument(
        "--version", default="1", help="treatment version string (default: 1)"
    )

    inspect_parser = subparsers.add_parser("inspect", help="inspect a treatment registry")
    inspect_parser.add_argument("registry", help="path to a registry JSON file")

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    treatments = generate_treatments(args.count, args.seed, version=args.version)
    registry = TreatmentRegistry(tuple(treatments))
    registry.save(args.output)
    print(
        json.dumps(
            {
                "count": len(treatments),
                "seed": args.seed,
                "version": args.version,
                "registry_hash": registry.registry_hash,
                "output": str(Path(args.output).expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        ),
        file=sys.stderr,
    )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    registry = TreatmentRegistry.load(args.registry)
    lines: list[str] = [
        f"registry: {Path(args.registry).expanduser().resolve()}",
        f"registry_hash: {registry.registry_hash}",
        f"treatments: {len(registry.treatments)}",
        "",
    ]
    for spec in registry.treatments:
        lines.append(
            f"  {spec.bundle_id}"
            f"  hash={spec.bundle_hash}"
            f"  tools={spec.allowed_tools_signature}"
            f"  interface={spec.tool_interface}"
            f"  max_tokens={spec.max_output_tokens}"
            f"  tool_limit={spec.tool_call_limit}"
            f"  cmd_timeout={spec.command_timeout_seconds}s"
            f"  wall_timeout={spec.wall_time_limit_seconds}s"
        )
        meta = spec.generator_metadata
        if meta:
            lines.append(
                f"    planning={meta.get('planning')}"
                f"  verification={meta.get('verification')}"
                f"  execution={meta.get('execution')}"
                f"  budget={meta.get('budget')}"
            )
    print("\n".join(lines))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "generate":
        return _cmd_generate(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
