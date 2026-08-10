from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class TaskSpec:
    id: str
    family: str
    template_id: str
    generator_version: str
    seed: int
    difficulty: str
    prompt: str
    contract: tuple[str, ...]
    public_metadata: Mapping[str, Any]
    workspace_ref: str
    verifier_ref: str
    split: str = "unassigned"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["contract"] = list(self.contract)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        return cls(
            id=str(value["id"]),
            family=str(value["family"]),
            template_id=str(value["template_id"]),
            generator_version=str(value["generator_version"]),
            seed=int(value["seed"]),
            difficulty=str(value["difficulty"]),
            prompt=str(value["prompt"]),
            contract=tuple(str(item) for item in value["contract"]),
            public_metadata=dict(value["public_metadata"]),
            workspace_ref=str(value["workspace_ref"]),
            verifier_ref=str(value["verifier_ref"]),
            split=str(value.get("split", "unassigned")),
        )


@dataclass(frozen=True)
class PolicySpec:
    id: str
    version: str
    system_prompt: str
    allowed_tools: tuple[str, ...]
    max_output_tokens: int
    tool_call_limit: int
    command_timeout_seconds: int
    wall_time_limit_seconds: int
    tool_interface: str = "native_bash"
    bundle_hash: str | None = None
    enforce_budget: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["allowed_tools"] = list(self.allowed_tools)
        return value


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    verifier_id: str
    verifier_version: str
    failure_code: str | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: str
    task_id: str
    policy_id: str
    policy_version: str
    workspace_ref: str
    created_at: str
    treatment_bundle_hash: str | None = None
    treatment_registry_hash: str | None = None
    rollout_replica: int | None = None
    sampling_seed: int | None = None
    pilot_manifest_hash: str | None = None
    pilot_panel_id: str | None = None
    status: str = "prepared"
    pi_events_ref: str | None = None
    normalized_events_ref: str | None = None
    verification_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AttemptRecord":
        return cls(**{key: value.get(key) for key in cls.__dataclass_fields__})
