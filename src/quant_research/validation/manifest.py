from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from quant_research.validation.report_schema import (
    ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS,
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS,
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ARTIFACT_ROOT,
    DEFAULT_V1_SCOPE_EXCLUSIONS,
    build_artifact_manifest_schema,
)

ARTIFACT_MANIFEST_BUILDER_SCHEMA_VERSION = "artifact_manifest_builder.v1"
SHA256_HEX_LENGTH = 64
STANDARD_ARTIFACT_GROUPS: tuple[str, ...] = (
    "dataset",
    "config",
    "model_output",
    "backtest_output",
)


@dataclass(frozen=True, slots=True)
class ManifestArtifactInput:
    path: str | Path
    artifact_type: str
    artifact_id: str | None = None
    schema_id: str | None = None
    schema_version: str | None = None
    description: str | None = None

    @classmethod
    def from_value(
        cls,
        value: str | Path | Mapping[str, object] | ManifestArtifactInput,
        *,
        artifact_type: str,
    ) -> ManifestArtifactInput:
        if isinstance(value, ManifestArtifactInput):
            return value
        if isinstance(value, str | Path):
            return cls(path=value, artifact_type=artifact_type)
        if isinstance(value, Mapping):
            path = value.get("path")
            if not isinstance(path, str | Path):
                raise TypeError(f"{artifact_type} artifact input requires a path")
            return cls(
                path=path,
                artifact_type=str(value.get("artifact_type") or artifact_type),
                artifact_id=_optional_str(value.get("artifact_id")),
                schema_id=_optional_str(value.get("schema_id")),
                schema_version=_optional_str(value.get("schema_version")),
                description=_optional_str(value.get("description")),
            )
        raise TypeError(f"unsupported {artifact_type} artifact input: {type(value).__name__}")


def build_artifact_manifest_from_paths(
    *,
    experiment_id: str,
    run_id: str,
    dataset_paths: Sequence[str | Path | Mapping[str, object] | ManifestArtifactInput] = (),
    config_paths: Sequence[str | Path | Mapping[str, object] | ManifestArtifactInput] = (),
    model_output_paths: Sequence[str | Path | Mapping[str, object] | ManifestArtifactInput] = (),
    backtest_output_paths: Sequence[str | Path | Mapping[str, object] | ManifestArtifactInput] = (),
    report_path: str | Path | None = None,
    universe_snapshot_path: str | Path | Mapping[str, object] | ManifestArtifactInput | None = None,
    feature_availability_cutoff_path: str | Path | Mapping[str, object] | ManifestArtifactInput | None = None,
    created_at: datetime | str | None = None,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    report_artifact_root: str | Path = DEFAULT_REPORT_ARTIFACT_ROOT,
    metadata_schema_id: str = "canonical_report_metadata",
    metadata_schema_version: str = "canonical_report_metadata.v1",
    system_validity_status: str = "not_evaluated",
    strategy_candidate_status: str = "not_evaluated",
    survivorship_bias_allowed: bool = True,
    survivorship_bias_disclosure: str = (
        "v1 allows survivorship bias; point-in-time universe reconstruction is v2 scope."
    ),
    v1_scope_exclusions: Sequence[str] = DEFAULT_V1_SCOPE_EXCLUSIONS,
    repo_path: str | Path | None = None,
) -> dict[str, object]:
    """Normalize run artifact paths into the canonical artifact manifest schema.

    This builder records reproducibility evidence only. It does not treat model outputs
    as order signals and does not introduce any live-trading execution fields.
    """

    experiment_id = _required_text(experiment_id, "experiment_id")
    run_id = _required_text(run_id, "run_id")
    created_at_text = _created_at_text(created_at)
    artifacts: list[dict[str, object]] = []

    artifacts.extend(_artifact_records(dataset_paths, "dataset", created_at_text))
    artifacts.extend(_artifact_records(config_paths, "config", created_at_text))
    artifacts.extend(_artifact_records(model_output_paths, "model_output", created_at_text))
    artifacts.extend(_artifact_records(backtest_output_paths, "backtest_output", created_at_text))

    universe_record = _optional_artifact_record(
        universe_snapshot_path,
        artifact_type="universe_snapshot",
        created_at=created_at_text,
    )
    feature_cutoff_record = _optional_artifact_record(
        feature_availability_cutoff_path,
        artifact_type="feature_availability_cutoff",
        created_at=created_at_text,
    )
    if universe_record is not None:
        artifacts.append(universe_record)
    if feature_cutoff_record is not None:
        artifacts.append(feature_cutoff_record)

    if report_path is not None:
        artifacts.append(
            _artifact_record(
                ManifestArtifactInput(report_path, "report"),
                created_at=created_at_text,
            )
        )

    config_artifacts = _filter_artifacts(artifacts, "config")
    dataset_artifacts = _filter_artifacts(artifacts, "dataset")
    universe_artifacts = _filter_artifacts(artifacts, "universe_snapshot")
    feature_cutoff_artifacts = _filter_artifacts(artifacts, "feature_availability_cutoff")

    manifest = {
        "schema_id": ARTIFACT_MANIFEST_SCHEMA_ID,
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "builder_schema_version": ARTIFACT_MANIFEST_BUILDER_SCHEMA_VERSION,
        "manifest_schema": build_artifact_manifest_schema(),
        "required_metadata_fields": list(ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS),
        "manifest_id": f"{experiment_id}:{run_id}:artifact_manifest",
        "experiment_id": experiment_id,
        "run_id": run_id,
        "created_at": created_at_text,
        "report_path": str(Path(report_path)) if report_path is not None else None,
        "artifact_root": str(Path(artifact_root)),
        "report_artifact_root": str(Path(report_artifact_root)),
        "metadata_schema_id": metadata_schema_id,
        "metadata_schema_version": metadata_schema_version,
        "config_hash": _combined_content_hash(config_artifacts),
        "universe_snapshot_hash": _combined_content_hash(universe_artifacts or dataset_artifacts),
        "feature_availability_cutoff_hash": _combined_content_hash(
            feature_cutoff_artifacts or config_artifacts
        ),
        "data_snapshot_hash": _combined_content_hash(dataset_artifacts),
        "system_validity_status": system_validity_status,
        "strategy_candidate_status": strategy_candidate_status,
        "survivorship_bias_allowed": bool(survivorship_bias_allowed),
        "survivorship_bias_disclosure": _required_text(
            survivorship_bias_disclosure,
            "survivorship_bias_disclosure",
        ),
        "v1_scope_exclusions": list(v1_scope_exclusions),
        "git_version": collect_git_version_info(repo_path=repo_path),
        "reproducible_input_metadata_required": True,
        "result_sections_required": list(CANONICAL_REPORT_REQUIRED_RESULT_SECTIONS),
        "artifact_groups": list(STANDARD_ARTIFACT_GROUPS),
        "artifacts": artifacts,
    }
    return manifest


def write_artifact_manifest_json(
    manifest: Mapping[str, object],
    path: str | Path,
    *,
    validate: bool = True,
) -> Path:
    """Persist a canonical artifact manifest as deterministic JSON."""

    if validate:
        validate_artifact_manifest_schema(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_artifact_manifest_json(
    path: str | Path,
    *,
    validate: bool = True,
) -> dict[str, object]:
    """Load a canonical artifact manifest JSON file and optionally validate it."""

    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"artifact manifest JSON is invalid: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact manifest JSON must contain an object")
    if validate:
        validate_artifact_manifest_schema(payload)
    return payload


def validate_artifact_manifest_schema(manifest: Mapping[str, object]) -> None:
    """Validate the canonical artifact manifest contract used for reproducible runs."""

    if not isinstance(manifest, Mapping):
        raise TypeError("artifact manifest must be a mapping")
    _require_equal(manifest, "schema_id", ARTIFACT_MANIFEST_SCHEMA_ID)
    _require_equal(manifest, "schema_version", ARTIFACT_MANIFEST_SCHEMA_VERSION)
    _require_equal(
        manifest,
        "builder_schema_version",
        ARTIFACT_MANIFEST_BUILDER_SCHEMA_VERSION,
    )
    if manifest.get("manifest_schema") != build_artifact_manifest_schema():
        raise ValueError("artifact manifest manifest_schema does not match canonical schema")

    for field_name in ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS:
        if field_name not in manifest:
            raise ValueError(f"artifact manifest missing required metadata field: {field_name}")
        if manifest[field_name] is None:
            raise ValueError(f"artifact manifest metadata field must not be null: {field_name}")

    for field_name in (
        "manifest_id",
        "experiment_id",
        "run_id",
        "created_at",
        "report_path",
        "artifact_root",
        "report_artifact_root",
        "metadata_schema_id",
        "metadata_schema_version",
        "system_validity_status",
        "strategy_candidate_status",
        "survivorship_bias_disclosure",
    ):
        _require_text(manifest, field_name)

    for field_name in (
        "config_hash",
        "universe_snapshot_hash",
        "feature_availability_cutoff_hash",
        "data_snapshot_hash",
    ):
        _require_sha256(manifest, field_name)

    if not isinstance(manifest.get("survivorship_bias_allowed"), bool):
        raise ValueError("artifact manifest survivorship_bias_allowed must be boolean")
    _require_string_list(manifest, "v1_scope_exclusions")
    _require_string_list(manifest, "result_sections_required")
    artifact_groups = _require_string_list(manifest, "artifact_groups")
    missing_groups = sorted(set(STANDARD_ARTIFACT_GROUPS).difference(artifact_groups))
    if missing_groups:
        raise ValueError(f"artifact manifest missing standard artifact groups: {missing_groups}")
    if manifest.get("required_metadata_fields") != list(
        ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    ):
        raise ValueError("artifact manifest required_metadata_fields must match canonical schema")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("artifact manifest artifacts must be a non-empty list")
    schema = build_artifact_manifest_schema()
    required_artifact_fields = schema["required_artifact_fields"]
    if not isinstance(required_artifact_fields, list):
        raise ValueError("canonical artifact manifest schema is malformed")
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"artifact manifest artifact[{index}] must be an object")
        for field_name in required_artifact_fields:
            if field_name not in artifact:
                raise ValueError(
                    f"artifact manifest artifact[{index}] missing required field: {field_name}"
                )
            if artifact[field_name] is None:
                raise ValueError(
                    f"artifact manifest artifact[{index}] field must not be null: {field_name}"
                )
        for field_name in ("artifact_id", "artifact_type", "path", "created_at"):
            _require_text(artifact, field_name, prefix=f"artifact[{index}]")
        _require_sha256(artifact, "content_hash", prefix=f"artifact[{index}]")


def validate_artifact_manifest_content_hashes(
    manifest: Mapping[str, object],
    *,
    base_path: str | Path | None = None,
    validate_schema: bool = True,
) -> dict[str, object]:
    """Compare manifest content hashes against the current artifact bytes on disk."""

    if validate_schema:
        validate_artifact_manifest_schema(manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("artifact manifest artifacts must be a list")

    root = Path.cwd() if base_path is None else Path(base_path)
    validation_rows: list[dict[str, object]] = []
    mismatch_count = 0
    missing_count = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise ValueError(f"artifact manifest artifact[{index}] must be an object")
        path_text = _require_text(artifact, "path", prefix=f"artifact[{index}]")
        expected_hash = _require_sha256(artifact, "content_hash", prefix=f"artifact[{index}]")
        artifact_path = Path(path_text)
        if not artifact_path.is_absolute():
            artifact_path = root / artifact_path

        if not artifact_path.exists():
            actual_hash = None
            status = "missing"
            matches = False
            missing_count += 1
        else:
            actual_hash = _path_content_hash(artifact_path)
            matches = actual_hash == expected_hash
            status = "match" if matches else "mismatch"
            if not matches:
                mismatch_count += 1

        validation_rows.append(
            {
                "artifact_id": artifact.get("artifact_id"),
                "artifact_type": artifact.get("artifact_type"),
                "path": path_text,
                "expected_content_hash": expected_hash,
                "actual_content_hash": actual_hash,
                "hash_matches": matches,
                "status": status,
            }
        )

    return {
        "status": "pass" if mismatch_count == 0 and missing_count == 0 else "fail",
        "checked_artifact_count": len(validation_rows),
        "mismatch_count": mismatch_count,
        "missing_count": missing_count,
        "artifacts": validation_rows,
    }


def collect_git_version_info(repo_path: str | Path | None = None) -> dict[str, object]:
    """Collect current repository version metadata for reproducible run manifests."""

    cwd = Path.cwd() if repo_path is None else Path(repo_path)
    commit_sha = _run_git_command(("rev-parse", "HEAD"), cwd=cwd)
    branch = _run_git_command(("branch", "--show-current"), cwd=cwd)
    if not branch:
        branch = _run_git_command(("rev-parse", "--abbrev-ref", "HEAD"), cwd=cwd)
    status = _run_git_command(("status", "--porcelain"), cwd=cwd)

    dirty = None if status is None else bool(status.strip())
    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "dirty": dirty,
        "status_porcelain": status,
    }


def _artifact_records(
    values: Sequence[str | Path | Mapping[str, object] | ManifestArtifactInput],
    artifact_type: str,
    created_at: str,
) -> list[dict[str, object]]:
    return [
        _artifact_record(
            ManifestArtifactInput.from_value(value, artifact_type=artifact_type),
            created_at=created_at,
        )
        for value in values
    ]


def _optional_artifact_record(
    value: str | Path | Mapping[str, object] | ManifestArtifactInput | None,
    *,
    artifact_type: str,
    created_at: str,
) -> dict[str, object] | None:
    if value is None:
        return None
    return _artifact_record(
        ManifestArtifactInput.from_value(value, artifact_type=artifact_type),
        created_at=created_at,
    )


def _artifact_record(input_value: ManifestArtifactInput, *, created_at: str) -> dict[str, object]:
    path = Path(input_value.path)
    if not path.exists():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    path_metadata = collect_artifact_path_metadata(path)
    artifact_id = input_value.artifact_id or _default_artifact_id(path, input_value.artifact_type)
    record: dict[str, object] = {
        "artifact_id": artifact_id,
        "artifact_type": input_value.artifact_type,
        "path": str(path),
        "created_at": created_at,
        **path_metadata,
    }
    row_count = _row_count(path)
    if row_count is not None:
        record["row_count"] = row_count
    if input_value.schema_id:
        record["schema_id"] = input_value.schema_id
    if input_value.schema_version:
        record["schema_version"] = input_value.schema_version
    if input_value.description:
        record["description"] = input_value.description
    return record


def collect_artifact_path_metadata(
    path: str | Path,
    *,
    relative_to: str | Path | None = None,
) -> dict[str, object]:
    """Collect reproducibility metadata for a file or directory artifact."""

    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact path does not exist: {artifact_path}")

    absolute_path = artifact_path.resolve()
    modified_at_epoch_ns = _path_modified_at_epoch_ns(artifact_path)
    return {
        "content_hash": _path_content_hash(artifact_path),
        "size_bytes": _path_size_bytes(artifact_path),
        "modified_at": datetime.fromtimestamp(
            modified_at_epoch_ns / 1_000_000_000,
            tz=UTC,
        ).isoformat(),
        "modified_at_epoch_ns": modified_at_epoch_ns,
        "relative_path": _relative_path_text(absolute_path, relative_to=relative_to),
        "absolute_path": str(absolute_path),
        "is_directory": artifact_path.is_dir(),
    }


def _path_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(child.read_bytes())
        return digest.hexdigest()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git_command(args: Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ("git", *args),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _path_size_bytes(path: Path) -> int:
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    return path.stat().st_size


def _path_modified_at_epoch_ns(path: Path) -> int:
    modified_times = [path.stat().st_mtime_ns]
    if path.is_dir():
        modified_times.extend(child.stat().st_mtime_ns for child in path.rglob("*") if child.is_file())
    return max(modified_times)


def _relative_path_text(path: Path, *, relative_to: str | Path | None) -> str:
    root = Path.cwd() if relative_to is None else Path(relative_to)
    root = root.resolve()
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _combined_content_hash(artifacts: Iterable[Mapping[str, object]]) -> str:
    payload = [
        {
            "artifact_id": artifact.get("artifact_id"),
            "artifact_type": artifact.get("artifact_type"),
            "content_hash": artifact.get("content_hash"),
            "path": artifact.get("path"),
        }
        for artifact in artifacts
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _filter_artifacts(
    artifacts: Sequence[Mapping[str, object]],
    artifact_type: str,
) -> list[Mapping[str, object]]:
    return [artifact for artifact in artifacts if artifact.get("artifact_type") == artifact_type]


def _row_count(path: Path) -> int | None:
    if path.is_dir():
        return None
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = sum(1 for _ in reader)
        return max(rows - 1, 0)
    if suffix == ".jsonl":
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return None


def _default_artifact_id(path: Path, artifact_type: str) -> str:
    stem = path.name if path.is_dir() else path.stem
    normalized = "".join(character if character.isalnum() else "_" for character in stem).strip("_")
    return f"{artifact_type}:{normalized or 'artifact'}"


def _created_at_text(value: datetime | str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    if isinstance(value, str):
        return _required_text(value, "created_at")
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _require_equal(manifest: Mapping[str, object], field_name: str, expected: str) -> None:
    observed = manifest.get(field_name)
    if observed != expected:
        raise ValueError(f"artifact manifest {field_name} must be {expected!r}")


def _require_text(
    manifest: Mapping[str, object],
    field_name: str,
    *,
    prefix: str = "metadata",
) -> str:
    value = manifest.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"artifact manifest {prefix} field must be non-empty text: {field_name}")
    return value


def _require_sha256(
    manifest: Mapping[str, object],
    field_name: str,
    *,
    prefix: str = "metadata",
) -> str:
    value = _require_text(manifest, field_name, prefix=prefix)
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"artifact manifest {prefix} field must be sha256 hex: {field_name}")
    return value


def _require_string_list(manifest: Mapping[str, object], field_name: str) -> list[str]:
    value = manifest.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"artifact manifest {field_name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"artifact manifest {field_name} must contain only non-empty strings")
    return value
