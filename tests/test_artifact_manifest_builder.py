from __future__ import annotations

from datetime import UTC, datetime

import pytest

from quant_research.validation import (
    ARTIFACT_MANIFEST_BUILDER_SCHEMA_VERSION,
    ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS,
    ARTIFACT_MANIFEST_SCHEMA_ID,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    STANDARD_ARTIFACT_GROUPS,
    ManifestArtifactInput,
    build_artifact_manifest_from_paths,
    build_artifact_manifest_schema,
    collect_artifact_path_metadata,
    collect_git_version_info,
    load_artifact_manifest_json,
    validate_artifact_manifest_content_hashes,
    validate_artifact_manifest_schema,
    write_artifact_manifest_json,
)


def test_manifest_builder_normalizes_dataset_config_model_and_backtest_paths(tmp_path) -> None:
    dataset = tmp_path / "prices.csv"
    dataset.write_text("date,ticker,close\n2025-01-02,AAPL,100\n", encoding="utf-8")
    config = tmp_path / "canonical_config.json"
    config.write_text('{"target_horizon":"forward_return_20"}', encoding="utf-8")
    model_output = tmp_path / "model_features.jsonl"
    model_output.write_text('{"ticker":"AAPL"}\n{"ticker":"MSFT"}\n', encoding="utf-8")
    backtest_output = tmp_path / "backtest.csv"
    backtest_output.write_text("date,portfolio_return\n2025-02-03,0.01\n", encoding="utf-8")
    universe = tmp_path / "universe_snapshot.json"
    universe.write_text('{"universe":["AAPL","MSFT"]}', encoding="utf-8")
    feature_cutoff = tmp_path / "feature_cutoff.json"
    feature_cutoff.write_text('{"price":"date <= t"}', encoding="utf-8")
    report = tmp_path / "canonical_run_report.md"
    report.write_text("# Report\n", encoding="utf-8")

    manifest = build_artifact_manifest_from_paths(
        experiment_id="stage1_exp",
        run_id="run_001",
        dataset_paths=[dataset],
        config_paths=[
            {
                "path": config,
                "artifact_id": "canonical_config",
                "schema_id": "canonical_config_schema",
                "schema_version": "v1",
            }
        ],
        model_output_paths=[
            ManifestArtifactInput(
                model_output,
                "model_output",
                description="Structured model feature outputs, not order signals.",
            )
        ],
        backtest_output_paths=[backtest_output],
        universe_snapshot_path=universe,
        feature_availability_cutoff_path=feature_cutoff,
        report_path=report,
        created_at=datetime(2025, 3, 4, tzinfo=UTC),
        system_validity_status="pass",
        strategy_candidate_status="warning",
    )

    assert manifest["schema_id"] == ARTIFACT_MANIFEST_SCHEMA_ID
    assert manifest["schema_version"] == ARTIFACT_MANIFEST_SCHEMA_VERSION
    assert manifest["builder_schema_version"] == ARTIFACT_MANIFEST_BUILDER_SCHEMA_VERSION
    assert manifest["manifest_schema"] == build_artifact_manifest_schema()
    assert manifest["required_metadata_fields"] == list(
        ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS
    )
    assert manifest["manifest_id"] == "stage1_exp:run_001:artifact_manifest"
    assert manifest["experiment_id"] == "stage1_exp"
    assert manifest["run_id"] == "run_001"
    assert manifest["created_at"] == "2025-03-04T00:00:00+00:00"
    assert manifest["report_path"] == str(report)
    assert manifest["system_validity_status"] == "pass"
    assert manifest["strategy_candidate_status"] == "warning"
    assert set(manifest["git_version"]) == {
        "commit_sha",
        "branch",
        "dirty",
        "status_porcelain",
    }
    assert manifest["survivorship_bias_allowed"] is True
    assert "point-in-time universe" in manifest["survivorship_bias_disclosure"]
    assert manifest["artifact_groups"] == list(STANDARD_ARTIFACT_GROUPS)
    assert len(manifest["config_hash"]) == 64
    assert len(manifest["universe_snapshot_hash"]) == 64
    assert len(manifest["feature_availability_cutoff_hash"]) == 64
    assert len(manifest["data_snapshot_hash"]) == 64

    artifacts = manifest["artifacts"]
    assert {artifact["artifact_type"] for artifact in artifacts} == {
        "dataset",
        "config",
        "model_output",
        "backtest_output",
        "universe_snapshot",
        "feature_availability_cutoff",
        "report",
    }
    dataset_artifact = _artifact_by_type(artifacts, "dataset")
    assert dataset_artifact["artifact_id"] == "dataset:prices"
    assert dataset_artifact["path"] == str(dataset)
    assert dataset_artifact["row_count"] == 1
    assert len(dataset_artifact["content_hash"]) == 64
    assert dataset_artifact["size_bytes"] == dataset.stat().st_size
    assert dataset_artifact["relative_path"].endswith("prices.csv")
    assert dataset_artifact["absolute_path"] == str(dataset.resolve())
    assert dataset_artifact["is_directory"] is False
    assert isinstance(dataset_artifact["modified_at"], str)
    assert isinstance(dataset_artifact["modified_at_epoch_ns"], int)

    config_artifact = _artifact_by_type(artifacts, "config")
    assert config_artifact["artifact_id"] == "canonical_config"
    assert config_artifact["schema_id"] == "canonical_config_schema"
    assert config_artifact["schema_version"] == "v1"

    model_artifact = _artifact_by_type(artifacts, "model_output")
    assert model_artifact["row_count"] == 2
    assert "not order signals" in model_artifact["description"]


def test_manifest_json_round_trip_persists_and_validates_schema(tmp_path) -> None:
    dataset = tmp_path / "prices.csv"
    dataset.write_text("date,ticker,close\n2025-01-02,AAPL,100\n", encoding="utf-8")
    config = tmp_path / "canonical_config.json"
    config.write_text('{"target_horizon":"forward_return_20"}', encoding="utf-8")
    report = tmp_path / "canonical_run_report.md"
    report.write_text("# Report\n", encoding="utf-8")

    manifest = build_artifact_manifest_from_paths(
        experiment_id="stage1_exp",
        run_id="run_001",
        dataset_paths=[dataset],
        config_paths=[config],
        universe_snapshot_path=dataset,
        feature_availability_cutoff_path=config,
        report_path=report,
        created_at=datetime(2025, 3, 4, tzinfo=UTC),
        system_validity_status="pass",
        strategy_candidate_status="warning",
    )
    output_path = tmp_path / "nested" / "artifact_manifest.json"

    written_path = write_artifact_manifest_json(manifest, output_path)
    loaded = load_artifact_manifest_json(written_path)

    assert written_path == output_path
    assert loaded == manifest
    validate_artifact_manifest_schema(loaded)


def test_manifest_hash_validation_passes_when_recorded_hashes_match_artifacts(tmp_path) -> None:
    manifest = _minimal_manifest(tmp_path)

    validation = validate_artifact_manifest_content_hashes(manifest)

    assert validation["status"] == "pass"
    assert validation["checked_artifact_count"] == len(manifest["artifacts"])
    assert validation["mismatch_count"] == 0
    assert validation["missing_count"] == 0
    assert all(row["hash_matches"] is True for row in validation["artifacts"])
    assert {row["status"] for row in validation["artifacts"]} == {"match"}


def test_manifest_hash_validation_fails_when_artifact_content_hash_is_stale(tmp_path) -> None:
    manifest = _minimal_manifest(tmp_path)
    dataset_artifact = _artifact_by_type(manifest["artifacts"], "dataset")
    dataset_path = tmp_path / "prices.csv"
    expected_hash = dataset_artifact["content_hash"]

    dataset_path.write_text(
        "date,ticker,close\n2025-01-02,AAPL,100\n2025-01-03,AAPL,101\n",
        encoding="utf-8",
    )

    validation = validate_artifact_manifest_content_hashes(manifest)
    dataset_validation = _artifact_by_type(validation["artifacts"], "dataset")

    assert validation["status"] == "fail"
    assert validation["mismatch_count"] == 2
    assert validation["missing_count"] == 0
    assert dataset_validation["status"] == "mismatch"
    assert dataset_validation["hash_matches"] is False
    assert dataset_validation["expected_content_hash"] == expected_hash
    assert dataset_validation["actual_content_hash"] != expected_hash
    assert _artifact_by_type(validation["artifacts"], "universe_snapshot")["status"] == "mismatch"


def test_manifest_builder_reproduces_same_order_and_values_for_same_inputs(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "prices.csv"
    dataset.write_text("date,ticker,close\n2025-01-02,AAPL,100\n", encoding="utf-8")
    config = tmp_path / "canonical_config.json"
    config.write_text('{"target_horizon":"forward_return_20"}', encoding="utf-8")
    model_output = tmp_path / "model_features.jsonl"
    model_output.write_text('{"ticker":"AAPL","sentiment_score":0.2}\n', encoding="utf-8")
    backtest_output = tmp_path / "backtest.csv"
    backtest_output.write_text("date,portfolio_return\n2025-02-03,0.01\n", encoding="utf-8")
    universe = tmp_path / "universe_snapshot.json"
    universe.write_text('{"universe":["AAPL","MSFT"]}', encoding="utf-8")
    feature_cutoff = tmp_path / "feature_cutoff.json"
    feature_cutoff.write_text('{"price":"date <= t"}', encoding="utf-8")
    report = tmp_path / "canonical_run_report.md"
    report.write_text("# Report\n", encoding="utf-8")

    git_version = {
        "commit_sha": "abc123",
        "branch": "feature/stage1",
        "dirty": False,
        "status_porcelain": "",
    }
    monkeypatch.setattr(
        "quant_research.validation.manifest.collect_git_version_info",
        lambda *, repo_path=None: git_version,
    )

    build_kwargs = {
        "experiment_id": "stage1_exp",
        "run_id": "run_001",
        "dataset_paths": [dataset],
        "config_paths": [
            {
                "path": config,
                "artifact_id": "canonical_config",
                "schema_id": "canonical_config_schema",
                "schema_version": "v1",
            }
        ],
        "model_output_paths": [
            ManifestArtifactInput(
                model_output,
                "model_output",
                description="Structured model feature outputs, not order signals.",
            )
        ],
        "backtest_output_paths": [backtest_output],
        "universe_snapshot_path": universe,
        "feature_availability_cutoff_path": feature_cutoff,
        "report_path": report,
        "created_at": datetime(2025, 3, 4, tzinfo=UTC),
        "system_validity_status": "pass",
        "strategy_candidate_status": "warning",
    }

    first_manifest = build_artifact_manifest_from_paths(**build_kwargs)
    second_manifest = build_artifact_manifest_from_paths(**build_kwargs)
    first_output = tmp_path / "first" / "artifact_manifest.json"
    second_output = tmp_path / "second" / "artifact_manifest.json"

    write_artifact_manifest_json(first_manifest, first_output)
    write_artifact_manifest_json(second_manifest, second_output)

    assert second_manifest == first_manifest
    assert second_output.read_text(encoding="utf-8") == first_output.read_text(encoding="utf-8")
    assert [artifact["artifact_type"] for artifact in first_manifest["artifacts"]] == [
        "dataset",
        "config",
        "model_output",
        "backtest_output",
        "universe_snapshot",
        "feature_availability_cutoff",
        "report",
    ]


def test_manifest_schema_validation_rejects_stale_schema_version(tmp_path) -> None:
    manifest = _minimal_manifest(tmp_path)
    manifest["schema_version"] = "canonical_artifact_manifest.v0"

    with pytest.raises(ValueError, match="schema_version"):
        validate_artifact_manifest_schema(manifest)


@pytest.mark.parametrize("field_name", ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS)
def test_manifest_schema_validation_rejects_missing_required_manifest_field(
    tmp_path,
    field_name: str,
) -> None:
    manifest = _minimal_manifest(tmp_path)
    manifest.pop(field_name)

    with pytest.raises(ValueError, match=field_name):
        validate_artifact_manifest_schema(manifest)


@pytest.mark.parametrize("field_name", ARTIFACT_MANIFEST_REQUIRED_METADATA_FIELDS)
def test_manifest_writer_rejects_missing_required_manifest_field(
    tmp_path,
    field_name: str,
) -> None:
    manifest = _minimal_manifest(tmp_path)
    manifest.pop(field_name)

    with pytest.raises(ValueError, match=field_name):
        write_artifact_manifest_json(manifest, tmp_path / "artifact_manifest.json")


def test_manifest_schema_validation_rejects_missing_artifact_required_field(tmp_path) -> None:
    manifest = _minimal_manifest(tmp_path)
    artifact = dict(manifest["artifacts"][0])
    artifact.pop("content_hash")
    manifest["artifacts"] = [artifact]

    with pytest.raises(ValueError, match="missing required field: content_hash"):
        write_artifact_manifest_json(manifest, tmp_path / "artifact_manifest.json")


def test_manifest_loader_rejects_non_object_json(tmp_path) -> None:
    path = tmp_path / "artifact_manifest.json"
    path.write_text("[]\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must contain an object"):
        load_artifact_manifest_json(path)


def test_collect_git_version_info_records_sha_branch_and_dirty_state(
    tmp_path, monkeypatch
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        assert kwargs["cwd"] == tmp_path
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        if args == ("git", "rev-parse", "HEAD"):
            return _Completed(stdout="abc123def456\n")
        if args == ("git", "branch", "--show-current"):
            return _Completed(stdout="feature/stage1\n")
        if args == ("git", "status", "--porcelain"):
            return _Completed(stdout=" M src/example.py\n?? new.txt\n")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr("quant_research.validation.manifest.subprocess.run", fake_run)

    version = collect_git_version_info(repo_path=tmp_path)

    assert version == {
        "commit_sha": "abc123def456",
        "branch": "feature/stage1",
        "dirty": True,
        "status_porcelain": "M src/example.py\n?? new.txt",
    }
    assert calls == [
        ("git", "rev-parse", "HEAD"),
        ("git", "branch", "--show-current"),
        ("git", "status", "--porcelain"),
    ]


def test_collect_git_version_info_marks_clean_worktree(tmp_path, monkeypatch) -> None:
    def fake_run(args, **kwargs):
        if args == ("git", "rev-parse", "HEAD"):
            return _Completed(stdout="abc123\n")
        if args == ("git", "branch", "--show-current"):
            return _Completed(stdout="main\n")
        if args == ("git", "status", "--porcelain"):
            return _Completed(stdout="")
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr("quant_research.validation.manifest.subprocess.run", fake_run)

    version = collect_git_version_info(repo_path=tmp_path)

    assert version["dirty"] is False
    assert version["status_porcelain"] == ""


def test_collect_artifact_path_metadata_supports_files_and_directories(tmp_path) -> None:
    file_path = tmp_path / "outputs" / "metrics.json"
    file_path.parent.mkdir()
    file_path.write_text('{"rank_ic":0.03}', encoding="utf-8")

    file_metadata = collect_artifact_path_metadata(file_path, relative_to=tmp_path)

    assert file_metadata["relative_path"] == "outputs/metrics.json"
    assert file_metadata["absolute_path"] == str(file_path.resolve())
    assert file_metadata["size_bytes"] == file_path.stat().st_size
    assert file_metadata["is_directory"] is False
    assert len(file_metadata["content_hash"]) == 64
    assert isinstance(file_metadata["modified_at"], str)
    assert isinstance(file_metadata["modified_at_epoch_ns"], int)

    artifact_dir = tmp_path / "bundle"
    artifact_dir.mkdir()
    first = artifact_dir / "a.txt"
    nested = artifact_dir / "nested"
    nested.mkdir()
    second = nested / "b.txt"
    first.write_text("alpha", encoding="utf-8")
    second.write_text("beta", encoding="utf-8")

    directory_metadata = collect_artifact_path_metadata(artifact_dir, relative_to=tmp_path)
    original_hash = directory_metadata["content_hash"]

    assert directory_metadata["relative_path"] == "bundle"
    assert directory_metadata["absolute_path"] == str(artifact_dir.resolve())
    assert directory_metadata["size_bytes"] == first.stat().st_size + second.stat().st_size
    assert directory_metadata["is_directory"] is True
    assert len(original_hash) == 64

    second.write_text("beta2", encoding="utf-8")
    changed_metadata = collect_artifact_path_metadata(artifact_dir, relative_to=tmp_path)

    assert changed_metadata["content_hash"] != original_hash
    assert changed_metadata["size_bytes"] == first.stat().st_size + second.stat().st_size


def test_manifest_builder_fails_fast_for_missing_artifact_path(tmp_path) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="artifact path does not exist"):
        build_artifact_manifest_from_paths(
            experiment_id="stage1_exp",
            run_id="run_001",
            dataset_paths=[missing],
        )


def test_manifest_builder_rejects_empty_identity(tmp_path) -> None:
    dataset = tmp_path / "prices.csv"
    dataset.write_text("date,ticker,close\n", encoding="utf-8")

    with pytest.raises(ValueError, match="experiment_id"):
        build_artifact_manifest_from_paths(
            experiment_id=" ",
            run_id="run_001",
            dataset_paths=[dataset],
        )


def _artifact_by_type(artifacts: list[dict[str, object]], artifact_type: str) -> dict[str, object]:
    for artifact in artifacts:
        if artifact["artifact_type"] == artifact_type:
            return artifact
    raise AssertionError(f"missing artifact type: {artifact_type}")


def _minimal_manifest(tmp_path) -> dict[str, object]:
    dataset = tmp_path / "prices.csv"
    dataset.write_text("date,ticker,close\n2025-01-02,AAPL,100\n", encoding="utf-8")
    config = tmp_path / "canonical_config.json"
    config.write_text('{"target_horizon":"forward_return_20"}', encoding="utf-8")
    report = tmp_path / "canonical_run_report.md"
    report.write_text("# Report\n", encoding="utf-8")
    return build_artifact_manifest_from_paths(
        experiment_id="stage1_exp",
        run_id="run_001",
        dataset_paths=[dataset],
        config_paths=[config],
        universe_snapshot_path=dataset,
        feature_availability_cutoff_path=config,
        report_path=report,
        created_at=datetime(2025, 3, 4, tzinfo=UTC),
    )


class _Completed:
    def __init__(self, *, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
