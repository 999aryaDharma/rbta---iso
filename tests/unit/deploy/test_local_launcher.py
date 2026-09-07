from pathlib import Path

import pytest

from scripts.deploy.local import (
    LocalConfigurationError,
    discover_replay_files,
    normalize_docker_host_path,
    validate_local_configuration,
)


def test_normalizes_native_windows_path_without_losing_drive_letter():
    assert normalize_docker_host_path(r"D:\KAMPUS\SKRIPSI\wazuh-data-2026\indexer-export", platform="win32") == (
        "D:/KAMPUS/SKRIPSI/wazuh-data-2026/indexer-export"
    )


def test_preserves_wsl_mount_path():
    assert normalize_docker_host_path("/mnt/d/KAMPUS/SKRIPSI/indexer-export", platform="linux") == (
        "/mnt/d/KAMPUS/SKRIPSI/indexer-export"
    )


def test_local_discovery_ignores_meta_and_accepts_jsonl_gzip(tmp_path: Path):
    (tmp_path / "2026.04.02.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "2026.04.02.meta").write_text("{}", encoding="utf-8")
    (tmp_path / "2026.04.03.jsonl.gz").write_bytes(b"gzip")

    assert [item.name for item in discover_replay_files(tmp_path)] == [
        "2026.04.02.jsonl",
        "2026.04.03.jsonl.gz",
    ]


def test_configuration_requires_dataset_model_and_safe_port(tmp_path: Path):
    replay = tmp_path / "replay"
    model = tmp_path / "models" / "reference-v1"
    replay.mkdir()
    model.mkdir(parents=True)
    (replay / "day.jsonl").write_text("{}\n", encoding="utf-8")
    config = {
        "RBTA_API_KEY": "secure-demo-key",
        "RBTA_MODEL_VERSION": "reference-v1",
        "RBTA_HOST_PORT": "8010",
        "RBTA_REPLAY_HOST_DIR": str(replay),
        "RBTA_MODEL_HOST_DIR": str(tmp_path / "models"),
    }

    validated = validate_local_configuration(config, platform="linux")
    assert validated["RBTA_REPLAY_HOST_DIR"] == str(replay)
    assert validated["RBTA_REPLAY_FILE_COUNT"] == "1"

    with pytest.raises(LocalConfigurationError, match="1024 and 65535"):
        validate_local_configuration({**config, "RBTA_HOST_PORT": "80"}, platform="linux")
