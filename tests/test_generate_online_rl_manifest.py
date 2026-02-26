"""Tests for online RL registry generator manifest/provenance output."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.generate_online_rl_from_registry import generate_online_rl_data


def test_generator_writes_manifest_and_provenance_hashes(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.yaml"
    metadata_path = tmp_path / "metadata.json"
    target_map_path = tmp_path / "target_map.json"
    output_path = tmp_path / "online_rl.jsonl"

    registry_path.write_text(
        yaml.safe_dump(
            {
                "challenges": [
                    {
                        "id": "chall-a",
                        "name": "Challenge A",
                        "category": "web",
                        "difficulty": "easy",
                        "infra_type": "docker",
                        "port": 32801,
                        "ground_truth_flag": "FLAG{A}",
                    },
                    {
                        "id": "chall-b",
                        "name": "Challenge B",
                        "category": "crypto",
                        "difficulty": "medium",
                        "infra_type": "static",
                        "ground_truth_flag": "FLAG{B}",
                    },
                ]
            }
        )
    )
    metadata_path.write_text(json.dumps({"descriptions": {"chall-a": "desc-a"}}))
    target_map_path.write_text(
        json.dumps({"challenge_targets": {"chall-a": "http://localhost:43001"}})
    )

    count = generate_online_rl_data(
        registry_path=str(registry_path),
        output_path=str(output_path),
        metadata_path=str(metadata_path),
        target_map_path=str(target_map_path),
        include_static=True,
        difficulty_max="master",
    )
    assert count == 2

    manifest_path = Path(f"{output_path}.manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sample_count"] == 2
    assert set(manifest["challenge_ids"]) == {"chall-a", "chall-b"}
    assert manifest["source"]["registry_sha256"]
    assert manifest["source"]["metadata_sha256"]
    assert manifest["source"]["target_map_sha256"]
    assert manifest["output_sha256"]

    rows = [json.loads(line) for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 2
    for row in rows:
        md = row.get("metadata", {})
        assert md.get("source_registry_sha256") == manifest["source"]["registry_sha256"]
        assert md.get("source_metadata_sha256") == manifest["source"]["metadata_sha256"]
        assert md.get("source_target_map_sha256") == manifest["source"]["target_map_sha256"]
