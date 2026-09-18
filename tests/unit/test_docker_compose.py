"""Security assertions for the rendered local Docker Compose configuration."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_compose_publishes_datastores_on_loopback_only() -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable; cannot render the Compose configuration")

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [docker, "compose", "config", "--format", "json"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    configuration = json.loads(completed.stdout)

    for service_name, container_port in (("postgres", 5432), ("redis", 6379)):
        ports = configuration["services"][service_name]["ports"]
        matching = [port for port in ports if port["target"] == container_port]
        assert len(matching) == 1
        binding = matching[0]
        assert binding["host_ip"] == "127.0.0.1"
        assert binding["published"] == str(container_port)
        assert binding["protocol"] == "tcp"
        assert binding["mode"] == "ingress"
