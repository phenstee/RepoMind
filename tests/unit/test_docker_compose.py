"""Security and wiring assertions for the rendered local Docker Compose configuration."""

import fnmatch
import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _dockerignore_excludes(dockerignore_path: Path, candidate: str) -> bool:
    """Replay one `.dockerignore` file's rules against a candidate path.

    Implements the minimal subset of dockerignore syntax this repository's
    ignore files actually use: blank/comment lines are skipped, each
    remaining line is a shell-style glob matched with `fnmatch`, and a
    leading `!` re-includes a path an earlier rule excluded. Rules are
    applied strictly in file order, matching Docker's own last-match-wins
    behavior, so this validates the real policy rather than grepping for one
    literal pattern.
    """
    excluded = False
    for raw_line in dockerignore_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line[1:] if negate else line
        if fnmatch.fnmatch(candidate, pattern):
            excluded = not negate
    return excluded


@pytest.fixture(scope="module")
def compose_config() -> dict:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is unavailable; cannot render the Compose configuration")

    completed = subprocess.run(
        [docker, "compose", "config", "--format", "json"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_compose_publishes_datastores_on_loopback_only(compose_config: dict) -> None:
    for service_name, container_port in (("postgres", 5432), ("redis", 6379)):
        ports = compose_config["services"][service_name]["ports"]
        matching = [port for port in ports if port["target"] == container_port]
        assert len(matching) == 1
        binding = matching[0]
        assert binding["host_ip"] == "127.0.0.1"
        assert binding["published"] == str(container_port)
        assert binding["protocol"] == "tcp"
        assert binding["mode"] == "ingress"


def test_compose_publishes_app_services_on_loopback_only(compose_config: dict) -> None:
    for service_name, container_port in (("api", 8000), ("frontend", 3000)):
        ports = compose_config["services"][service_name]["ports"]
        matching = [port for port in ports if port["target"] == container_port]
        assert len(matching) == 1
        binding = matching[0]
        assert binding["host_ip"] == "127.0.0.1"
        assert binding["published"] == str(container_port)


def test_api_and_worker_reach_datastores_by_service_name(compose_config: dict) -> None:
    for service_name in ("api", "worker"):
        environment = compose_config["services"][service_name]["environment"]
        assert environment["DATABASE_URL"].startswith(
            "postgresql+psycopg://repomind:repomind@postgres:5432/"
        )
        assert environment["REDIS_URL"] == "redis://redis:6379/0"
        assert "localhost" not in environment["DATABASE_URL"]
        assert "127.0.0.1" not in environment["DATABASE_URL"]


def test_migrate_runs_before_api_and_worker_depend_on_it(compose_config: dict) -> None:
    services = compose_config["services"]

    migrate_depends_on = services["migrate"]["depends_on"]
    assert migrate_depends_on["postgres"]["condition"] == "service_healthy"

    for service_name in ("api", "worker"):
        depends_on = services[service_name]["depends_on"]
        assert depends_on["migrate"]["condition"] == "service_completed_successfully"
        assert depends_on["postgres"]["condition"] == "service_healthy"
        assert depends_on["redis"]["condition"] == "service_healthy"


def test_api_and_worker_share_the_persistent_workspace_mount(compose_config: dict) -> None:
    for service_name in ("api", "worker"):
        volumes = compose_config["services"][service_name]["volumes"]
        workspace_mounts = [volume for volume in volumes if volume["target"] == "/workspace"]
        assert len(workspace_mounts) == 1
        assert workspace_mounts[0]["type"] == "bind"


def test_app_services_declare_healthchecks(compose_config: dict) -> None:
    for service_name in ("api", "frontend"):
        assert "healthcheck" in compose_config["services"][service_name]


def test_frontend_waits_for_api_to_be_healthy(compose_config: dict) -> None:
    depends_on = compose_config["services"]["frontend"]["depends_on"]
    assert depends_on["api"]["condition"] == "service_healthy"


def test_api_and_worker_share_the_same_backend_identity(compose_config: dict) -> None:
    # REPOMIND_UID/REPOMIND_GID are user-overridable (see
    # test_backend_identity_defaults_to_uid_gid_1000_in_compose_source for the
    # default), so this only pins the architectural invariant: whatever
    # identity is configured, api/worker/migrate must all build with it.
    services = compose_config["services"]
    api_args = services["api"]["build"]["args"]
    worker_args = services["worker"]["build"]["args"]
    migrate_args = services["migrate"]["build"]["args"]

    assert "REPOMIND_UID" in api_args
    assert "REPOMIND_GID" in api_args
    assert api_args == worker_args == migrate_args


def test_backend_identity_defaults_to_uid_gid_1000_in_compose_source() -> None:
    source = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "REPOMIND_UID: ${REPOMIND_UID:-1000}" in source
    assert "REPOMIND_GID: ${REPOMIND_GID:-1000}" in source


def test_backend_runtime_image_includes_coding_verifiers_and_git() -> None:
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text(encoding="utf-8")
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    sync_commands = [
        line.strip() for line in dockerfile.splitlines() if line.startswith("RUN uv sync ")
    ]
    assert len(sync_commands) == 2
    assert all("--group dev" in command for command in sync_commands)
    assert all("--no-dev" not in command for command in sync_commands)
    assert '"pytest>=' in pyproject
    assert '"ruff>=' in pyproject

    assert "apt-get install -y --no-install-recommends git" in dockerfile
    assert "rm -rf /var/lib/apt/lists/*" in dockerfile
    safe_directory = "RUN git config --system --add safe.directory '*'"
    assert safe_directory in dockerfile
    assert dockerfile.index(safe_directory) < dockerfile.index("USER repomind")


def test_compose_source_does_not_hardcode_secrets() -> None:
    source = (REPOSITORY_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${OPENAI_API_KEY" in source
    assert "sk-" not in source


def test_frontend_dockerignore_excludes_generic_env_files() -> None:
    # The frontend Dockerfile does `COPY . .`, so any generic Next.js env
    # file left in the build context would be sent to the Docker daemon.
    # frontend/.dockerignore must exclude them the same way the backend's
    # root .dockerignore excludes .env/.env.* for the API/worker images.
    dockerignore_path = REPOSITORY_ROOT / "frontend" / ".dockerignore"

    for excluded_name in (
        ".env",
        ".env.production",
        ".env.development",
        ".env.test",
        ".env.local",
        ".env.production.local",
    ):
        assert _dockerignore_excludes(dockerignore_path, excluded_name), (
            f"frontend/.dockerignore must exclude {excluded_name}"
        )

    # The checked-in example must remain usable/inspectable in the build
    # context - it holds no real secret.
    assert (REPOSITORY_ROOT / "frontend" / ".env.local.example").exists()
    assert not _dockerignore_excludes(dockerignore_path, ".env.local.example")
