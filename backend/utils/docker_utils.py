"""Docker sandbox helpers for Node 5 (Executor).

Manages container lifecycle: create → run → capture logs → extract
metrics.json → destroy. All containers run with --network=none,
--read-only, --security-opt=no-new-privileges, and cgroup limits.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import docker
from docker.errors import ContainerError, ImageNotFound
from docker.models.containers import Container

from backend.config import SANDBOX, CACHE_ROOT


def get_docker_client() -> docker.DockerClient:
    return docker.from_env()


def run_sandboxed(
    python_code: str,
    resolved_dependencies: list[str],
    dataset_cache_path: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Execute *python_code* inside the hardened sandbox container.

    Returns (success, logs, metrics_json_or_none).
    """
    client = get_docker_client()

    try:
        client.images.get(SANDBOX.image_tag)
    except ImageNotFound:
        raise RuntimeError(
            f"Sandbox image '{SANDBOX.image_tag}' not found. "
            "Run: docker build -f Dockerfile.sandbox -t auto-mini-claw-sandbox:latest ."
        )

    with tempfile.TemporaryDirectory(prefix="miniclaw_") as workdir:
        script_path = Path(workdir) / "main.py"
        script_path.write_text(python_code, encoding="utf-8")

        metrics_path = Path(workdir) / "metrics.json"

        pip_cache = str(CACHE_ROOT / "pip")
        hf_cache = str(CACHE_ROOT / "hf")
        sklearn_cache = str(CACHE_ROOT / "sklearn")

        volumes = {
            workdir: {"bind": SANDBOX.workdir, "mode": "rw"},
            pip_cache: {"bind": SANDBOX.pip_mount, "mode": "ro"},
            hf_cache: {"bind": SANDBOX.hf_mount, "mode": "ro"},
            sklearn_cache: {"bind": SANDBOX.sklearn_mount, "mode": "ro"},
        }

        environment = {
            "PIP_FIND_LINKS": SANDBOX.pip_mount,
            "HF_DATASETS_CACHE": SANDBOX.hf_mount,
            "HF_HOME": SANDBOX.hf_mount,
            "SCIKIT_LEARN_DATA": SANDBOX.sklearn_mount,
            "PYTHONUNBUFFERED": "1",
        }

        try:
            container: Container = client.containers.run(
                image=SANDBOX.image_tag,
                command=["python", "-u", "main.py"],
                working_dir=SANDBOX.workdir,
                volumes=volumes,
                environment=environment,
                network_mode=SANDBOX.network_mode,
                mem_limit=SANDBOX.memory_limit,
                nano_cpus=int(SANDBOX.cpu_limit * 1e9),
                read_only=SANDBOX.read_only_root,
                security_opt=(
                    ["no-new-privileges"] if SANDBOX.no_new_privileges else []
                ),
                tmpfs={"/tmp": "size=512m"},
                detach=False,
                stdout=True,
                stderr=True,
                remove=False,
            )

            if isinstance(container, bytes):
                logs = container.decode("utf-8", errors="replace")
            else:
                logs = container.logs().decode("utf-8", errors="replace")
                container.remove(force=True)

        except ContainerError as exc:
            logs = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
            return False, logs, None

        metrics: dict[str, Any] | None = None
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass

        success = True
        return success, logs, metrics
