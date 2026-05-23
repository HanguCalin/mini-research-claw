"""Unit tests for backend/utils/docker_utils.py

Tests:
- run_sandboxed success path (container runs, logs captured, metrics read)
- run_sandboxed handles ContainerError (code crashes inside)
- run_sandboxed raises RuntimeError if image is missing
- Docker client is called with correct security parameters
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import ContainerError, ImageNotFound

from backend.utils.docker_utils import run_sandboxed


@pytest.fixture
def mock_docker():
    with patch("backend.utils.docker_utils.get_docker_client") as mock_get:
        client = MagicMock()
        mock_get.return_value = client
        yield client


# ─── Success Path ─────────────────────────────────────────────────────────────

def test_run_sandboxed_success(mock_docker):
    mock_container = MagicMock()
    mock_container.logs.return_value = b"standard output logs"
    mock_docker.containers.run.return_value = mock_container
    mock_docker.images.get.return_value = MagicMock()

    with patch("backend.utils.docker_utils.Path.exists", return_value=True), \
         patch("backend.utils.docker_utils.Path.read_text", return_value='{"accuracy": 0.99}'):

        success, logs, metrics = run_sandboxed("print('hi')", ["numpy"], "/tmp/cache")

        assert success is True
        assert "standard output logs" in logs
        assert metrics == {"accuracy": 0.99}


# ─── Error Handling ───────────────────────────────────────────────────────────

def test_run_sandboxed_container_error(mock_docker):
    # Setup ContainerError
    mock_docker.images.get.return_value = MagicMock()
    
    exc = ContainerError(
        container="id123",
        exit_status=1,
        command="python main.py",
        image="sandbox",
        stderr=b"Traceback: ZeroDivisionError"
    )
    mock_docker.containers.run.side_effect = exc
    
    success, logs, metrics = run_sandboxed("1/0", [], "/tmp/cache")
    
    assert success is False
    assert "ZeroDivisionError" in logs
    assert metrics is None


def test_run_sandboxed_image_missing_raises_runtime_error(mock_docker):
    mock_docker.images.get.side_effect = ImageNotFound("not found")
    
    with pytest.raises(RuntimeError, match="Sandbox image .* not found"):
        run_sandboxed("print('hi')", [], "/tmp/cache")


# ─── Security Parameters ──────────────────────────────────────────────────────

def test_run_sandboxed_uses_strict_security_configs(mock_docker):
    mock_docker.images.get.return_value = MagicMock()
    mock_container = MagicMock()
    mock_container.logs.return_value = b"ok"
    mock_docker.containers.run.return_value = mock_container

    run_sandboxed("print('hi')", [], "/tmp/cache")
    
    # Capture the kwargs passed to containers.run
    args, kwargs = mock_docker.containers.run.call_args
    
    assert kwargs["network_mode"] == "none"
    assert kwargs["read_only"] is True
    assert "no-new-privileges" in kwargs["security_opt"][0]
    assert kwargs["mem_limit"] == "4g"
