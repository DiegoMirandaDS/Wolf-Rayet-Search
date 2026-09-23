from typer.testing import CliRunner

from wr_detector import cli
from wr_detector.cli import app


def test_cli_exposes_simbad_and_color_locus_commands():
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "build-simbad-negative" in result.output
    assert "export-simbad-negative-datasets" in result.output
    assert "export-color-locus" in result.output
    assert "build-prediction-pool" in result.output
    assert "audit-prediction-pool" in result.output
    assert "benchmark-models" in result.output
    assert "explore-models" in result.output
    assert "train-second-layer" in result.output


def test_explore_models_cli_builds_streamlit_command(monkeypatch):
    calls = []

    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda name: object() if name == "streamlit" else None)
    monkeypatch.setattr(cli.subprocess, "run", lambda command, check: calls.append((command, check)))

    result = CliRunner().invoke(app, ["explore-models", "--config", "configs/models.yaml", "--port", "8777"])

    assert result.exit_code == 0
    assert calls
    command, check = calls[0]
    assert check is True
    assert command[:4] == [cli.sys.executable, "-m", "streamlit", "run"]
    assert "--server.port" in command
    assert "8777" in command
    assert command[command.index("--server.address") + 1] == "127.0.0.1"
    assert command[command.index("--server.headless") + 1] == "true"
    assert command[command.index("--server.enableCORS") + 1] == "true"
    assert command[command.index("--server.enableXsrfProtection") + 1] == "true"
    assert command[command.index("--browser.gatherUsageStats") + 1] == "false"
    assert command[command.index("--config") + 1].replace("\\", "/") == "configs/models.yaml"
    assert (
        command[command.index("--candidate-config") + 1].replace("\\", "/")
        == "configs/prediction_pool_candidates.yaml"
    )
