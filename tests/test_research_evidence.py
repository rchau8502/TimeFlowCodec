import importlib.util
import json
from pathlib import Path

import pytest


def test_research_plot_points_come_from_supplied_measurements(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    path = Path(__file__).parents[1] / "scripts/generate_research_figures.py"
    spec = importlib.util.spec_from_file_location("research_figures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = tmp_path / "results.json"
    points = [dict(codec="tfc", clip="my_clip", rate_kbps=123.45, psnr=41.2, ssim=.99)]
    report.write_text(json.dumps(dict(timestamp="test run", points=points)))
    assert module.load_rd_points(report, "my_clip") == points
    with pytest.raises(ValueError, match="No measured"):
        module.load_rd_points(report, "absent")
