import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
from dataset.build_dataset import build
from dataset.run_baselines import main as run_baselines
from tests.dataset.test_export_episode import _mk   # reuse crafted episode

def test_full_pipeline(tmp_path):
    raw = tmp_path/"raw";
    for h in ("homeA","homeB","homeC"):
        _mk(str(raw/f"{h}__qwen__1"))
    out = str(tmp_path/"ds")
    build(str(raw), out)
    assert os.path.exists(f"{out}/episodes/homeA__qwen__1/windows.parquet")
    res = run_baselines(out)
    assert "random_forest" in res and "isolation_forest" in res
    assert os.path.exists(f"{out}/tables/baselines.tex")
