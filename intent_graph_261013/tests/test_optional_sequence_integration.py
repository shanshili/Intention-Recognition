"""Apply the optional patch in isolation and exercise its training save hook."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class OptionalSequenceIntegrationTests(unittest.TestCase):
    def test_patch_saves_adjacent_test_figures_after_one_small_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            project = base / "intent_graph_261013"
            project.mkdir()
            for name in ("run_experiment.py", "compare_runs.py"):
                shutil.copy2(ROOT / name, project / name)
            shutil.copytree(ROOT / "experiment", project / "experiment",
                            ignore=shutil.ignore_patterns("__pycache__"))
            subprocess.run(["git", "apply", str(ROOT / "integration/consecutive_training.patch")],
                           cwd=base, check=True, capture_output=True, text=True)
            config = json.loads((ROOT / "configs/protocol_smoke.json").read_text(encoding="utf-8"))
            for key in ("flow_path", "coord_path"):
                config[key] = str((ROOT / config[key]).resolve())
            protocol = base / "protocol.json"
            protocol.write_text(json.dumps(config), encoding="utf-8")
            plot_config = base / "plots.json"
            plot_config.write_text(json.dumps({"count": 2, "panel_count": 2,
                                              "formats": ["png"], "dpi": 40}), encoding="utf-8")
            result = subprocess.run([
                sys.executable, str(project / "run_experiment.py"),
                "--protocol", str(protocol), "--experiments", str(ROOT / "configs/experiments.json"),
                "--models", "dynamic_gate", "--seeds", "0", "--epochs", "1", "--max-steps", "64",
                "--device", "cpu", "--skip-interventions", "--sequence-plots", str(plot_config),
                "--out", str(base / "outputs")], cwd=project, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            suite = next((base / "outputs").iterdir())
            run = suite / "dynamic_gate/seed_0"
            summary = json.loads((run / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(set(summary["split_results"]), {"train", "val", "test"})
            manifests = list((run / "splits/test/figures/consecutive").rglob("manifest_*.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["rendered_windows"], 2)
            self.assertTrue(manifest["consecutive_at_saved_stride"])
            self.assertEqual(len(list(manifests[0].parent.glob("intent_graph_window_*.png"))), 2)
            self.assertFalse((run / "splits/train/figures/consecutive").exists())
            self.assertFalse((run / "splits/val/figures/consecutive").exists())


if __name__ == "__main__":
    unittest.main()
