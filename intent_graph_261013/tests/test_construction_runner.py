"""Construction aliases, complete shared targets, and per-epoch logging."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from compare_runs import compare_runs
from experiment.runner import run_suite


ROOT = Path(__file__).resolve().parents[1]


class ConstructionRunnerTests(unittest.TestCase):
    def test_input_views_share_targets_and_every_epoch_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            times = np.arange(64)
            values = np.stack([10 + times * .2 + np.sin(times / 3),
                               12 + np.cos(times / 4), 14 + times * .1], axis=1)
            pd.DataFrame(values, columns=["a", "b", "c"]).to_csv(base / "flow.csv", index=False)
            pd.DataFrame({"node_id": ["a", "b", "c"], "x": [0, 1, 2], "y": [0, 1, 0]}).to_csv(base / "coords.csv", index=False)
            protocol = json.loads((ROOT / "configs/protocol_construction_smoke.json").read_text())
            protocol.update(flow_path=str(base / "flow.csv"), coord_path=str(base / "coords.csv"), max_steps=64)
            path = base / "protocol.json"
            path.write_text(json.dumps(protocol), encoding="utf-8")
            stream = io.StringIO()
            with redirect_stdout(stream):
                summary = run_suite(path, ROOT / "configs/experiments_construction.json",
                                    models=["B6", "M1", "M7"], seeds=[0],
                                    training_overrides={"epochs": 2, "patience": 2, "device": "cpu"},
                                    evaluation_overrides={"graph_interventions": False},
                                    out=base / "outputs", make_plots=False)
            comparison = compare_runs([row["run"] for row in summary["runs"]])
            self.assertEqual(comparison["split_coverage"], "all_runs_three_splits")
            self.assertEqual([row["model"] for row in summary["runs"]], ["dynamic_gate", "m1", "m7"])
            for row in summary["runs"]:
                run = Path(row["run"])
                stored = json.loads((run / "run_summary.json").read_text())
                for epoch in (0, 1, 2):
                    self.assertIn(f"model={row['model']} seed=0 epoch={epoch}/2", stream.getvalue())
                self.assertEqual(stored["fit"]["actual_epochs"], 2)
                config = json.loads((run / "resolved_config.json").read_text())
                channels = config["protocol"]["feature_channels"]
                self.assertEqual(len(channels), 7 if row["model"] == "m7" else 1)
                self.assertIsNotNone(stored["fit"]["history"][1]["val_prediction_loss"])
            self.assertIn("total=", stream.getvalue())
            self.assertIn("no_improve=", stream.getvalue())

    def test_duplicate_b6_m0_aliases_fail_before_any_run_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "unique model"):
                run_suite(ROOT / "configs/protocol_construction_smoke.json",
                          ROOT / "configs/experiments_construction.json", models=["B6", "M0"], out=directory)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
