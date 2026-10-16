"""Strict ablation config guards and shared multi-split artifact contracts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from compare_runs import compare_runs
from experiment.ablation import resolve_model_configs, validate_strict_ablation
from experiment.runner import run_suite


ROOT = Path(__file__).resolve().parents[1]


class AblationRunnerTests(unittest.TestCase):
    def test_inheritance_is_canonical_independent_and_rejects_bad_links(self):
        models = resolve_model_configs({"M7": {"input_features": "full", "tcn_dilations": [1, 2, 4]},
                                        "M5_ATTENTION": {"inherits": "M7_L0"}})
        self.assertEqual(models["m7"], models["m5_attention"])
        models["m5_attention"]["tcn_dilations"].append(8)
        self.assertEqual(models["m7"]["tcn_dilations"], [1, 2, 4])
        for bad in ({"m7": {"inherits": "absent"}},
                    {"m7": {"inherits": "m6"}, "m6": {"inherits": "m7"}},
                    {"M7": {}, "M7_L0": {}}, {"m7": {"inherits": 7}}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_model_configs(bad)

    def test_strict_guard_enforces_same_mother_and_prediction_only_loss(self):
        document = json.loads((ROOT / "configs/experiments_ablation.json").read_text())
        models = resolve_model_configs(document["models"])
        self.assertEqual(len(models), 11)
        design = validate_strict_ablation(models, list(models), document["training"])
        self.assertEqual(design["mother_model"], "m7")
        self.assertEqual(design["model_roles"]["m6"], "without_temporal_attention")
        for name in models:
            self.assertEqual(models[name], models["m7"])
        for weight in (0.1, -1, float("nan")):
            with self.assertRaises(ValueError):
                validate_strict_ablation(models, ["m7"], {"loss": {"sparse": weight}})
        with self.assertRaises(ValueError):
            validate_strict_ablation(models, ["m7"], {}, {"encoder": "tcn"})
        with self.assertRaises(ValueError):
            validate_strict_ablation(models, ["m5"], {})
        models["m5_attention"]["encoder_hidden"] = 99
        with self.assertRaises(ValueError):
            validate_strict_ablation(models, ["m5_attention"], {})

    def test_bad_strict_loss_and_duplicate_alias_fail_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            for models, overrides in ((["M7"], {"loss": {"temporal": 0.1}}),
                                      (["M5_ATTENTION", "M7_NO_DYNAMIC_EDGE"], {}),
                                      (["M6", "M7_NO_TEMPORAL_ATTENTION"], {})):
                with self.subTest(models=models), self.assertRaises(ValueError):
                    run_suite(ROOT / "configs/protocol_construction_smoke.json",
                              ROOT / "configs/experiments_ablation.json", models=models,
                              training_overrides=overrides, out=directory, strict_ablation=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_all_variants_save_completed_three_split_results_and_epoch_logs(self):
        from experiment.sequence_visualization import discover_runs
        from experiment.sequence_plot_hook import render_saved_split
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
                summary = run_suite(path, ROOT / "configs/experiments_ablation.json", seeds=[0],
                                    training_overrides={"epochs": 1, "patience": 1, "device": "cpu"},
                                    evaluation_overrides={"top_fractions": [.5], "random_seeds": [0]},
                                    out=base / "outputs", make_plots=False, strict_ablation=True)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(summary["runs"]), 11)
            comparison = compare_runs([row["run"] for row in summary["runs"]])
            self.assertEqual(comparison["split_coverage"], "all_runs_three_splits")
            suite = Path(summary["output_dir"])
            self.assertTrue((suite / "experiment_design.json").exists())
            for row in summary["runs"]:
                run = Path(row["run"])
                saved = json.loads((run / "run_summary.json").read_text())
                self.assertEqual(saved["status"], "completed")
                self.assertEqual(saved["experiment_design"]["kind"], "strict_m7_l0")
                if row["model"] == "m6":
                    self.assertEqual(saved["experiment_design"]["ablation_role"], "without_temporal_attention")
                self.assertEqual(saved["fit"]["actual_epochs"], 1)
                self.assertNotIn("inherits", saved["model_config"])
                self.assertIn(f"model={row['model']} seed=0 epoch=1/1", stream.getvalue())
                for split in ("train", "val", "test"):
                    self.assertTrue((run / "splits" / split / "predictions.npz").exists())
                    has_scores = (run / "splits" / split / "edge_scores.npz").exists()
                    self.assertEqual(has_scores, row["model"] != "m7_no_intent")
                graph = saved["graph_metrics"]
                if row["model"] == "m7_no_intent":
                    self.assertFalse(graph["available"])
                    self.assertEqual(render_saved_split(run / "splits/test", {}), [])
                elif row["model"] == "m7_fixed_intent":
                    self.assertFalse(graph["available"])
                    self.assertIn("all_messages_off_mae", graph)
                else:
                    self.assertTrue(graph["available"])
                    self.assertTrue(graph["interventions_computed"])
            records = discover_runs(suite)
            no_intent = next(record for record in records if record["model"] == "m7_no_intent")
            self.assertEqual(no_intent["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
