"""Offline plotting contracts: edge budget, time continuity, and provenance."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from experiment.sequence_visualization import (
    PlotSettings, SequenceData, discover_runs, load_sequence, render_sequence,
    select_mask, shared_normalization, validate_sequence,
)


def toy_data(scores=None, split="test", stride=1, kind="dynamic"):
    ids = np.arange(3) * stride + 11
    times = (np.datetime64("2025-01-01T00:00") +
             (ids[:, None] + np.arange(1, 3)) * np.timedelta64(5, "m"))
    return SequenceData(scores=np.array(scores if scores is not None else [[.1, .4, .2, .3]] * 3),
                        coords=np.array([[0., 0.], [1., 0.], [0., 1.]]),
                        edges=np.array([[0, 1], [1, 0], [1, 2], [2, 1]]),
                        window_ids=ids, target_times=times, node_ids=np.array(["n0", "n1", "n2"]),
                        protocol={"W": 12, "H": 2, "step_minutes": 5, "train_stride": 3, "eval_stride": 1},
                        model="toy", seed="seed_0", split=split, score_kind=kind)


class SequenceVisualizationTests(unittest.TestCase):
    def test_draw_clamps_float32_normalization_endpoint_roundoff(self):
        import matplotlib.pyplot as plt
        from experiment.sequence_visualization import _draw
        data = toy_data()
        fig, ax = plt.subplots()
        try:
            # Reproduce a float32 LogNorm result marginally above its endpoint.
            _draw(ax, data, 0, np.ones(4, dtype=bool),
                  lambda value: np.float32(1.0000001192092896), PlotSettings())
            self.assertEqual(len(ax.patches), 4)
            for arrow in ax.patches:
                self.assertEqual(arrow.get_alpha(), 1.0)
                self.assertEqual(arrow.get_linewidth(), PlotSettings().max_linewidth)
        finally:
            plt.close(fig)

    def test_exact_topk_stable_ties_and_zero_suppression(self):
        keep, drawn, _ = select_mask(np.array([.8, .8, .8, 0.]), "topk", fraction=.5)
        np.testing.assert_array_equal(np.flatnonzero(keep), [0, 1])
        np.testing.assert_array_equal(keep, drawn)
        keep, drawn, _ = select_mask(np.zeros(4), "topk", fraction=.5)
        self.assertEqual(keep.sum(), 2)
        self.assertEqual(drawn.sum(), 0)

    def test_quantile_ties_are_not_exact_topk(self):
        keep, drawn, threshold = select_mask(np.ones(4), "window_quantile", quantile=.9)
        self.assertEqual(keep.sum(), 4)
        self.assertEqual(drawn.sum(), 4)
        self.assertEqual(threshold, 1)
        _, drawn, _ = select_mask(np.zeros(4), "window_quantile")
        self.assertFalse(drawn.any())

    def test_continuity_uses_saved_split_stride_and_timestamps(self):
        data = toy_data(split="train", stride=3)
        picks, stride, contiguous = validate_sequence(data, PlotSettings(count=3))
        self.assertTrue(contiguous)
        self.assertEqual(stride, 3)
        data.window_ids[-1] += 1
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            validate_sequence(data, PlotSettings(count=3))
        self.assertFalse(validate_sequence(data, PlotSettings(count=3, allow_gaps=True))[2])
        data = toy_data()
        data.target_times[-1] += np.timedelta64(5, "m")
        with self.assertRaisesRegex(ValueError, "not consecutive"):
            validate_sequence(data, PlotSettings(count=3))

    def test_shared_color_range_is_sequence_wide_and_configurable(self):
        scores = np.array([[1e-5, .1], [.5, .9]])
        norm, metadata = shared_normalization(scores, PlotSettings())
        self.assertGreater(float(norm(.5)), float(norm(.1)))
        self.assertTrue(metadata["shared_across_all_sequence_figures"])
        self.assertLess(metadata["vmin"], .1)
        self.assertGreater(metadata["vmax"], .5)
        _, fixed = shared_normalization(scores, PlotSettings(norm_scale="linear", vmin=0., vmax=1.))
        self.assertEqual(fixed["range_source"], "explicit_fixed_range")
        with self.assertRaisesRegex(ValueError, "both"):
            shared_normalization(scores, PlotSettings(vmin=.01))

    def test_threshold_requires_frozen_validation_provenance(self):
        data = toy_data()
        data.threshold = {"enabled": True, "value": .25, "calibration_split": "test", "rule": "test_quantile"}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "provenance"):
                render_sequence(data, directory, PlotSettings(count=1, mode="saved_threshold", formats=("png",), dpi=40))
            data.threshold = {"enabled": True, "value": .25, "calibration_split": "val", "rule": "validation_positive_quantile"}
            result = render_sequence(data, directory, PlotSettings(count=2, mode="saved_threshold", formats=("png",), dpi=40))
            self.assertEqual([row["selection_threshold"] for row in result["windows"]], [.25, .25])
            self.assertEqual([row["positive_drawn_edges"] for row in result["windows"]], [2, 2])

    def test_gate_validation_rejects_nonfinite_out_of_range_and_false_fixed(self):
        for value in (np.nan, np.inf, -0.01, 1.01):
            data = toy_data()
            data.scores[0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite.*\\[0, 1\\]"):
                validate_sequence(data, PlotSettings(count=1))
        data = toy_data(kind="fixed")
        with self.assertRaisesRegex(ValueError, "all-one"):
            validate_sequence(data, PlotSettings(count=1))
        for edges in (np.array([[0, 0], [1, 0], [1, 2], [2, 1]]),
                      np.array([[0, 1], [0, 1], [1, 2], [2, 1]])):
            data = toy_data()
            data.edges = edges
            with self.assertRaisesRegex(ValueError, "unique.*self-loops"):
                validate_sequence(data, PlotSettings(count=1))

    def test_fixed_graph_and_empty_gate_sequences_render_truthfully(self):
        with tempfile.TemporaryDirectory() as directory:
            fixed = render_sequence(toy_data(np.ones((3, 4)), kind="fixed"), directory,
                                    PlotSettings(count=2, formats=("png",), dpi=40), timestamp="fixed")
            self.assertEqual(fixed["effective_mode"], "all_candidate")
            self.assertEqual(fixed["windows"][0]["positive_drawn_edges"], 4)
            zero = render_sequence(toy_data(np.zeros((3, 4))), directory,
                                   PlotSettings(count=2, formats=("png",), dpi=40), timestamp="zero")
            self.assertEqual(zero["windows"][0]["positive_drawn_edges"], 0)
            self.assertEqual(zero["windows"][0]["selected_edges"], 1)
            self.assertEqual(zero["windows"][0]["input_start_time"], "2025-01-01 00:00")
            self.assertEqual(zero["windows"][0]["target_end_time"], "2025-01-01 01:05")
            self.assertTrue(Path(zero["window_index_csv"]).exists())
            self.assertTrue(all(Path(path).exists() for path in zero["figures"]))
            self.assertFalse(zero["timestamp_provenance"]["real_world_clock_independently_verified_by_plotter"])
            with self.assertRaises(FileExistsError):
                render_sequence(toy_data(), directory, PlotSettings(count=1), timestamp="zero")

    def test_discovery_all_graph_seeds_and_explicit_no_score_baselines(self):
        with tempfile.TemporaryDirectory() as directory:
            for model in ("fixed_graph", "static_gate", "dynamic_gate"):
                for seed in range(3):
                    split = Path(directory) / model / f"seed_{seed}" / "splits" / "test"
                    split.mkdir(parents=True)
                    np.savez(split / "edge_scores.npz", edge_scores=np.ones((1, 1)))
            for model, seeds in (("persistence", [0]), ("ridge_ar", [0]), ("local", range(3))):
                for seed in seeds:
                    split = Path(directory) / model / f"seed_{seed}" / "splits" / "test"
                    split.mkdir(parents=True)
                    (split / "graph_evaluation.json").write_text(json.dumps({"score_kind": "none"}), encoding="utf-8")
            records = discover_runs(directory)
            self.assertEqual(sum(row["status"] == "ready" for row in records), 9)
            self.assertEqual(sum(row["status"] == "not_applicable" for row in records), 5)
            self.assertEqual(len(discover_runs(directory, models=["dynamic_gate"], seeds=[1])), 1)

    def test_loader_uses_saved_validation_threshold_and_only_artifact_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "dynamic_gate" / "seed_0"
            test = run / "splits" / "test"
            val = run / "splits" / "val"
            test.mkdir(parents=True)
            val.mkdir()
            data = toy_data()
            np.savez(test / "edge_scores.npz", edge_scores=data.scores, coords=data.coords,
                     candidate_edges=data.edges, window_id=data.window_ids, target_times=data.target_times,
                     score_kind=np.asarray("dynamic"))
            (test / "resolved_config.json").write_text(json.dumps({"protocol": data.protocol, "model": "dynamic_gate"}), encoding="utf-8")
            (test / "graph_evaluation.json").write_text(json.dumps({"threshold": {"enabled": True, "value": 999., "rule": "test_quantile"}}), encoding="utf-8")
            (val / "graph_evaluation.json").write_text(json.dumps({"threshold": {"enabled": True, "value": .3, "rule": "validation_positive_quantile", "calibration_split": "val"}}), encoding="utf-8")
            loaded = load_sequence(test / "edge_scores.npz")
            self.assertEqual(loaded.threshold["value"], .3)
            self.assertIn("val", loaded.threshold["source_path"])
            self.assertEqual(loaded.model, "dynamic_gate")

    def test_predeclared_absolute_threshold_matches_evaluator_rule(self):
        data = toy_data()
        data.threshold = {"enabled": True, "value": .3, "rule": "predeclared_absolute"}
        with tempfile.TemporaryDirectory() as directory:
            result = render_sequence(data, directory, PlotSettings(count=1, mode="saved_threshold", formats=("png",), dpi=40))
            self.assertEqual(result["threshold"]["value"], .3)
            self.assertEqual(result["windows"][0]["positive_drawn_edges"], 2)

    def test_missing_expected_scores_produce_failed_cli_manifest_and_nonzero_exit(self):
        from plot_intent_sequences import main, parser
        self.assertEqual(parser().parse_args([]).viz_count, 10)
        self.assertEqual(PlotSettings().count, 10)
        with tempfile.TemporaryDirectory() as directory:
            split = Path(directory) / "dynamic_gate" / "seed_0" / "splits" / "test"
            split.mkdir(parents=True)
            (split / "graph_evaluation.json").write_text(json.dumps({"score_kind": "dynamic"}), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                exit_code = main(["--suite", directory])
            self.assertEqual(exit_code, 1)
            manifest_path = next((Path(directory) / "figures" / "consecutive").glob("*/suite_plot_manifest_*.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed_with_errors")
            self.assertEqual(len(manifest["failed"]), 1)
            self.assertEqual(len(manifest["skipped"]), 0)
            with redirect_stdout(io.StringIO()):
                exit_code = main(["--suite", directory, "--splits", "val"])
            self.assertEqual(exit_code, 1)

    def test_module_does_not_import_torch_or_training_modules(self):
        result = subprocess.run([sys.executable, "-c", "import sys; import experiment.sequence_visualization; "
                                 "assert 'torch' not in sys.modules; "
                                 "assert 'experiment.trainer' not in sys.modules; "
                                 "assert 'experiment.data' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
