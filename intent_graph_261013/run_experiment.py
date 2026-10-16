"""Run replaceable models with one dataset protocol and one evaluator."""

import argparse
import json
from pathlib import Path

from experiment.models import MODEL_NAMES, canonical_model_name
from experiment.runner import run_suite


ROOT = Path(__file__).resolve().parent


def main(argv=None, *, strict_ablation=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/protocol_dev.json")
    parser.add_argument("--experiments", type=Path, default=ROOT / "configs/experiments.json")
    parser.add_argument("--models", nargs="+", type=canonical_model_name, choices=MODEL_NAMES,
                        help="Canonical names, B1-B6/M0-M7 or M7 ablation aliases; M0 equals dynamic_gate")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int, help="Stop after this many validation epochs without improvement")
    parser.add_argument("--min-delta", type=float, help="Minimum standardized validation MAE improvement")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--encoder", choices=("mlp", "tcn"), help="Original B3-B6 controls only; M1-M7 fix their encoder")
    parser.add_argument("--lambda-sparse", type=float)
    parser.add_argument("--lambda-direction", type=float)
    parser.add_argument("--lambda-temporal", type=float)
    parser.add_argument("--random-seeds", type=int, help="Number of independent deletion/retention seeds")
    parser.add_argument("--skip-interventions", action="store_true", help="Prediction and score diagnostics only")
    parser.add_argument("--graph-intervention-splits", nargs="+", choices=("train", "val", "test"),
                        help="Splits with deletion/retention tests; predictions and gates are always saved for all splits")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--sequence-plots", type=Path, help="JSON settings for adjacent test-window plots")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = parser.parse_args(argv)
    data = {} if args.max_steps is None else {"max_steps": args.max_steps}
    training = {key: value for key, value in {"device": args.device, "epochs": args.epochs,
                                           "patience": args.patience, "min_delta": args.min_delta,
                                           "batch_size": args.batch_size}.items() if value is not None}
    loss = {key: value for key, value in {"sparse": args.lambda_sparse,
                                        "direction": args.lambda_direction,
                                        "temporal": args.lambda_temporal}.items() if value is not None}
    if loss:
        training["loss"] = loss
    evaluation = {}
    if args.random_seeds is not None:
        if args.random_seeds < 1:
            parser.error("--random-seeds must be positive")
        evaluation["random_seeds"] = list(range(args.random_seeds))
    if args.skip_interventions:
        evaluation["graph_interventions"] = False
    if args.graph_intervention_splits is not None:
        evaluation["graph_intervention_splits"] = args.graph_intervention_splits
    if args.sequence_plots is not None:
        evaluation["sequence_plots"] = json.loads(args.sequence_plots.read_text(encoding="utf-8"))
    model = {} if args.encoder is None else {"encoder": args.encoder}
    return run_suite(args.protocol, args.experiments, args.models, args.seeds, data,
                     training, evaluation, model, args.out, not args.no_plots, args.prepare_only,
                     strict_ablation=strict_ablation)


if __name__ == "__main__":
    main()
