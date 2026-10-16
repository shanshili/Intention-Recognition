"""Run the section-6 strict M7-L0 ablations on the shared data/evaluation protocol."""

import sys
from pathlib import Path

from run_experiment import main as experiment_main


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    defaults = ["--protocol", str(ROOT / "configs/protocol_construction_dev.json"),
                "--experiments", str(ROOT / "configs/experiments_ablation.json")]
    return experiment_main([*defaults, *supplied], strict_ablation=True)


if __name__ == "__main__":
    main()
