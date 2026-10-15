"""Run B1--B6 and M1--M7 using the shared construction data protocol."""

import sys
from pathlib import Path

from run_experiment import main as experiment_main


ROOT = Path(__file__).resolve().parent


def main(argv=None):
    supplied = list(sys.argv[1:] if argv is None else argv)
    defaults = ["--protocol", str(ROOT / "configs/protocol_construction_dev.json"),
                "--experiments", str(ROOT / "configs/experiments_construction.json")]
    return experiment_main([*defaults, *supplied])


if __name__ == "__main__":
    main()
