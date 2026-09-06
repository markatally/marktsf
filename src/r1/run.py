"""Public runpy entry point; notebook contains no local-package imports."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True
source_root = str(Path(__file__).resolve().parents[1])
if source_root not in sys.path:
    sys.path.insert(0, source_root)

from r1.workflow import run_step


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True)
    parser.add_argument("--parameters", required=True)
    arguments = parser.parse_args()
    run_step(arguments.step, **json.loads(arguments.parameters))
