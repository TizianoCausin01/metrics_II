import os
import sys
from pathlib import Path

import yaml


ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]

with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils.feature_extraction import (
    parse_feature_extraction_args,
    run_feature_extraction,
)


if __name__ == "__main__":
    run_feature_extraction(paths, parse_feature_extraction_args())
# EOF
