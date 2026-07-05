import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from torchvision.datasets import ImageFolder

ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from project_specific_utils.dataloader import (
    load_img_natraster,
    map_image_order_from_ann_to_monkey,
)
from II_analyses.static_dyn import compute_static_linear_encoding
from useful_stuff.general_utils.utils import print_wise
from useful_stuff.image_processing.computational_models import get_relevant_output_layers
from useful_stuff.parallel.parallel_funcs import master_workers_queue, parallel_setup


# Example:
# mpiexec -np 5 python3 run_static_linear_encoding.py --monkey_name=three0 --date=250313 \
#   --brain_area=AIT --folder_name=talia_20each_tizi --model_name=vit_l_16 --img_size=384 \
#   --pooling=mean --new_fs=100 --pkg=timm --regression_type=ridge --score_type=corr \
#   --alpha_min=1e-6 --alpha_max=1e3 --n_alphas=10 --normalize --feature_center
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-validate static model features against dynamic neural activity."
    )
    parser.add_argument("--monkey_name", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--brain_area", required=True)
    parser.add_argument("--folder_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--img_size", type=int, required=True)
    parser.add_argument("--pooling", required=True)
    parser.add_argument("--new_fs", type=int, required=True)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--regression_type", choices=("lr", "ridge", "lasso", "en"), default="ridge")
    parser.add_argument("--score_type", choices=("r2", "corr"), default="r2")
    parser.add_argument("--cv_type", choices=("same", "loo", "kf"), default="loo")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=1e-6)
    parser.add_argument("--alpha_max", type=float, default=1e3)
    parser.add_argument("--n_alphas", type=int, default=10)
    parser.add_argument("--normalize", action="store_true", help="Use unit-norm population vectors (norm).")
    parser.add_argument("--feature_center", action="store_true", help="Center each feature across images (feat_cnt).")
    parser.add_argument("--mean_center", action="store_true", help="Center each population vector (mean_cnt).")
    cfg = parser.parse_args()

    if cfg.alpha_min <= 0 or cfg.alpha_max <= 0 or cfg.alpha_min > cfg.alpha_max:
        parser.error("alpha_min and alpha_max must be positive, with alpha_min <= alpha_max")
    if cfg.n_alphas < 1:
        parser.error("n_alphas must be at least 1")
    if cfg.cv_type == "kf" and cfg.n_splits < 2:
        parser.error("n_splits must be at least 2 for k-fold cross-validation")

    alphas = np.logspace(np.log10(cfg.alpha_min), np.log10(cfg.alpha_max), cfg.n_alphas)
    task_list = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    _, rank, _ = parallel_setup()
    if rank == 0:
        print_wise(cfg)
        raster = None
        idx_ord = None
    else:
        raster = load_img_natraster(
            paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs, brain_area=cfg.brain_area
        )
        dataset = ImageFolder(
            root=f"{paths['livingstone_lab']}/Stimuli/{cfg.folder_name}/",
            is_valid_file=lambda path: not path.endswith("Thumbs.db"),
            allow_empty=True,
        )
        idx_ord = map_image_order_from_ann_to_monkey(
            paths, cfg.monkey_name, cfg.date, dataset
        )

    master_workers_queue(
        task_list,
        paths,
        compute_static_linear_encoding,
        *(
            raster, idx_ord, cfg.monkey_name, cfg.date, cfg.brain_area,
            cfg.folder_name, cfg.model_name, cfg.img_size, cfg.pooling,
            cfg.regression_type, cfg.score_type, alphas, cfg.cv_type, cfg.n_splits,
            cfg.normalize, cfg.feature_center, cfg.mean_center,
        ),
    )
