import os, yaml, sys
import argparse
from itertools import combinations_with_replacement
from pathlib import Path

ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from II_analyses.metrics_comparison import layer_distance_comparison_subsampled
from useful_stuff.general_utils.utils import print_wise
from useful_stuff.image_processing.computational_models import get_relevant_output_layers
from useful_stuff.parallel.parallel_funcs import master_workers_queue, parallel_setup

# e.g. to call it:
# mpiexec -np 5 python3 run_layer_distance_comparison_subsampled.py --folder_name=talia_20each_tizi --model_name=vit_l_16 --img_size=384 --pooling=mean --pkg=timm --k=1 --subsamples_size=200 --n_iterations=20 --random_seed=None --metrics cosine correlation euclidean


def optional_int(value):
    if value is None or value.lower() == "none":
        return None
    return int(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run subsampled Information Imbalance comparisons between model layer distances"
    )
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--metrics", type=str, nargs="+")
    parser.add_argument("--subsamples_size", type=int)
    parser.add_argument("--n_iterations", type=int)
    parser.add_argument("--random_seed", type=optional_int, default=None)

    cfg = parser.parse_args()

    layers = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    task_list = list(combinations_with_replacement(cfg.metrics, 2))
    _, rank, _ = parallel_setup()
    if rank == 0:
        print_wise(cfg)

    master_workers_queue(
        task_list,
        paths,
        layer_distance_comparison_subsampled,
        *(
            layers,
            cfg.k,
            cfg.folder_name,
            cfg.model_name,
            cfg.img_size,
            cfg.pooling,
            cfg.subsamples_size,
            cfg.n_iterations,
            cfg.random_seed,
        ),
    )
