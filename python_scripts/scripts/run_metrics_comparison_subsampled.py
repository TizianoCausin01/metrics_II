import os, yaml, sys
import argparse
from pathlib import Path

ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])

from useful_stuff.general_utils.utils import get_triu_perms
from useful_stuff.parallel.parallel_funcs import master_workers_queue
from project_specific_utils.dataloader import load_img_natraster
from II_analyses.metrics_comparison import dyn_compare_similarity_metrics_subsampled

# e.g. to call it:
# mpiexec -np 5 python3 run_metrics_comparison_subsampled.py --monkey_name three0 --date 250313 --brain_area AIT --new_fs 100 --k 1 --subsamples_size 200 --n_iterations 20 --metrics cosine_cnt cosine cosine_cnt correlation euclidean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--brain_area", type=str)
    parser.add_argument("--new_fs", type=int)
    parser.add_argument("--metrics", type=str, nargs="+")
    parser.add_argument("--k", type=int)
    parser.add_argument("--subsamples_size", type=int)
    parser.add_argument("--n_iterations", type=int)
    parser.add_argument("--random_seed", type=int, default=0)

    cfg = parser.parse_args()
    task_list = get_triu_perms(cfg.metrics)
    area_rasters = load_img_natraster(
        paths,
        cfg.monkey_name,
        cfg.date,
        new_fs=cfg.new_fs,
        brain_area=cfg.brain_area,
    )

    master_workers_queue(
        task_list,
        paths,
        dyn_compare_similarity_metrics_subsampled,
        *(
            area_rasters,
            cfg.k,
            cfg.monkey_name,
            cfg.date,
            cfg.brain_area,
            cfg.new_fs,
            cfg.subsamples_size,
            cfg.n_iterations,
            cfg.random_seed,
        ),
    )
