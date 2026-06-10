import os, yaml, sys
import argparse
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import get_triu_perms
from project_specific_utils.dataloader import load_img_natraster
from II_analyses.metrics_comparison import dyn_compare_similarity_metrics 
from useful_stuff.parallel.parallel_funcs import master_workers_queue

# e.g. to call it:
# mpiexec -np 5 python3 run_metrics_comparison.py --monkey_name three0 --date 250313 --brain_area AIT --new_fs 100 --k 1 --metrics cosine_cnt cosine cosine_cnt correlation euclidean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--brain_area", type=str)
    parser.add_argument("--new_fs", type=int) 
    parser.add_argument("--metrics", type=str, nargs='+') 
    parser.add_argument("--k", type=int) 

    cfg = parser.parse_args()
    task_list = get_triu_perms(cfg.metrics)
    area_rasters = load_img_natraster(paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs, brain_area=cfg.brain_area)

    master_workers_queue(task_list, paths, dyn_compare_similarity_metrics, *(area_rasters, cfg.k, cfg.monkey_name, cfg.date, cfg.brain_area, cfg.new_fs)) 
