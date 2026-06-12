import os, yaml, sys
import argparse
from pathlib import Path
from torchvision.datasets import ImageFolder
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from project_specific_utils.dataloader import load_img_natraster
from II_analyses.static_dyn import compute_static_dynII_subsampled
from project_specific_utils.dataloader import map_image_order_from_ann_to_monkey
from useful_stuff.parallel.parallel_funcs import master_workers_queue, parallel_setup
from useful_stuff.general_utils.utils import print_wise
from useful_stuff.image_processing.computational_models import get_relevant_output_layers



# e.g. to call it:
# mpiexec -np 5 python3 run_static_dynII_subsampled.py --monkey_name=three0 --date=250313 --brain_area=AIT --folder_name=talia_20each_tizi --signal_RDM_metric=cosine --model_RDM_metric=euclidean --model_name=vit_l_16 --img_size=384 --pooling=mean --new_fs=100 --pkg=timm --k=1 --subsamples_size=200 --n_iterations=20 --random_seed=0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run subsampled static-dynamic Information Imbalance for CNN layers")
    parser.add_argument("--monkey_name", type=str)
    parser.add_argument("--date", type=str)
    parser.add_argument("--brain_area", type=str)
    parser.add_argument("--folder_name", type=str)
    parser.add_argument("--signal_RDM_metric", type=str)
    parser.add_argument("--model_RDM_metric", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--img_size", type=int)
    parser.add_argument("--pooling", type=str)
    parser.add_argument("--new_fs", type=int)
    parser.add_argument("--pkg", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--subsamples_size", type=int)
    parser.add_argument("--n_iterations", type=int)
    parser.add_argument("--random_seed", type=int, default=None)

    cfg = parser.parse_args()

    task_list = get_relevant_output_layers(cfg.model_name, cfg.pkg)
    _, rank, _ = parallel_setup()
    if rank==0:
        print_wise(cfg)
        area_rasters = None
        dataset = None
        idx_ord = None
    else:
        area_rasters = load_img_natraster(paths, cfg.monkey_name, cfg.date, new_fs=cfg.new_fs, brain_area=cfg.brain_area)

        dataset = ImageFolder(
            root=f"{paths['livingstone_lab']}/Stimuli/{cfg.folder_name}/",
            is_valid_file=lambda x: not x.endswith("Thumbs.db"),
            allow_empty=True,
        )

        idx_ord = map_image_order_from_ann_to_monkey(paths, cfg.monkey_name, cfg.date, dataset)

    master_workers_queue(
        task_list,
        paths,
        compute_static_dynII_subsampled,
        *(
            area_rasters,
            idx_ord,
            cfg.signal_RDM_metric,
            cfg.model_RDM_metric,
            cfg.k,
            cfg.monkey_name,
            cfg.date,
            cfg.brain_area,
            cfg.folder_name,
            cfg.model_name,
            cfg.img_size,
            cfg.pooling,
            cfg.subsamples_size,
            cfg.n_iterations,
            cfg.random_seed,
        ),
    )
