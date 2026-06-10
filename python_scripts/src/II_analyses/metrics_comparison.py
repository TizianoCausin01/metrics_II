import os, sys, yaml
import numpy as np
ENV = os.getenv("MY_ENV", "dev")
with open("../../config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries, print_wise
from useful_stuff.general_utils.II import InformationImbalance, dynInformationImbalance

"""
compare_similarity_metrics
Compute Information Imbalance (II) between two similarity metrics to compute RDMs
INPUT
- data : np.ndarray (D, N) -> Input data used to compute both RDMs.
- metric1, metric2 : str -> Similarity / distance metrics for signal and model RDMs.
- k : int -> Number of nearest neighbors used for conditioning.

OUTPUT:
- ii_obj : InformationImbalance -> Initialized and fully computed InformationImbalance object.
- ii_A2B : float -> Information Imbalance from metric1 to metric2.
- ii_B2A : float -> Information Imbalance from metric2 to metric1.
"""
def compare_similarity_metrics(data: np.ndarray, metric1: str, metric2: str, k: int): 
    ii_obj = InformationImbalance(metric1, metric2, k)
    ii_obj.compute_RDM(data, "signal")
    ii_obj.compute_RDM(data, "model")
    ii_obj.compute_both_distance_ranks()
    ii_A2B, ii_B2A = ii_obj.compute_both_II()
    return ii_obj, ii_A2B, ii_B2A
# EOF


def dyn_compare_similarity_metrics(paths, rank, metrics_tuple, raster, k, monkey_name, date, brain_area, new_fs):
    save_name_A2B = f"{paths['data_path']}/results/metric_comparison_k{k}_{metrics_tuple[0]}-{metrics_tuple[1]}_{monkey_name}_{date}_{brain_area}_{new_fs}Hz.npz"
    save_name_B2A = f"{paths['data_path']}/results/metric_comparison_k{k}_{metrics_tuple[1]}-{metrics_tuple[0]}_{monkey_name}_{date}_{brain_area}_{new_fs}Hz.npz"
    if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
        print_wise(f"model already exists at {save_name_A2B}", rank=rank)
    else:
        A2B_list = []
        B2A_list = []
        for idx, resp_t in enumerate(raster):
            _, A2B, B2A =compare_similarity_metrics(resp_t, metrics_tuple[0], metrics_tuple[1], k)
            A2B_list.append(A2B)
            B2A_list.append(B2A)
        # end for idx, resp_t in enumerate(ba_raster):
        np.savez_compressed(save_name_A2B, np.stack(A2B_list))
        np.savez_compressed(save_name_B2A, np.stack(B2A_list))
        print_wise(f"comparison saved at {save_name_A2B}", rank=rank)
    # end if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
# EOF
