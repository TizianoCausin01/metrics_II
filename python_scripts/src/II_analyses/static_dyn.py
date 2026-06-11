import os, sys, yaml
from pathlib import Path
import numpy as np
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
if "useful_stuff_path" in paths:
    sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries, print_wise
from useful_stuff.general_utils.II import InformationImbalance, dynInformationImbalance
from useful_stuff.general_utils.RSA import dRSA


def init_static_dynII(ba_raster: "TimeSeries", signal_RDM_metric, model_RDM_metric, k) -> "dynInformationImbalance":
    dyn_ii_obj = dynInformationImbalance(signal_RDM_metric, model_RDM_metric, k)
    dyn_ii_obj.compute_RDM_timeseries(ba_raster, "signal")
    dyn_ii_obj.compute_distance_ranks_timeseries("signal")
    return  dyn_ii_obj
# EOF


def init_static_dRSA(ba_raster: "TimeSeries", signal_RDM_metric, model_RDM_metric) -> "dRSA":
    drsa_obj = dRSA(signal_RDM_metric, model_RDM_metric)
    drsa_obj.compute_RDM_timeseries(ba_raster, "signal")
    return drsa_obj
# EOF


def compute_static_dRSA(
    paths: dict[str, str],
    rank: int,
    layer_name: str,
    drsa_obj: "dRSA",
    idx_ord: np.ndarray,
    monkey_name: str,
    date: str,
    brain_area: str,
    folder_name: str,
    model_name: str,
    img_size: int,
    pooling: str,
) -> "TimeSeries":
    save_name = f"{paths['data_path']}/results/static_dRSA_{drsa_obj.signal_RDM_metric}-{drsa_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{drsa_obj.get_RDM_timeseries('signal').get_fs()}Hz.npz"
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return
    if not hasattr(drsa_obj, "signal_RDM_timeseries"):
        raise AttributeError("drsa_obj must have 'signal_RDM_timeseries'")
    # end if not hasattr(drsa_obj, "signal_RDM_timeseries"):
    feats_filename = f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    features = np.load(feats_filename)["arr_0"][:, idx_ord]
    drsa_obj.compute_RDM(features, "model")
    drsa = drsa_obj.compute_static_dRSA()
    np.savez_compressed(save_name, drsa.get_array())
    print_wise(f"model saved at {save_name}", rank=rank)
    return drsa
# EOF
