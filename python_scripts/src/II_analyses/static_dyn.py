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


def compute_static_dynII(
    paths: dict[str, str],
    rank: int,
    layer_name: str,
    dyn_ii_obj: "dynInformationImbalance",
    idx_ord: np.ndarray,
    monkey_name: str,
    date: str,
    brain_area: str,
    folder_name: str,
    model_name: str,
    img_size: int,
    pooling: str,
) -> tuple["TimeSeries", "TimeSeries"]:
    fs = dyn_ii_obj.get_RDM_timeseries("signal").get_fs()
    save_name_A2B = f"{paths['data_path']}/results/dynII_A2B_k{dyn_ii_obj.k}_{dyn_ii_obj.signal_RDM_metric}-{dyn_ii_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{fs}Hz.npz"
    save_name_B2A = f"{paths['data_path']}/results/dynII_B2A_k{dyn_ii_obj.k}_{dyn_ii_obj.signal_RDM_metric}-{dyn_ii_obj.model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{fs}Hz.npz"
    if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
        print_wise(f"model already exists at {save_name_A2B}", rank=rank)
        return
    if not hasattr(dyn_ii_obj, "signal_distance_ranks_timeseries"):
        raise AttributeError("dyn_ii_obj must have 'signal_distance_ranks_timeseries'")
    # end if not hasattr(dyn_ii_obj, "signal_distance_ranks_timeseries"):
    feats_filename = f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    features = np.load(feats_filename)["arr_0"][:, idx_ord]
    dyn_ii_obj.compute_RDM(features, "model")
    dyn_ii_obj.compute_distance_ranks("model")
    dyn_ii_A2B, dyn_ii_B2A = dyn_ii_obj.compute_both_static_dynII()
    np.savez_compressed(save_name_A2B, dyn_ii_A2B.get_array())
    np.savez_compressed(save_name_B2A, dyn_ii_B2A.get_array())
    print_wise(f"model saved at {save_name_A2B}", rank=rank)
    return dyn_ii_A2B, dyn_ii_B2A
# EOF


def static_dynII_save_name(
    paths,
    direction,
    k,
    signal_RDM_metric,
    model_RDM_metric,
    monkey_name,
    date,
    brain_area,
    model_name,
    img_size,
    layer_name,
    fs,
    subsamples_size=None,
    n_iterations=None,
):
    save_name = f"{paths['data_path']}/results/dynII_{direction}_k{k}_{signal_RDM_metric}-{model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{fs}Hz"
    if subsamples_size is not None:
        save_name += f"_{subsamples_size}subsamples"
    if n_iterations is not None:
        save_name += f"_{n_iterations}iterations"
    return f"{save_name}.npz"
# EOF


def compute_static_dynII_subsampled(
    paths: dict[str, str],
    rank: int,
    layer_name: str,
    raster: "TimeSeries",
    idx_ord: np.ndarray,
    signal_RDM_metric: str,
    model_RDM_metric: str,
    k: int,
    monkey_name: str,
    date: str,
    brain_area: str,
    folder_name: str,
    model_name: str,
    img_size: int,
    pooling: str,
    subsamples_size: int,
    n_iterations: int,
    random_seed: int = 0,
) -> tuple["TimeSeries", "TimeSeries"]:
    save_name_A2B = static_dynII_save_name(
        paths,
        "A2B",
        k,
        signal_RDM_metric,
        model_RDM_metric,
        monkey_name,
        date,
        brain_area,
        model_name,
        img_size,
        layer_name,
        raster.get_fs(),
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    save_name_B2A = static_dynII_save_name(
        paths,
        "B2A",
        k,
        signal_RDM_metric,
        model_RDM_metric,
        monkey_name,
        date,
        brain_area,
        model_name,
        img_size,
        layer_name,
        raster.get_fs(),
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
        print_wise(f"model already exists at {save_name_A2B}", rank=rank)
        return

    raster_array = raster.get_array()
    n_trials = raster_array.shape[2]
    if subsamples_size > n_trials:
        raise ValueError(
            f"subsamples_size={subsamples_size} exceeds available trials ({n_trials})"
        )
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    feats_filename = f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    features = np.load(feats_filename)["arr_0"][:, idx_ord]

    rng = np.random.default_rng(random_seed)
    A2B_iterations = []
    B2A_iterations = []
    for _ in range(n_iterations):
        subset = rng.choice(n_trials, size=subsamples_size, replace=False)
        subset_raster = TimeSeries(raster_array[:, :, subset], raster.get_fs())
        subset_features = features[:, subset]
        dyn_ii_obj = init_static_dynII(
            subset_raster,
            signal_RDM_metric,
            model_RDM_metric,
            k,
        )
        dyn_ii_obj.compute_RDM(subset_features, "model")
        dyn_ii_obj.compute_distance_ranks("model")
        dyn_ii_A2B, dyn_ii_B2A = dyn_ii_obj.compute_both_static_dynII()
        A2B_iterations.append(dyn_ii_A2B.get_array())
        B2A_iterations.append(dyn_ii_B2A.get_array())

    mean_A2B = TimeSeries(np.mean(A2B_iterations, axis=0), raster.get_fs())
    mean_B2A = TimeSeries(np.mean(B2A_iterations, axis=0), raster.get_fs())
    np.savez_compressed(save_name_A2B, mean_A2B.get_array())
    np.savez_compressed(save_name_B2A, mean_B2A.get_array())
    print_wise(f"model saved at {save_name_A2B}", rank=rank)
    return mean_A2B, mean_B2A
# EOF
