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
from useful_stuff.general_utils.CKA import dCKA
from useful_stuff.general_utils.regression import dyn_linear_encoding


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


"""
init_static_dCKA
Initializes static dCKA and computes the dynamic neural Gram matrices.

INPUT:
    - ba_raster: TimeSeries -> neural responses with shape (features, time, samples).
    - signal_metric, model_metric: str -> neural and model Gram metrics.
    - method: str -> HSIC estimator, either "biased" or "unbiased".
    - signal_metric_type, model_metric_type: str -> "kernel" or "distance".

OUTPUT:
    - dcka_obj: dCKA -> initialized object containing neural Gram matrices over time.
"""
def init_static_dCKA(
    ba_raster: "TimeSeries",
    signal_metric: str,
    model_metric: str,
    method: str,
    signal_metric_type: str,
    model_metric_type: str,
) -> "dCKA":
    dcka_obj = dCKA(
        signal_metric=signal_metric,
        model_metric=model_metric,
        method=method,
        signal_metric_type=signal_metric_type,
        model_metric_type=model_metric_type,
    )
    dcka_obj.compute_gram_timeseries(ba_raster, "signal")
    return dcka_obj
# EOF


"""
static_dRSA_save_name
Builds the static-dRSA result filename, including optional subsampling settings.

INPUT:
    - paths: dict -> project paths containing data_path.
    - signal_RDM_metric, model_RDM_metric: str -> metrics used to build the RDMs.
    - monkey_name, date, brain_area: str -> neural recording identifiers.
    - model_name, layer_name: str -> computational model identifiers.
    - img_size: int -> model input image size.
    - fs: int or float -> neural sampling frequency.
    - subsamples_size, n_iterations: int or None -> optional subsampling settings.

OUTPUT:
    - save_name: str -> full path of the compressed NumPy result file.
"""
def static_dRSA_save_name(
    paths,
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
    save_name = f"{paths['data_path']}/results/static_dRSA_{signal_RDM_metric}-{model_RDM_metric}_{monkey_name}_{date}_{brain_area}_{model_name}_{img_size}_{layer_name}_{fs}Hz"
    if subsamples_size is not None:
        save_name += f"_{subsamples_size}subsamples"
    # end if subsamples_size is not None
    if n_iterations is not None:
        save_name += f"_{n_iterations}iterations"
    # end if n_iterations is not None
    return f"{save_name}.npz"
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
    save_name = static_dRSA_save_name(
        paths,
        drsa_obj.signal_RDM_metric,
        drsa_obj.model_RDM_metric,
        monkey_name,
        date,
        brain_area,
        model_name,
        img_size,
        layer_name,
        drsa_obj.get_RDM_timeseries("signal").get_fs(),
    )
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


"""
compute_static_dRSA_subsampled
Computes static dRSA on random trial subsets and saves the mean time course.

INPUT:
    - paths: dict -> project paths containing model and result directories.
    - rank: int -> MPI worker rank used for logging.
    - layer_name: str -> model layer processed by this worker task.
    - raster: TimeSeries -> neural responses with trials/images on the last axis.
    - idx_ord: np.ndarray -> mapping from model-feature order to neural trial order.
    - signal_RDM_metric, model_RDM_metric: str -> metrics used to build the RDMs.
    - monkey_name, date, brain_area: str -> neural recording identifiers.
    - folder_name, model_name: str -> stimulus and computational model identifiers.
    - img_size: int -> model input image size.
    - pooling: str -> feature pooling method used in the saved model features.
    - subsamples_size: int -> number of trials sampled without replacement.
    - n_iterations: int -> number of random subsampling iterations.
    - random_seed: int or None -> seed controlling the repeated trial subsets.

OUTPUT:
    - mean_dRSA: TimeSeries -> static-dRSA time course averaged across iterations.
"""
def compute_static_dRSA_subsampled(
    paths: dict[str, str],
    rank: int,
    layer_name: str,
    raster: "TimeSeries",
    idx_ord: np.ndarray,
    signal_RDM_metric: str,
    model_RDM_metric: str,
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
) -> "TimeSeries":
    # Keep the non-subsampled naming convention and append sampling metadata.
    save_name = static_dRSA_save_name(
        paths,
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
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return
    # end if os.path.exists(save_name)

    # Validate the requested sampling against the neural trial axis.
    raster_array = raster.get_array()
    n_trials = raster_array.shape[2]
    if subsamples_size > n_trials:
        raise ValueError(
            f"subsamples_size={subsamples_size} exceeds available trials ({n_trials})"
        )
    # end if subsamples_size > n_trials
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")
    # end if n_iterations < 1

    # Load this layer's features once and align them to the neural image order.
    feats_filename = f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    features = np.load(feats_filename)["arr_0"][:, idx_ord]

    # Use matching neural and model subsets in every iteration.
    rng = np.random.default_rng(random_seed)
    dRSA_iterations = []
    for _ in range(n_iterations):
        subset = rng.choice(n_trials, size=subsamples_size, replace=False)
        subset_raster = TimeSeries(raster_array[:, :, subset], raster.get_fs())
        subset_features = features[:, subset]

        drsa_obj = init_static_dRSA(
            subset_raster,
            signal_RDM_metric,
            model_RDM_metric,
        )
        drsa_obj.compute_RDM(subset_features, "model")
        static_dRSA = drsa_obj.compute_static_dRSA()
        dRSA_iterations.append(static_dRSA.get_array())
    # end for _ in range(n_iterations)

    # Average iteration curves without changing the original sampling frequency.
    mean_dRSA = TimeSeries(np.mean(dRSA_iterations, axis=0), raster.get_fs())
    np.savez_compressed(save_name, mean_dRSA.get_array())
    print_wise(f"model saved at {save_name}", rank=rank)
    return mean_dRSA
# EOF


"""
static_dCKA_save_name
Builds a static-dCKA result filename containing all CKA and subsampling settings.

INPUT:
    - paths: dict -> project paths containing data_path.
    - signal_metric, model_metric: str -> neural and model Gram metrics.
    - method: str -> HSIC estimator used for CKA.
    - signal_metric_type, model_metric_type: str -> Gram construction approaches.
    - monkey_name, date, brain_area: str -> neural recording identifiers.
    - model_name, layer_name: str -> computational model identifiers.
    - img_size: int -> model input image size.
    - fs: int or float -> neural sampling frequency.
    - subsamples_size, n_iterations: int or None -> optional subsampling settings.

OUTPUT:
    - save_name: str -> full path of the compressed NumPy result file.
"""
def static_dCKA_save_name(
    paths,
    signal_metric,
    model_metric,
    method,
    signal_metric_type,
    model_metric_type,
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
    save_name = (
        f"{paths['data_path']}/results/static_dCKA_{method}_"
        f"{signal_metric_type}-{signal_metric}_"
        f"{model_metric_type}-{model_metric}_{monkey_name}_{date}_{brain_area}_"
        f"{model_name}_{img_size}_{layer_name}_{fs}Hz"
    )
    if subsamples_size is not None:
        save_name += f"_{subsamples_size}subsamples"
    # end if subsamples_size is not None
    if n_iterations is not None:
        save_name += f"_{n_iterations}iterations"
    # end if n_iterations is not None
    return f"{save_name}.npz"
# EOF


"""
compute_static_dCKA_subsampled
Computes static dCKA on random trial subsets and saves the mean time course.

INPUT:
    - paths: dict -> project paths containing model and result directories.
    - rank: int -> MPI worker rank used for logging.
    - layer_name: str -> model layer processed by this worker task.
    - raster: TimeSeries -> neural responses with trials/images on the last axis.
    - idx_ord: np.ndarray -> mapping from model-feature order to neural trial order.
    - signal_metric, model_metric: str -> neural and model Gram metrics.
    - method: str -> HSIC estimator, either "biased" or "unbiased".
    - signal_metric_type, model_metric_type: str -> "kernel" or "distance".
    - monkey_name, date, brain_area: str -> neural recording identifiers.
    - folder_name, model_name: str -> stimulus and computational model identifiers.
    - img_size: int -> model input image size.
    - pooling: str -> feature pooling method used in the saved model features.
    - subsamples_size: int -> number of trials sampled without replacement.
    - n_iterations: int -> number of random subsampling iterations.
    - random_seed: int or None -> seed controlling the repeated trial subsets.

OUTPUT:
    - mean_dCKA: TimeSeries -> static-dCKA time course averaged across iterations.
"""
def compute_static_dCKA_subsampled(
    paths: dict[str, str],
    rank: int,
    layer_name: str,
    raster: "TimeSeries",
    idx_ord: np.ndarray,
    signal_metric: str,
    model_metric: str,
    method: str,
    signal_metric_type: str,
    model_metric_type: str,
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
) -> "TimeSeries":
    # Encode all CKA settings in the filename to prevent ambiguous results.
    save_name = static_dCKA_save_name(
        paths,
        signal_metric,
        model_metric,
        method,
        signal_metric_type,
        model_metric_type,
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
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return
    # end if os.path.exists(save_name)

    # Validate the neural trial axis before loading the model features.
    raster_array = raster.get_array()
    if raster_array.ndim != 3:
        raise ValueError(
            "Expected raster shape (features, timepoints, trials), "
            f"received {raster_array.shape}."
        )
    # end if raster_array.ndim != 3

    n_trials = raster_array.shape[2]
    minimum_samples = 4 if method == "unbiased" else 2
    if subsamples_size < minimum_samples:
        raise ValueError(
            f"method={method} requires at least {minimum_samples} samples, "
            f"received subsamples_size={subsamples_size}"
        )
    # end if subsamples_size < minimum_samples
    if subsamples_size > n_trials:
        raise ValueError(
            f"subsamples_size={subsamples_size} exceeds available trials ({n_trials})"
        )
    # end if subsamples_size > n_trials
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")
    # end if n_iterations < 1

    # Load this layer's features once and align them to the neural image order.
    feats_filename = f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_{layer_name}_features_{pooling}pool.npz"
    features = np.load(feats_filename)["arr_0"][:, idx_ord]
    if features.shape[1] != n_trials:
        raise ValueError(
            "Model features and neural raster have different trial counts: "
            f"{features.shape[1]} and {n_trials}."
        )
    # end if features.shape[1] != n_trials

    # Compute CKA from matching neural and model subsets in every iteration.
    rng = np.random.default_rng(random_seed)
    dCKA_iterations = []
    for _ in range(n_iterations):
        subset = rng.choice(n_trials, size=subsamples_size, replace=False)
        subset_raster = TimeSeries(raster_array[:, :, subset], raster.get_fs())
        subset_features = features[:, subset]

        dcka_obj = init_static_dCKA(
            subset_raster,
            signal_metric,
            model_metric,
            method,
            signal_metric_type,
            model_metric_type,
        )
        dcka_obj.compute_static_model_gram(subset_features)
        static_dCKA = dcka_obj.compute_static_dCKA()
        dCKA_iterations.append(static_dCKA.get_array())
    # end for _ in range(n_iterations)

    # Average iteration curves without changing the original sampling frequency.
    mean_dCKA = TimeSeries(np.mean(dCKA_iterations, axis=0), raster.get_fs())
    np.savez_compressed(save_name, mean_dCKA.get_array())
    print_wise(f"model saved at {save_name}", rank=rank)
    return mean_dCKA
# EOF


def preprocess_population_vectors(data, normalize=False, feature_center=False, mean_center=False):
    """Preprocess a (features, images) population-vector matrix."""
    data = np.asarray(data, dtype=float).copy()
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D features-by-images array, got {data.shape}")
    if feature_center:  # cosine_cnt: center every feature across images
        data -= data.mean(axis=1, keepdims=True)
    if mean_center:  # correlation: center each population vector across features
        data -= data.mean(axis=0, keepdims=True)
    if normalize:  # cosine: unit L2 norm for each population vector
        norms = np.linalg.norm(data, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        data /= norms
    return data
# EOF


def compute_static_linear_encoding(
    paths,
    rank,
    layer_name,
    raster,
    idx_ord,
    monkey_name,
    date,
    brain_area,
    folder_name,
    model_name,
    img_size,
    pooling,
    regression_type,
    score_type,
    alphas,
    cv_type,
    n_splits,
    normalize,
    feature_center,
    mean_center,
):
    """Cross-validate static model features against neural activity over time."""
    preprocessing = []
    if normalize:
        preprocessing.append("norm")
    if feature_center:
        preprocessing.append("feat_cnt")
    if mean_center:
        preprocessing.append("mean_cnt")
    preprocessing_label = "_".join(preprocessing) if preprocessing else "raw"
    alpha_label = f"{np.min(alphas):g}to{np.max(alphas):g}_n{len(alphas)}"
    fs = raster.get_fs()
    save_name = (
        f"{paths['data_path']}/results/static_linear_encoding_"
        f"{regression_type}_{score_type}_{cv_type}_alpha{alpha_label}_"
        f"{preprocessing_label}_{monkey_name}_{date}_{brain_area}_{model_name}_"
        f"{img_size}_{layer_name}_{fs}Hz.npz"
    )
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return

    features_path = (
        f"{paths['data_path']}/models/{folder_name}_{model_name}_{img_size}_"
        f"{layer_name}_features_{pooling}pool.npz"
    )
    features = np.load(features_path)["arr_0"][:, idx_ord]
    features = preprocess_population_vectors(
        features, normalize, feature_center, mean_center
    )
    raster_array = raster.get_array()
    if raster_array.ndim != 3:
        raise ValueError(
            "Expected neural raster data with shape (neurons, timepoints, images)."
        )
    if features.shape[1] != raster_array.shape[2]:
        raise ValueError(
            "Model features and neural raster have different image counts: "
            f"{features.shape[1]} and {raster_array.shape[2]}."
        )
    processed_raster = np.stack(
        [
            preprocess_population_vectors(
                raster_array[:, timepoint, :], normalize, feature_center, mean_center
            )
            for timepoint in range(raster_array.shape[1])
        ],
        axis=1,
    )
    encoding = dyn_linear_encoding(
        regression_type=regression_type,
        cv_type=cv_type,
        max_lag=0,
        alphas=np.asarray(alphas, dtype=float),
        score_type=score_type,
        n_splits=n_splits,
    )
    time_scores = []
    n_timepoints = processed_raster.shape[1]
    for timepoint in range(n_timepoints):
        print_wise(
            f"{layer_name}: timepoint {timepoint + 1}/{n_timepoints}", rank=rank
        )
        time_score = encoding.crossvalidate_static_dyn(
            features, TimeSeries(processed_raster[:, timepoint:timepoint + 1, :], fs)
        )
        time_scores.append(time_score.get_array()[..., 0])
        print_wise(
            f"{layer_name}: completed timepoint {timepoint + 1}/{n_timepoints}",
            rank=rank,
        )
    scores = TimeSeries(np.stack(time_scores, axis=-1), fs)
    np.savez_compressed(save_name, scores.get_array())
    print_wise(f"model saved at {save_name}", rank=rank)
    return scores
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
