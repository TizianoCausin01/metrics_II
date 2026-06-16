import os, sys, yaml
from pathlib import Path
import numpy as np
ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
with open(PROJECT_ROOT / "config.yaml", "r") as f:
    config = yaml.safe_load(f)
paths = config[ENV]["paths"]
sys.path.append(paths["src_path"])
sys.path.append(paths["useful_stuff_path"])
from useful_stuff.general_utils.utils import TimeSeries, print_wise
from useful_stuff.general_utils.II import InformationImbalance, dynInformationImbalance
from useful_stuff.general_utils.RSA import RSA

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


"""
RSA_compare_similarity_metrics
Compute RSA similarity between two metrics used to build RDMs from the same data.

INPUT
- data : np.ndarray (D, N) -> Input data used to compute both RDMs.
- metric1, metric2 : str -> Similarity / distance metrics for the two RDMs.
- RSA_metric : str -> Similarity metric between RDM vectors.

OUTPUT:
- rsa_obj : RSA -> Initialized and fully computed RSA object.
- similarity : float -> RSA similarity between the two metric-defined RDMs.
"""
def RSA_compare_similarity_metrics(data: np.ndarray, metric1: str, metric2: str, RSA_metric: str = "correlation"):
    rsa_obj = RSA(metric1, metric2, RSA_metric=RSA_metric)
    rsa_obj.compute_RDM(data, "signal")
    rsa_obj.compute_RDM(data, "model")
    similarity = rsa_obj.compute_RSA()
    return rsa_obj, similarity
# EOF


"""
metric_comparison_save_name
Builds the output path for metric-comparison results.

INPUT:
- paths: dict -> dictionary containing the base data path.
- mt_A, mt_B: str -> metrics used for the A->B comparison.
- k: int -> number of nearest neighbors used for conditioning.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size, n_iterations: int or None -> optional subsampling parameters.

OUTPUT:
- save_name: str -> full .npz path where the comparison is saved.
"""
def metric_comparison_save_name(
    paths,
    mt_A,
    mt_B,
    k,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size=None,
    n_iterations=None,
):
    save_name = f"{paths['data_path']}/results/metric_comparison_k{k}_{mt_A}-{mt_B}_{monkey_name}_{date}_{brain_area}_{new_fs}Hz"
    if subsamples_size is not None:
        save_name += f"_{subsamples_size}subsamples"
    if n_iterations is not None:
        save_name += f"_{n_iterations}iterations"
    return f"{save_name}.npz"
# EOF


"""
RSA_metric_comparison_save_name
Builds the output path for RSA metric-comparison results.

INPUT:
- paths: dict -> dictionary containing the base data path.
- mt_A, mt_B: str -> metrics used to compute the compared RDMs.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size, n_iterations: int or None -> optional subsampling parameters.

OUTPUT:
- save_name: str -> full .npz path where the comparison is saved.
"""
def RSA_metric_comparison_save_name(
    paths,
    mt_A,
    mt_B,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size=None,
    n_iterations=None,
):
    save_name = f"{paths['data_path']}/results/RSA_metric_comparison_{mt_A}-{mt_B}_{monkey_name}_{date}_{brain_area}_{new_fs}Hz"
    if subsamples_size is not None:
        save_name += f"_{subsamples_size}subsamples"
    if n_iterations is not None:
        save_name += f"_{n_iterations}iterations"
    return f"{save_name}.npz"
# EOF


"""
save_metric_comparison
Saves metric-comparison values using the standard filename convention.

INPUT:
- values: np.ndarray -> comparison values to save.
- paths: dict -> dictionary containing the base data path.
- mt_A, mt_B: str -> metrics used for the A->B comparison.
- k: int -> number of nearest neighbors used for conditioning.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size, n_iterations: int or None -> optional subsampling parameters.

OUTPUT:
- save_name: str -> full .npz path where values were saved.
"""
def save_metric_comparison(
    values,
    paths,
    mt_A,
    mt_B,
    k,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size=None,
    n_iterations=None,
):
    save_name = metric_comparison_save_name(
        paths,
        mt_A,
        mt_B,
        k,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.savez_compressed(save_name, values)
    return save_name
# EOF


"""
save_RSA_metric_comparison
Saves RSA metric-comparison values using the standard filename convention.

INPUT:
- values: np.ndarray -> comparison values to save.
- paths: dict -> dictionary containing the base data path.
- mt_A, mt_B: str -> metrics used to compute the compared RDMs.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size, n_iterations: int or None -> optional subsampling parameters.

OUTPUT:
- save_name: str -> full .npz path where values were saved.
"""
def save_RSA_metric_comparison(
    values,
    paths,
    mt_A,
    mt_B,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size=None,
    n_iterations=None,
):
    save_name = RSA_metric_comparison_save_name(
        paths,
        mt_A,
        mt_B,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    np.savez_compressed(save_name, values)
    return save_name
# EOF


"""
dyn_compare_similarity_metrics
Computes dynamic Information Imbalance between two metrics across all time points.

INPUT:
- paths: dict -> dictionary containing the base data path.
- rank: int -> process rank used for printing.
- metrics_tuple: tuple[str, str] -> metrics compared in both directions.
- raster: TimeSeries -> neural responses iterated across time.
- k: int -> number of nearest neighbors used for conditioning.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.

OUTPUT:
- Saves two .npz files with A->B and B->A dynamic II values.
"""
def dyn_compare_similarity_metrics(paths, rank, metrics_tuple, raster, k, monkey_name, date, brain_area, new_fs):
    mt_A, mt_B = metrics_tuple
    save_name_A2B = metric_comparison_save_name(paths, mt_A, mt_B, k, monkey_name, date, brain_area, new_fs)
    save_name_B2A = metric_comparison_save_name(paths, mt_B, mt_A, k, monkey_name, date, brain_area, new_fs)
    if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
        print_wise(f"model already exists at {save_name_A2B}", rank=rank)
    else:
        A2B_list = []
        B2A_list = []
        for idx, resp_t in enumerate(raster):
            _, A2B, B2A = compare_similarity_metrics(resp_t, mt_A, mt_B, k)
            A2B_list.append(A2B)
            B2A_list.append(B2A)
        # end for idx, resp_t in enumerate(ba_raster):
        save_metric_comparison(np.stack(A2B_list), paths, mt_A, mt_B, k, monkey_name, date, brain_area, new_fs)
        save_metric_comparison(np.stack(B2A_list), paths, mt_B, mt_A, k, monkey_name, date, brain_area, new_fs)
        print_wise(f"comparison saved at {save_name_A2B}", rank=rank)
    # end if os.path.exists(save_name_A2B) and os.path.exists(save_name_B2A):
# EOF


"""
dyn_RSA_compare_similarity_metrics
Computes dynamic RSA similarity between two metrics across all time points.

INPUT:
- paths: dict -> dictionary containing the base data path.
- rank: int -> process rank used for printing.
- metrics_tuple: tuple[str, str] -> metrics whose RDMs are compared.
- raster: TimeSeries -> neural responses iterated across time.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- RSA_metric: str -> similarity metric between RDM vectors.

OUTPUT:
- Saves one .npz file with dynamic RSA values.
"""
def dyn_RSA_compare_similarity_metrics(
    paths,
    rank,
    metrics_tuple,
    raster,
    monkey_name,
    date,
    brain_area,
    new_fs,
    RSA_metric="correlation",
):
    mt_A, mt_B = metrics_tuple
    save_name = RSA_metric_comparison_save_name(
        paths,
        mt_A,
        mt_B,
        monkey_name,
        date,
        brain_area,
        new_fs,
    )
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return

    RSA_list = []
    for resp_t in raster:
        _, similarity = RSA_compare_similarity_metrics(resp_t, mt_A, mt_B, RSA_metric=RSA_metric)
        RSA_list.append(similarity)
    save_RSA_metric_comparison(
        np.stack(RSA_list),
        paths,
        mt_A,
        mt_B,
        monkey_name,
        date,
        brain_area,
        new_fs,
    )
    print_wise(f"comparison saved at {save_name}", rank=rank)
# EOF


"""
dyn_RSA_compare_similarity_metrics_subsampled
Computes dynamic RSA similarity between two metrics on random trial subsamples
and saves the mean timecourse.

INPUT:
- paths: dict -> dictionary containing the base data path.
- rank: int -> process rank used for printing.
- metrics_tuple: tuple[str, str] -> metrics whose RDMs are compared.
- raster: TimeSeries -> neural responses to subsample and iterate across time.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size: int -> number of trials sampled without replacement.
- n_iterations: int -> number of random subsampling iterations.
- random_seed: int or None -> seed used by the random number generator.
- RSA_metric: str -> similarity metric between RDM vectors.

OUTPUT:
- Saves one .npz file with the mean dynamic RSA values.
"""
def dyn_RSA_compare_similarity_metrics_subsampled(
    paths,
    rank,
    metrics_tuple,
    raster,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size,
    n_iterations,
    random_seed=None,
    RSA_metric="correlation",
):
    mt_A, mt_B = metrics_tuple
    save_name = RSA_metric_comparison_save_name(
        paths,
        mt_A,
        mt_B,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    if os.path.exists(save_name):
        print_wise(f"model already exists at {save_name}", rank=rank)
        return

    raster_array = raster.get_array()
    n_trials = raster_array.shape[2]
    if subsamples_size > n_trials:
        raise ValueError(
            f"subsamples_size={subsamples_size} exceeds available trials ({n_trials})"
        )
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")

    rng = np.random.default_rng(random_seed)
    RSA_iterations = []
    for _ in range(n_iterations):
        subset = rng.choice(n_trials, size=subsamples_size, replace=False)
        subset_raster = TimeSeries(raster_array[:, :, subset], raster.get_fs())
        RSA_list = []
        for resp_t in subset_raster:
            _, similarity = RSA_compare_similarity_metrics(resp_t, mt_A, mt_B, RSA_metric=RSA_metric)
            RSA_list.append(similarity)
        RSA_iterations.append(np.stack(RSA_list))

    save_RSA_metric_comparison(
        np.mean(RSA_iterations, axis=0),
        paths,
        mt_A,
        mt_B,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    print_wise(f"comparison saved at {save_name}", rank=rank)
# EOF


"""
dyn_compare_similarity_metrics_subsampled
Computes dynamic Information Imbalance on random trial subsamples and saves the mean.

INPUT:
- paths: dict -> dictionary containing the base data path.
- rank: int -> process rank used for printing.
- metrics_tuple: tuple[str, str] -> metrics compared in both directions.
- raster: TimeSeries -> neural responses to subsample and iterate across time.
- k: int -> number of nearest neighbors used for conditioning.
- monkey_name, date, brain_area: str -> session identifiers.
- new_fs: float -> raster sampling frequency.
- subsamples_size: int -> number of trials sampled without replacement.
- n_iterations: int -> number of random subsampling iterations.
- random_seed: int -> seed used by the random number generator.

OUTPUT:
- Saves two .npz files with mean A->B and B->A dynamic II values.
"""
def dyn_compare_similarity_metrics_subsampled(
    paths,
    rank,
    metrics_tuple,
    raster,
    k,
    monkey_name,
    date,
    brain_area,
    new_fs,
    subsamples_size,
    n_iterations,
    random_seed=0,
):
    mt_A, mt_B = metrics_tuple
    save_name_A2B = metric_comparison_save_name(
        paths,
        mt_A,
        mt_B,
        k,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    save_name_B2A = metric_comparison_save_name(
        paths,
        mt_B,
        mt_A,
        k,
        monkey_name,
        date,
        brain_area,
        new_fs,
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

    rng = np.random.default_rng(random_seed)
    A2B_iterations = []
    B2A_iterations = []
    for _ in range(n_iterations):
        subset = rng.choice(n_trials, size=subsamples_size, replace=False)
        subset_raster = TimeSeries(raster_array[:, :, subset], raster.fs)
        A2B_list = []
        B2A_list = []
        for resp_t in subset_raster:
            _, A2B, B2A = compare_similarity_metrics(resp_t, mt_A, mt_B, k)
            A2B_list.append(A2B)
            B2A_list.append(B2A)
        A2B_iterations.append(np.stack(A2B_list))
        B2A_iterations.append(np.stack(B2A_list))

    save_metric_comparison(
        np.mean(A2B_iterations, axis=0),
        paths,
        mt_A,
        mt_B,
        k,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    save_metric_comparison(
        np.mean(B2A_iterations, axis=0),
        paths,
        mt_B,
        mt_A,
        k,
        monkey_name,
        date,
        brain_area,
        new_fs,
        subsamples_size=subsamples_size,
        n_iterations=n_iterations,
    )
    print_wise(f"comparison saved at {save_name_A2B}", rank=rank)
# EOF
