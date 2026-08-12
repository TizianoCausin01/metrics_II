import numpy as np
import pandas as pd

from useful_stuff.general_utils.II import InformationImbalance
from useful_stuff.general_utils.regression import linear_encoding


"""
generate_parabola_branch
Generate a one-dimensional parabola branch and a target space containing the
curved coordinate plus one independent orthogonal coordinate.

INPUT:
    - n_samples: int -> number of matched observations
    - latent_min: float -> lower endpoint of the non-negative latent interval
    - latent_max: float -> upper endpoint of the latent interval
    - orthogonal_noise_scale: float -> standard deviation of target-only noise
    - seed: int -> random seed

OUTPUT:
    - source: np.ndarray -> latent branch coordinate, shape (samples, 1)
    - target: np.ndarray -> squared coordinate and orthogonal noise, shape (samples, 2)
"""
def generate_parabola_branch(
    n_samples,
    latent_min=0.0,
    latent_max=1.0,
    orthogonal_noise_scale=0.25,
    seed=0,
):
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3.")
    # end if n_samples < 3
    if latent_min < 0 or latent_max <= latent_min:
        raise ValueError("Use a non-negative interval with latent_max > latent_min.")
    # end if latent interval is invalid
    if orthogonal_noise_scale < 0:
        raise ValueError("orthogonal_noise_scale must be non-negative.")
    # end if orthogonal_noise_scale < 0

    rng = np.random.default_rng(seed)
    latent = rng.uniform(latent_min, latent_max, size=n_samples)
    orthogonal_noise = orthogonal_noise_scale * rng.normal(size=n_samples)
    source = latent[:, np.newaxis]
    target = np.column_stack((latent**2, orthogonal_noise))
    return source, target
# EOF


"""
generate_gaussian_with_orthogonal_noise
Generate a two-dimensional Gaussian source, an invertible scaled copy in the
target, and one independent target-only coordinate.

INPUT:
    - n_samples: int -> number of matched observations
    - orthogonal_noise_scale: float -> standard deviation of target-only noise
    - signal_scales: tuple -> gains applied to the two recoverable source axes
    - seed: int -> random seed; reusing it across scales holds all draws fixed

OUTPUT:
    - source: np.ndarray -> standard-Gaussian source, shape (samples, 2)
    - target: np.ndarray -> recoverable block plus target-only noise, shape (samples, 3)
"""
def generate_gaussian_with_orthogonal_noise(
    n_samples,
    orthogonal_noise_scale,
    signal_scales=(2.0, 0.5),
    seed=0,
):
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4.")
    # end if n_samples < 4
    if orthogonal_noise_scale < 0:
        raise ValueError("orthogonal_noise_scale must be non-negative.")
    # end if orthogonal_noise_scale < 0

    signal_scales = np.asarray(signal_scales, dtype=float)
    if signal_scales.shape != (2,) or np.any(signal_scales == 0):
        raise ValueError("signal_scales must contain two nonzero gains.")
    # end if signal_scales is invalid

    rng = np.random.default_rng(seed)
    source = rng.normal(size=(n_samples, 2))
    target_signal = source * signal_scales
    orthogonal_noise = orthogonal_noise_scale * rng.normal(size=n_samples)
    target = np.column_stack((target_signal, orthogonal_noise))
    return source, target
# EOF


"""
validate_sample_spaces
Validate two matched matrices that store samples in rows and features in columns.

INPUT:
    - source: np.ndarray -> source representation, shape (samples, features)
    - target: np.ndarray -> target representation, shape (samples, features)

OUTPUT:
    - source: np.ndarray -> validated floating-point source matrix
    - target: np.ndarray -> validated floating-point target matrix
"""
def validate_sample_spaces(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be two-dimensional.")
    # end if source.ndim != 2 or target.ndim != 2
    if source.shape[0] != target.shape[0]:
        raise ValueError(
            "source and target must contain the same number of samples: "
            f"{source.shape[0]} and {target.shape[0]}."
        )
    # end if source and target have different sample counts
    if source.shape[0] < 3:
        raise ValueError("At least three matched samples are required.")
    # end if source.shape[0] < 3
    if source.shape[1] < 1 or target.shape[1] < 1:
        raise ValueError("source and target must each contain at least one feature.")
    # end if either space contains no features
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("source and target must contain only finite values.")
    # end if either space contains non-finite values
    return source, target
# EOF


"""
pooled_r2
Compute one multivariate R-squared by pooling squared error and centered target
variation across every output feature.

INPUT:
    - observed: np.ndarray -> observed outputs, shape (samples, outputs)
    - predicted: np.ndarray -> predicted outputs with the same shape

OUTPUT:
    - score: float -> pooled R-squared value
"""
def pooled_r2(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if observed.ndim != 2 or predicted.shape != observed.shape:
        raise ValueError("observed and predicted must have the same 2D shape.")
    # end if observed and predicted shapes are invalid

    observed_centered = observed - observed.mean(axis=0, keepdims=True)
    residual_sum_squares = np.sum((observed - predicted) ** 2)
    total_sum_squares = np.sum(observed_centered**2)
    if total_sum_squares == 0:
        raise ValueError("Pooled R-squared is undefined for constant outputs.")
    # end if total_sum_squares == 0
    return float(1 - residual_sum_squares / total_sum_squares)
# EOF


"""
fit_bidirectional_ols
Fit full-data ordinary least squares in both directions and retain predictions,
per-output scores, and pooled scores.

INPUT:
    - source: np.ndarray -> source representation, shape (samples, features)
    - target: np.ndarray -> target representation, shape (samples, features)

OUTPUT:
    - result: dict -> fitted encoders, predictions, and directional R-squared values
"""
def fit_bidirectional_ols(source, target):
    source, target = validate_sample_spaces(source, target)

    source_to_target = linear_encoding(
        regression_type="lr", cv_type="same", score_type="r2", shuffle=False
    )
    target_to_source = linear_encoding(
        regression_type="lr", cv_type="same", score_type="r2", shuffle=False
    )

    # transpose=False preserves the module convention (samples, features).
    source_to_target.fit(source, target, transpose=False)
    target_to_source.fit(target, source, transpose=False)
    target_hat = source_to_target.predict(
        source, transpose=False, transpose_output=False
    )
    source_hat = target_to_source.predict(
        target, transpose=False, transpose_output=False
    )

    source_to_target_r2 = source_to_target.score(
        source,
        target,
        y_hat=target_hat,
        transpose=False,
        transpose_prediction=False,
    )
    target_to_source_r2 = target_to_source.score(
        target,
        source,
        y_hat=source_hat,
        transpose=False,
        transpose_prediction=False,
    )

    return {
        "source_to_target": source_to_target,
        "target_to_source": target_to_source,
        "target_hat": target_hat,
        "source_hat": source_hat,
        "r2_source_to_target": source_to_target_r2,
        "r2_target_to_source": target_to_source_r2,
        "pooled_r2_source_to_target": pooled_r2(target, target_hat),
        "pooled_r2_target_to_source": pooled_r2(source, source_hat),
    }
# EOF


"""
compute_bidirectional_ii
Compute Information Imbalance directly between two matched representations.

INPUT:
    - source: np.ndarray -> source representation, shape (samples, features)
    - target: np.ndarray -> target representation, shape (samples, features)
    - source_metric: str -> distance metric used for the source RDM
    - target_metric: str -> distance metric used for the target RDM
    - k: int -> number of source-space nearest neighbors used by II

OUTPUT:
    - result: dict -> II object and both directional II values
"""
def compute_bidirectional_ii(
    source,
    target,
    source_metric="euclidean",
    target_metric="euclidean",
    k=1,
):
    source, target = validate_sample_spaces(source, target)
    if not 1 <= k < source.shape[0]:
        raise ValueError(f"k must be between 1 and {source.shape[0] - 1}.")
    # end if k is outside the valid range

    # InformationImbalance expects features in rows and matched samples in columns.
    ii_obj = InformationImbalance(source_metric, target_metric, k=k)
    ii_obj.compute_both_RDMs(source.T, target.T)
    ii_obj.compute_both_distance_ranks()
    source_to_target, target_to_source = ii_obj.compute_both_II()
    return {
        "object": ii_obj,
        "source_to_target": float(source_to_target),
        "target_to_source": float(target_to_source),
    }
# EOF


"""
sigma_inverse_row_transform
Use the compact SVD of a fitted coefficient matrix to remove its nonzero
singular-value gains. For W = U Sigma V.T, the returned row transform is
V I_r U.T = W.T U Sigma^-1 U.T.

INPUT:
    - encoding: linear_encoding -> fitted regression wrapper
    - relative_tolerance: float or None -> numerical-rank cutoff relative to
      the largest singular value; None uses NumPy's matrix-rank convention

OUTPUT:
    - result: dict -> row transform, singular values, Sigma inverse, and rank mask
"""
def sigma_inverse_row_transform(encoding, relative_tolerance=None):
    coefficient = np.asarray(encoding.get_weights(), dtype=float)
    if coefficient.ndim == 1:
        coefficient = coefficient[np.newaxis, :]
    # end if coefficient.ndim == 1

    U, singular_values, Vt = np.linalg.svd(coefficient, full_matrices=False)
    if relative_tolerance is None:
        relative_tolerance = max(coefficient.shape) * np.finfo(float).eps
    # end if relative_tolerance is None
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative.")
    # end if relative_tolerance < 0

    largest_singular_value = singular_values.max(initial=0.0)
    active = singular_values > relative_tolerance * largest_singular_value
    inverse_singular_values = np.zeros_like(singular_values)
    inverse_singular_values[active] = 1 / singular_values[active]
    sigma_inverse = np.diag(inverse_singular_values)

    # This is the stable form of W.T U Sigma^-1 U.T: Sigma cancels to I_r.
    unit_singular_values = np.diag(active.astype(float))
    row_transform = (U @ unit_singular_values @ Vt).T
    return {
        "row_transform": row_transform,
        "singular_values": singular_values,
        "sigma_inverse": sigma_inverse,
        "active": active,
        "rank": int(np.sum(active)),
    }
# EOF


"""
run_asymmetry_pipeline
Compute directional II and pooled R-squared before and after removing the
singular-value gains of both reciprocal OLS maps.

INPUT:
    - source: np.ndarray -> source representation, shape (samples, features)
    - target: np.ndarray -> target representation, shape (samples, features)
    - source_metric: str -> source-space RDM metric
    - target_metric: str -> target-space RDM metric
    - k: int -> number of nearest neighbors used by II
    - relative_tolerance: float or None -> numerical-rank cutoff for both SVDs

OUTPUT:
    - result: dict -> original and transformed spaces, metrics, fits, and SVD details
"""
def run_asymmetry_pipeline(
    source,
    target,
    source_metric="euclidean",
    target_metric="euclidean",
    k=1,
    relative_tolerance=None,
):
    source, target = validate_sample_spaces(source, target)

    original_fit = fit_bidirectional_ols(source, target)
    original_ii = compute_bidirectional_ii(
        source, target, source_metric, target_metric, k
    )

    source_svd = sigma_inverse_row_transform(
        original_fit["source_to_target"], relative_tolerance
    )
    target_svd = sigma_inverse_row_transform(
        original_fit["target_to_source"], relative_tolerance
    )

    # Each representation is expressed through its own reciprocal predictive map.
    transformed_source = source @ source_svd["row_transform"]
    transformed_target = target @ target_svd["row_transform"]

    transformed_fit = fit_bidirectional_ols(
        transformed_source, transformed_target
    )
    transformed_ii = compute_bidirectional_ii(
        transformed_source,
        transformed_target,
        source_metric,
        target_metric,
        k,
    )

    return {
        "source": source,
        "target": target,
        "original_fit": original_fit,
        "original_ii": original_ii,
        "source_svd": source_svd,
        "target_svd": target_svd,
        "transformed_source": transformed_source,
        "transformed_target": transformed_target,
        "transformed_fit": transformed_fit,
        "transformed_ii": transformed_ii,
    }
# EOF


"""
summarize_asymmetry_pipeline
Create one readable table with directional pooled R-squared and II at each stage.

INPUT:
    - result: dict -> output of run_asymmetry_pipeline
    - source_name: str -> display name for the source representation
    - target_name: str -> display name for the target representation

OUTPUT:
    - summary: pd.DataFrame -> four rows: two directions at two stages
"""
def summarize_asymmetry_pipeline(result, source_name="source", target_name="target"):
    rows = []
    stage_values = (
        ("original", result["original_fit"], result["original_ii"]),
        ("Sigma^-1 normalized", result["transformed_fit"], result["transformed_ii"]),
    )

    for stage, fit_result, ii_result in stage_values:
        for direction, per_output_r2, pooled_score, ii_score in (
            (
                f"{source_name} -> {target_name}",
                fit_result["r2_source_to_target"],
                fit_result["pooled_r2_source_to_target"],
                ii_result["source_to_target"],
            ),
            (
                f"{target_name} -> {source_name}",
                fit_result["r2_target_to_source"],
                fit_result["pooled_r2_target_to_source"],
                ii_result["target_to_source"],
            ),
        ):
            rows.append(
                {
                    "stage": stage,
                    "direction": direction,
                    "R2 per output": np.round(per_output_r2, 6).tolist(),
                    "pooled R2": pooled_score,
                    "II": ii_score,
                }
            )
        # end for direction, per_output_r2, pooled_score, ii_score
    # end for stage, fit_result, ii_result
    return pd.DataFrame(rows)
# EOF
