import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA

from useful_stuff.general_utils.II import InformationImbalance
from useful_stuff.general_utils.regression import linear_encoding
from useful_stuff.general_utils.utils import create_RDM


# create_RDM returns 1 - cosine for the cosine family, and 1 - cos(u, v) is
# already half the squared Euclidean distance between the normalized (and, for
# the centered variants, pre-centered) vectors. Those RDMs therefore enter
# classical MDS as squared distances, whereas "euclidean" and "magnitude_diff"
# return plain distances that must be squared first.
RDM_IS_SQUARED_DISTANCE = {
    "euclidean": False,
    "magnitude_diff": False,
    "cosine": True,
    "cosine_cnt": True,
    "cosine_mean_cnt": True,
    "cosine_double_cnt": True,
    "correlation": True,
}


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
generate_variance_split_gaussians
Generate two independent-looking 2D Gaussian spaces that share a single
coordinate. The source is X = [x1, x2] with a prescribed share of its variance
on x1, and the target is Y = [y1, shared_gain * x1], where y1 is an independent
Gaussian scaled so that it carries a prescribed share of the variance of Y.
The only information common to both spaces is therefore x1, the dominant
coordinate of X but, when target_leading_share is large, a minor one in Y.

INPUT:
    - n_samples: int -> number of matched observations
    - source_leading_share: float -> fraction of the variance of X carried by x1, in (0, 1)
    - target_leading_share: float -> fraction of the variance of Y carried by y1, in (0, 1)
    - shared_gain: float -> gain applied to x1 inside Y
    - source_total_variance: float -> total variance of X, split between x1 and x2
    - seed: int -> random seed

OUTPUT:
    - source: np.ndarray -> X = [x1, x2], shape (samples, 2)
    - target: np.ndarray -> Y = [y1, shared_gain * x1], shape (samples, 2)
    - variances: dict -> the theoretical variance of every coordinate
"""
def generate_variance_split_gaussians(
    n_samples,
    source_leading_share=0.9,
    target_leading_share=0.9,
    shared_gain=1.0,
    source_total_variance=1.0,
    seed=0,
):
    if n_samples < 4:
        raise ValueError("n_samples must be at least 4.")
    # end if n_samples < 4
    if not 0 < source_leading_share < 1:
        raise ValueError("source_leading_share must lie in (0, 1).")
    # end if source_leading_share is invalid
    if not 0 < target_leading_share < 1:
        raise ValueError("target_leading_share must lie in (0, 1).")
    # end if target_leading_share is invalid
    if shared_gain == 0:
        raise ValueError("shared_gain must be nonzero, otherwise Y ignores x1.")
    # end if shared_gain == 0
    if source_total_variance <= 0:
        raise ValueError("source_total_variance must be positive.")
    # end if source_total_variance <= 0

    # Split the variance of X between its leading and its minor coordinate.
    var_x1 = source_leading_share * source_total_variance
    var_x2 = (1 - source_leading_share) * source_total_variance

    # Y inherits x1 through shared_gain, so requiring y1 to hold
    # target_leading_share of the variance of Y fixes the variance of y1:
    # var_y1 / (var_y1 + gain^2 var_x1) = target_leading_share.
    var_shared = shared_gain**2 * var_x1
    var_y1 = target_leading_share / (1 - target_leading_share) * var_shared

    rng = np.random.default_rng(seed)
    x1 = rng.normal(scale=np.sqrt(var_x1), size=n_samples)
    x2 = rng.normal(scale=np.sqrt(var_x2), size=n_samples)
    y1 = rng.normal(scale=np.sqrt(var_y1), size=n_samples)

    source = np.column_stack((x1, x2))
    target = np.column_stack((y1, shared_gain * x1))
    variances = {"x1": var_x1, "x2": var_x2, "y1": var_y1, "y2": var_shared}
    return source, target, variances
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
_fit_active_pca
Fit a complete PCA basis and remove only numerically zero-variance components.

INPUT:
    - samples: np.ndarray -> representation, shape (samples, features)
    - variance_rtol: float or None -> relative cutoff for active PCA eigenvalues

OUTPUT:
    - result: dict -> fitted PCA, active scores, and explained-variance ratios
"""
def _fit_active_pca(samples, variance_rtol=None):
    samples = np.asarray(samples, dtype=float)
    max_components = min(samples.shape)
    pca = PCA(n_components=max_components).fit(samples)

    explained_variance = np.asarray(pca.explained_variance_, dtype=float)
    if variance_rtol is None:
        variance_rtol = max(samples.shape) * np.finfo(float).eps
    # end if variance_rtol is None
    if variance_rtol < 0:
        raise ValueError("variance_rtol must be non-negative.")
    # end if variance_rtol < 0

    largest_variance = explained_variance.max(initial=0.0)
    if largest_variance == 0:
        raise ValueError("PCA is undefined for a representation with zero variance.")
    # end if largest_variance == 0

    active = explained_variance > variance_rtol * largest_variance
    scores = pca.transform(samples)[:, active]
    variance_ratio = np.asarray(pca.explained_variance_ratio_, dtype=float)[active]
    return {
        "pca": pca,
        "active": active,
        "scores": scores,
        "variance_ratio": variance_ratio,
        "cumulative_variance_ratio": np.cumsum(variance_ratio),
        "full_rank": max_components,
        "active_rank": int(np.sum(active)),
    }
# EOF


"""
compute_pc_coupling_matrix
Fit PCA separately to two representations and compute the non-negative matrix
of squared correlations between their active PC scores.

INPUT:
    - X: np.ndarray -> first centered-by-PCA representation, shape (samples, features_X)
    - Y: np.ndarray -> second centered-by-PCA representation, shape (samples, features_Y)
    - variance_rtol: float or None -> relative cutoff for zero-variance PCs

OUTPUT:
    - result: dict -> PCA fits, scores, variance spectra, and C with shape (PC_X, PC_Y)
"""
def compute_pc_coupling_matrix(X, Y, variance_rtol=None):
    X, Y = validate_sample_spaces(X, Y)
    pca_X = _fit_active_pca(X, variance_rtol)
    pca_Y = _fit_active_pca(Y, variance_rtol)

    # PCA centers each score. Recenter once to suppress floating-point offsets.
    Z_X = pca_X["scores"] - pca_X["scores"].mean(axis=0, keepdims=True)
    Z_Y = pca_Y["scores"] - pca_Y["scores"].mean(axis=0, keepdims=True)
    norm_X = np.sqrt(np.sum(Z_X**2, axis=0))
    norm_Y = np.sqrt(np.sum(Z_Y**2, axis=0))
    correlation = (Z_X.T @ Z_Y) / np.outer(norm_X, norm_Y)
    coupling = np.clip(correlation**2, 0.0, 1.0)

    return {
        "pca_X": pca_X["pca"],
        "pca_Y": pca_Y["pca"],
        "active_X": pca_X["active"],
        "active_Y": pca_Y["active"],
        "scores_X": Z_X,
        "scores_Y": Z_Y,
        "variance_ratio_X": pca_X["variance_ratio"],
        "variance_ratio_Y": pca_Y["variance_ratio"],
        "cumulative_variance_ratio_X": pca_X["cumulative_variance_ratio"],
        "cumulative_variance_ratio_Y": pca_Y["cumulative_variance_ratio"],
        "coupling": coupling,
        "fit_scope": "in-sample",
    }
# EOF


"""
compute_variance_asymmetry
Weight a PC-to-PC coupling matrix by signed and unsigned differences in the
variance importance and normalized rank of its paired axes.

INPUT:
    - coupling: np.ndarray -> non-negative PC coupling matrix, shape (PC_X, PC_Y)
    - variance_ratio_X: np.ndarray -> explained-variance ratios for X PCs
    - variance_ratio_Y: np.ndarray -> explained-variance ratios for Y PCs

OUTPUT:
    - result: dict -> contribution matrices and scalar asymmetry/mismatch metrics
"""
def compute_variance_asymmetry(coupling, variance_ratio_X, variance_ratio_Y):
    coupling = np.asarray(coupling, dtype=float)
    variance_ratio_X = np.asarray(variance_ratio_X, dtype=float)
    variance_ratio_Y = np.asarray(variance_ratio_Y, dtype=float)
    expected_shape = (variance_ratio_X.size, variance_ratio_Y.size)

    if coupling.shape != expected_shape:
        raise ValueError(
            f"coupling must have shape {expected_shape}, received {coupling.shape}."
        )
    # end if coupling shape is invalid
    if np.any(coupling < 0) or not np.isfinite(coupling).all():
        raise ValueError("coupling must contain finite, non-negative values.")
    # end if coupling values are invalid

    variance_difference = variance_ratio_X[:, np.newaxis] - variance_ratio_Y
    contribution = coupling * variance_difference
    coupling_sum = float(np.sum(coupling))
    if coupling_sum == 0:
        normalized_asymmetry = np.nan
        variance_mismatch = np.nan
        rank_asymmetry = np.nan
    else:
        normalized_asymmetry = float(np.sum(contribution) / coupling_sum)
        variance_mismatch = float(
            np.sum(coupling * np.abs(variance_difference)) / coupling_sum
        )

        rank_X = np.zeros(variance_ratio_X.size)
        rank_Y = np.zeros(variance_ratio_Y.size)
        if variance_ratio_X.size > 1:
            rank_X = np.arange(variance_ratio_X.size) / (variance_ratio_X.size - 1)
        # end if X has more than one PC
        if variance_ratio_Y.size > 1:
            rank_Y = np.arange(variance_ratio_Y.size) / (variance_ratio_Y.size - 1)
        # end if Y has more than one PC
        rank_difference = rank_Y[np.newaxis, :] - rank_X[:, np.newaxis]
        rank_asymmetry = float(np.sum(coupling * rank_difference) / coupling_sum)
    # end if coupling_sum == 0

    return {
        "contribution": contribution,
        "A_var": float(np.sum(contribution)),
        "A_var_norm": normalized_asymmetry,
        "M_var": variance_mismatch,
        "A_rank": rank_asymmetry,
        "coupling_sum": coupling_sum,
    }
# EOF


"""
decompose_pooled_r2_in_pc_basis
Fit reciprocal multivariate OLS models in a PCA-score basis and decompose each
pooled R-squared into target-PC R-squared values weighted by target variance.

INPUT:
    - scores_X: np.ndarray -> centered X PC scores, shape (samples, PC_X)
    - scores_Y: np.ndarray -> centered Y PC scores, shape (samples, PC_Y)
    - variance_ratio_X: np.ndarray -> explained-variance ratios for X PCs
    - variance_ratio_Y: np.ndarray -> explained-variance ratios for Y PCs

OUTPUT:
    - result: dict -> component R-squared arrays and exact weighted decompositions
"""
def decompose_pooled_r2_in_pc_basis(
    scores_X,
    scores_Y,
    variance_ratio_X,
    variance_ratio_Y,
):
    scores_X, scores_Y = validate_sample_spaces(scores_X, scores_Y)
    variance_ratio_X = np.asarray(variance_ratio_X, dtype=float)
    variance_ratio_Y = np.asarray(variance_ratio_Y, dtype=float)
    if variance_ratio_X.shape != (scores_X.shape[1],):
        raise ValueError("variance_ratio_X must contain one value per X PC.")
    # end if variance_ratio_X shape is invalid
    if variance_ratio_Y.shape != (scores_Y.shape[1],):
        raise ValueError("variance_ratio_Y must contain one value per Y PC.")
    # end if variance_ratio_Y shape is invalid

    # Include an intercept so this exactly matches ordinary regression conventions.
    design_X = np.column_stack((np.ones(scores_X.shape[0]), scores_X))
    design_Y = np.column_stack((np.ones(scores_Y.shape[0]), scores_Y))
    coefficient_X_to_Y = np.linalg.lstsq(design_X, scores_Y, rcond=None)[0]
    coefficient_Y_to_X = np.linalg.lstsq(design_Y, scores_X, rcond=None)[0]
    predicted_Y = design_X @ coefficient_X_to_Y
    predicted_X = design_Y @ coefficient_Y_to_X

    total_X = np.sum((scores_X - scores_X.mean(axis=0)) ** 2, axis=0)
    total_Y = np.sum((scores_Y - scores_Y.mean(axis=0)) ** 2, axis=0)
    component_r2_X_from_Y = (
        1 - np.sum((scores_X - predicted_X) ** 2, axis=0) / total_X
    )
    component_r2_Y_from_X = (
        1 - np.sum((scores_Y - predicted_Y) ** 2, axis=0) / total_Y
    )
    weighted_component_X_from_Y = variance_ratio_X * component_r2_X_from_Y
    weighted_component_Y_from_X = variance_ratio_Y * component_r2_Y_from_X
    pooled_r2_Y_to_X = float(np.sum(weighted_component_X_from_Y))
    pooled_r2_X_to_Y = float(np.sum(weighted_component_Y_from_X))

    return {
        "component_r2_X_from_Y": component_r2_X_from_Y,
        "component_r2_Y_from_X": component_r2_Y_from_X,
        "weighted_component_X_from_Y": weighted_component_X_from_Y,
        "weighted_component_Y_from_X": weighted_component_Y_from_X,
        "pooled_r2_Y_to_X": pooled_r2_Y_to_X,
        "pooled_r2_X_to_Y": pooled_r2_X_to_Y,
        "delta_r2": pooled_r2_Y_to_X - pooled_r2_X_to_Y,
        "predicted_X": predicted_X,
        "predicted_Y": predicted_Y,
    }
# EOF


"""
plot_pc_coupling
Visualize PC coupling, variance-weighted contributions, cumulative-variance
coordinates, marginal spectra, and strongly coupled variance pairs.

INPUT:
    - pc_result: dict -> output of compute_pc_coupling_matrix
    - asymmetry_result: dict -> output of compute_variance_asymmetry
    - X_name: str -> display name for the first representation
    - Y_name: str -> display name for the second representation
    - coupling_quantile: float -> positive-coupling quantile shown in the scatter

OUTPUT:
    - figures: dict -> named matplotlib figures for the four requested views
"""
def plot_pc_coupling(
    pc_result,
    asymmetry_result,
    X_name="X",
    Y_name="Y",
    coupling_quantile=0.95,
):
    import matplotlib.pyplot as plt

    if not 0 <= coupling_quantile <= 1:
        raise ValueError("coupling_quantile must lie in [0, 1].")
    # end if coupling_quantile is invalid

    coupling = pc_result["coupling"]
    variance_X = pc_result["variance_ratio_X"]
    variance_Y = pc_result["variance_ratio_Y"]
    contribution = asymmetry_result["contribution"]
    n_X, n_Y = coupling.shape

    # Couple the heatmap to both variance spectra using shared PC-index axes.
    figure_coupling = plt.figure(figsize=(10, 8))
    grid = figure_coupling.add_gridspec(
        2, 2, width_ratios=(1.4, 6), height_ratios=(1.4, 6),
        hspace=0.05, wspace=0.05,
    )
    axis_Y_spectrum = figure_coupling.add_subplot(grid[0, 1])
    axis_X_spectrum = figure_coupling.add_subplot(grid[1, 0])
    axis_coupling = figure_coupling.add_subplot(grid[1, 1])
    image = axis_coupling.imshow(coupling, aspect="auto", cmap="magma")
    axis_coupling.set(
        xlabel=f"{Y_name} PC (high to low variance)",
        ylabel=f"{X_name} PC (high to low variance)",
        title=f"PC coupling $C_{{ij}}=r^2$ ({pc_result['fit_scope']})",
    )
    figure_coupling.colorbar(image, ax=axis_coupling, label=r"$r^2$")
    axis_Y_spectrum.bar(np.arange(n_Y), variance_Y, color="tab:orange")
    axis_Y_spectrum.set(ylabel="variance", xlim=(-0.5, n_Y - 0.5))
    axis_Y_spectrum.tick_params(labelbottom=False)
    axis_X_spectrum.barh(np.arange(n_X), variance_X, color="tab:blue")
    axis_X_spectrum.set(xlabel="variance", ylim=(n_X - 0.5, -0.5))
    axis_X_spectrum.tick_params(labelleft=False)

    # Signed contributions use a centered diverging color scale.
    figure_asymmetry, axis_asymmetry = plt.subplots(figsize=(9, 7))
    contribution_limit = np.max(np.abs(contribution), initial=0.0)
    if contribution_limit == 0:
        contribution_limit = 1.0
    # end if all contributions are zero
    image = axis_asymmetry.imshow(
        contribution,
        aspect="auto",
        cmap="coolwarm",
        vmin=-contribution_limit,
        vmax=contribution_limit,
    )
    axis_asymmetry.set(
        xlabel=f"{Y_name} PC (high to low variance)",
        ylabel=f"{X_name} PC (high to low variance)",
        title=r"Signed contribution $C_{ij}(v_i^X-v_j^Y)$",
    )
    figure_asymmetry.colorbar(image, ax=axis_asymmetry, label="contribution")
    figure_asymmetry.tight_layout()

    # Variable-width cells locate coupling in cumulative explained-variance space.
    cumulative_edges_X = 100 * np.r_[0.0, np.cumsum(variance_X)]
    cumulative_edges_Y = 100 * np.r_[0.0, np.cumsum(variance_Y)]
    figure_cumulative, axis_cumulative = plt.subplots(figsize=(9, 7))
    mesh = axis_cumulative.pcolormesh(
        cumulative_edges_Y,
        cumulative_edges_X,
        coupling,
        shading="flat",
        cmap="magma",
    )
    axis_cumulative.invert_yaxis()
    axis_cumulative.set(
        xlabel=f"{Y_name} cumulative explained variance (%)",
        ylabel=f"{X_name} cumulative explained variance (%)",
        title=f"PC coupling in variance-percentile coordinates ({pc_result['fit_scope']})",
    )
    figure_cumulative.colorbar(mesh, ax=axis_cumulative, label=r"$r^2$")
    figure_cumulative.tight_layout()

    # Show only the strongest positive couplings to keep dense matrices readable.
    positive_coupling = coupling[coupling > 0]
    threshold = (
        np.quantile(positive_coupling, coupling_quantile)
        if positive_coupling.size
        else np.inf
    )
    strong_X, strong_Y = np.where(coupling >= threshold)
    strong_coupling = coupling[strong_X, strong_Y]
    figure_scatter, axis_scatter = plt.subplots(figsize=(7, 7))
    if strong_coupling.size:
        sizes = 25 + 275 * strong_coupling / strong_coupling.max()
        scatter = axis_scatter.scatter(
            variance_X[strong_X],
            variance_Y[strong_Y],
            s=sizes,
            c=strong_coupling,
            cmap="viridis",
            alpha=0.75,
            edgecolors="none",
        )
        figure_scatter.colorbar(scatter, ax=axis_scatter, label=r"$C_{ij}=r^2$")
    # end if strong couplings exist
    diagonal_limit = max(variance_X.max(), variance_Y.max())
    axis_scatter.plot([0, diagonal_limit], [0, diagonal_limit], "--", color="0.4")
    axis_scatter.set(
        xlabel=fr"{X_name} explained-variance ratio $v_i^X$",
        ylabel=fr"{Y_name} explained-variance ratio $v_j^Y$",
        title=f"Strongest PC pairs (top {100 * (1 - coupling_quantile):.0f}%)",
        xlim=(0, diagonal_limit * 1.03),
        ylim=(0, diagonal_limit * 1.03),
    )
    axis_scatter.set_aspect("equal", adjustable="box")
    figure_scatter.tight_layout()

    return {
        "coupling": figure_coupling,
        "asymmetry": figure_asymmetry,
        "cumulative": figure_cumulative,
        "scatter": figure_scatter,
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


"""
squared_distance_matrix_from_rdm
Turn a condensed RDM into the full squared-distance matrix that classical MDS
expects, using RDM_IS_SQUARED_DISTANCE to know whether the measure already
returns squared distances.

INPUT:
    - rdm_vector: np.ndarray -> condensed RDM, shape (samples * (samples - 1) / 2,)
    - metric: str -> measure used by create_RDM to build the RDM

OUTPUT:
    - squared_distances: np.ndarray -> full squared-distance matrix, shape (samples, samples)
"""
def squared_distance_matrix_from_rdm(rdm_vector, metric):
    if metric not in RDM_IS_SQUARED_DISTANCE:
        raise KeyError(
            f"Unknown measure '{metric}'. Add it to RDM_IS_SQUARED_DISTANCE "
            "stating whether its RDM holds squared distances."
        )
    # end if metric is unknown

    rdm = squareform(np.asarray(rdm_vector, dtype=float))
    if RDM_IS_SQUARED_DISTANCE[metric]:
        # 1 - cos(u, v) = 0.5 * ||u_hat - v_hat||^2, so twice the RDM is the squared distance.
        squared_distances = 2.0 * rdm
    else:
        squared_distances = rdm**2
    # end if the RDM already holds squared distances
    return squared_distances
# EOF


"""
components_for_variance
Count how many leading components are needed to reach a fraction of the total
embedded variance.

INPUT:
    - cumulative_variance: np.ndarray -> cumulative share of the positive eigenvalues
    - variance_explained: float -> fraction to reach, in (0, 1]

OUTPUT:
    - n_components: int -> smallest number of components reaching the fraction
"""
def components_for_variance(cumulative_variance, variance_explained):
    if not 0 < variance_explained <= 1:
        raise ValueError("variance_explained must lie in (0, 1].")
    # end if variance_explained is invalid
    n_components = int(np.searchsorted(cumulative_variance, variance_explained) + 1)
    return min(n_components, len(cumulative_variance))
# EOF


"""
participation_ratio
Effective dimensionality of a spectrum: 1 when a single eigenvalue dominates and
n when all n eigenvalues are equal. It reads the shape of the spectrum and is
insensitive to its overall scale.

INPUT:
    - eigenvalues: np.ndarray -> non-negative eigenvalues

OUTPUT:
    - ratio: float -> participation ratio
"""
def participation_ratio(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    squared_sum = np.sum(eigenvalues**2)
    if squared_sum == 0:
        raise ValueError("Participation ratio is undefined for a zero spectrum.")
    # end if squared_sum == 0
    return float(np.sum(eigenvalues) ** 2 / squared_sum)
# EOF


"""
classical_mds_embedding
Embed an RDM in Euclidean coordinates with classical MDS and keep the leading
components that reach a target fraction of the embedded variance.

INPUT:
    - rdm_vector: np.ndarray -> condensed RDM, shape (samples * (samples - 1) / 2,)
    - metric: str -> measure used by create_RDM to build the RDM
    - variance_explained: float -> fraction of positive-eigenvalue mass to retain
    - relative_tolerance: float -> eigenvalue cutoff relative to the largest eigenvalue

OUTPUT:
    - result: dict -> retained coordinates, full-rank coordinates, eigenvalues, and
      the diagnostics needed to judge how faithful the embedding is
"""
def classical_mds_embedding(
    rdm_vector,
    metric,
    variance_explained=0.99,
    relative_tolerance=1e-10,
):
    if not 0 < variance_explained <= 1:
        raise ValueError("variance_explained must lie in (0, 1].")
    # end if variance_explained is invalid

    squared_distances = squared_distance_matrix_from_rdm(rdm_vector, metric)
    n_samples = squared_distances.shape[0]

    # Double centering turns squared distances into the Gram matrix of centered points.
    centering = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples
    gram = -0.5 * centering @ squared_distances @ centering
    gram = (gram + gram.T) / 2  # symmetrize away the round-off asymmetry

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Only positive eigenvalues carry real coordinates; the negative mass measures
    # how far the measure is from being exactly Euclidean-embeddable.
    active = eigenvalues > relative_tolerance * eigenvalues.max()
    positive_mass = eigenvalues[active].sum()
    negative_mass = float(-eigenvalues[eigenvalues < 0].sum() / positive_mass)

    coordinates = eigenvectors[:, active] * np.sqrt(eigenvalues[active])
    eigenvalue_shares = eigenvalues[active] / positive_mass
    cumulative_variance = np.cumsum(eigenvalue_shares)
    n_components = components_for_variance(cumulative_variance, variance_explained)

    return {
        "coordinates": coordinates[:, :n_components],
        "full_coordinates": coordinates,
        "eigenvalues": eigenvalues,
        "eigenvalue_shares": eigenvalue_shares,
        "cumulative_variance": cumulative_variance,
        # Sum of the positive eigenvalues: the cloud's total dispersion in this
        # measure's own units, so it is comparable only between measures that
        # share units. Divided by the sample count it is the mean squared
        # distance to the centroid.
        "total_variance": float(positive_mass),
        "participation_ratio": participation_ratio(eigenvalues[active]),
        "n_components": n_components,
        "full_rank": int(active.sum()),
        "variance_kept": float(cumulative_variance[n_components - 1]),
        "negative_eigenvalue_mass": negative_mass,
    }
# EOF


"""
embed_measures_with_mds
Compute one RDM per distance measure on the same data and embed each of them
with classical MDS.

INPUT:
    - data: np.ndarray -> responses with features in rows and samples in columns, shape (features, samples)
    - metrics: sequence of str -> distance measures passed to create_RDM
    - variance_explained: float -> fraction of embedded variance kept per measure

OUTPUT:
    - embeddings: dict -> measure name to the classical_mds_embedding result, with
      the condensed RDM stored under "rdm"
"""
def embed_measures_with_mds(data, metrics, variance_explained=0.99):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"data must be two-dimensional (features, samples), received {data.shape}."
        )
    # end if data.ndim != 2

    embeddings = {}
    for metric in metrics:
        rdm = create_RDM(data, metric)
        embedding = classical_mds_embedding(rdm, metric, variance_explained)
        embedding["rdm"] = rdm
        embeddings[metric] = embedding
    # end for metric in metrics
    return embeddings
# EOF


"""
metric_asymmetry_table
Run the asymmetry pipeline on every ordered pair of MDS-embedded measures and
collect the linear and rank-based asymmetry side by side.

Each row reports, for one direction A -> B: the pooled OLS R-squared of A
predicting B (the linear part of the relation), 1 - II between the embeddings
(the full, possibly nonlinear part), and 1 - II after both spaces are pushed
through their reciprocal Sigma^-1 transform, which removes the linear
anisotropy and leaves what OLS cannot account for.

INPUT:
    - embeddings: dict -> output of embed_measures_with_mds
    - k: int -> number of nearest neighbors used by II
    - metric_labels: dict or None -> short display names for the measures
    - relative_tolerance: float or None -> numerical-rank cutoff for the two SVDs

OUTPUT:
    - table: pd.DataFrame -> one row per ordered pair of measures
"""
def metric_asymmetry_table(
    embeddings,
    k=1,
    metric_labels=None,
    relative_tolerance=None,
):
    metrics = list(embeddings)
    metric_labels = {} if metric_labels is None else metric_labels
    rows = []

    for metric_A, metric_B in itertools.combinations(metrics, 2):
        embedding_A = embeddings[metric_A]
        embedding_B = embeddings[metric_B]
        # The MDS coordinates are Euclidean by construction, so II runs on plain
        # Euclidean distances in both spaces regardless of the original measure.
        pipeline = run_asymmetry_pipeline(
            embedding_A["coordinates"],
            embedding_B["coordinates"],
            source_metric="euclidean",
            target_metric="euclidean",
            k=k,
            relative_tolerance=relative_tolerance,
        )
        # Reference II straight from the original RDMs: it should match the
        # embedded II whenever the retained components describe the RDM well.
        rdm_ii = InformationImbalance("euclidean", "euclidean", k=k)
        rdm_ii.set_RDM(embedding_A["rdm"], "signal")
        rdm_ii.set_RDM(embedding_B["rdm"], "model")
        rdm_ii.compute_both_distance_ranks()
        rdm_A2B, rdm_B2A = rdm_ii.compute_both_II()

        for direction, source_metric, target_metric, r2, ii, ii_svd, ii_rdm in (
            (
                "A2B",
                metric_A,
                metric_B,
                pipeline["original_fit"]["pooled_r2_source_to_target"],
                pipeline["original_ii"]["source_to_target"],
                pipeline["transformed_ii"]["source_to_target"],
                rdm_A2B,
            ),
            (
                "B2A",
                metric_B,
                metric_A,
                pipeline["original_fit"]["pooled_r2_target_to_source"],
                pipeline["original_ii"]["target_to_source"],
                pipeline["transformed_ii"]["target_to_source"],
                rdm_B2A,
            ),
        ):
            rows.append(
                {
                    "direction": (
                        f"{metric_labels.get(source_metric, source_metric)} -> "
                        f"{metric_labels.get(target_metric, target_metric)}"
                    ),
                    "dim source": embeddings[source_metric]["n_components"],
                    "dim target": embeddings[target_metric]["n_components"],
                    "R2": r2,
                    "1 - II": 1.0 - ii,
                    "1 - II (Sigma^-1)": 1.0 - ii_svd,
                    "1 - II (raw RDM)": 1.0 - float(ii_rdm),
                }
            )
        # end for direction, source_metric, target_metric, r2, ii, ii_svd, ii_rdm
    # end for metric_A, metric_B
    return pd.DataFrame(rows)
# EOF


"""
mds_dimensionality_table
Summarize how many MDS components each measure needs and how Euclidean it is.

INPUT:
    - embeddings: dict -> output of embed_measures_with_mds
    - metric_labels: dict or None -> short display names for the measures
    - quantiles: sequence of float -> variance fractions to report a dimension for

OUTPUT:
    - table: pd.DataFrame -> one row per measure

NOTES:
    The full rank is capped by the number of recorded channels and saturates it
    for any measure that does not project a dimension away, so it separates the
    measures poorly. PR (participation ratio) and the low quantiles read the
    shape of the spectrum instead and are the informative columns here.

    PR is scale-free, so "total var" and "top eig." carry the scale it discards.
    They live in each measure's own units: the cosine family is dimensionless and
    bounded (1 - cos lies in [0, 2], so total var / n_samples cannot exceed 1),
    while euclidean and magnitude_diff are in squared response units and can
    therefore be compared with each other directly.
"""
def mds_dimensionality_table(embeddings, metric_labels=None, quantiles=(0.5, 0.9)):
    metric_labels = {} if metric_labels is None else metric_labels
    rows = []
    for metric, embedding in embeddings.items():
        row = {
            "measure": metric_labels.get(metric, metric),
            "PR": embedding["participation_ratio"],
        }
        for quantile in quantiles:
            row[f"d{round(100 * quantile)}"] = components_for_variance(
                embedding["cumulative_variance"], quantile
            )
        # end for quantile in quantiles
        # PR reads only the shape of the spectrum, so carry the scale alongside it.
        row["total var"] = embedding["total_variance"]
        row["top eig."] = float(embedding["eigenvalues"][0])
        row["top eig. share"] = float(embedding["eigenvalue_shares"][0])
        row["dim (99% var)"] = embedding["n_components"]
        row["full rank"] = embedding["full_rank"]
        row["negative eig. mass"] = embedding["negative_eigenvalue_mass"]
        rows.append(row)
    # end for metric, embedding in embeddings.items()
    return pd.DataFrame(rows)
# EOF
