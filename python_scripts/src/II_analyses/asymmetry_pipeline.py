import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

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
