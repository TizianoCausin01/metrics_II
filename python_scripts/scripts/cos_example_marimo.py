import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Cosine synthetic example: linear encoding and Information Imbalance

    This notebook treats $x$ as the **model representation** and $y$ as the
    **neural representation**. Roles are now swapped relative to the earlier
    version: the nonlinearly informative variable lives in $x$ (one feature),
    and the nuisance dimension lives in $y$.

    1. generates a 1-D latent $x$ and the 2-D vector
       $y=[\cos(x),\ \varepsilon\cdot s]$ with $\varepsilon$ independent
       Gaussian noise;
    2. fits the static projections $y\rightarrow\hat{x}$ and
       $x\rightarrow\hat{y}$;
    3. computes $R^2$ for both target representations;
    4. computes Information Imbalance between $\hat{x}$ and $\hat{y}$; and
    5. repeats the reciprocal OLS and II analysis after replacing the non-zero
       singular values of each fitted map with one.

    The point of this configuration: linear encoding favours $y\rightarrow x$
    (the 1-D target has nothing unpredictable), while the singular-value
    normalized Information Imbalance favours $x\rightarrow y$ ($x$ determines
    $\cos(x)$, but the cosine fold means $\cos(x)$ does not determine $x$).

    By default the projections are in-sample. Set `use_cv=True` to assemble
    out-of-fold predictions with the configured cross-validation scheme.
    """)
    return


@app.cell
def _():
    import os
    import sys
    from dataclasses import dataclass
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import yaml

    # Locate the repository whether marimo starts in the project root or scripts folder.
    cwd = Path.cwd().resolve()
    candidate_roots = [cwd, *cwd.parents]
    PROJECT_ROOT = next(
        (path for path in candidate_roots if (path / "config.yaml").is_file()),
        None,
    )
    if PROJECT_ROOT is None:
        raise FileNotFoundError("Could not locate config.yaml from the current directory.")
    # end if PROJECT_ROOT is None

    ENV = os.getenv("MY_ENV", "tiziano_mac_mini")
    with open(PROJECT_ROOT / "config.yaml", "r") as f:
        project_config = yaml.safe_load(f)
    # end with open config.yaml
    paths = project_config[ENV]["paths"]
    sys.path.append(paths["src_path"])
    sys.path.append(paths["useful_stuff_path"])

    from II_analyses.asymmetry_pipeline import (
        run_asymmetry_pipeline,
        summarize_asymmetry_pipeline,
    )
    from II_analyses.static_encoding import (
        compute_static_prediction_II,
        fit_static_projection,
    )
    from useful_stuff.general_utils.regression import linear_encoding
    from useful_stuff.general_utils.utils import get_triu_perms
    from useful_stuff.general_utils.II import InformationImbalance

    return (
        InformationImbalance,
        compute_static_prediction_II,
        dataclass,
        fit_static_projection,
        get_triu_perms,
        linear_encoding,
        np,
        pd,
        plt,
        run_asymmetry_pipeline,
        summarize_asymmetry_pipeline,
    )


@app.cell
def _(dataclass):
    @dataclass
    class Cfg:
        # Synthetic relation.
        n_samples: int = 1000
        x_min: float = 0  # -.5
        # ~1.27*pi: cos(x) keeps a net linear trend (so linear R2 stays
        # directional) but folds past x=pi (so II becomes directional).
        x_max: float = 4.0
        noise_std: float = 0  # latent noise on x; kept at 0
        y_noise_scale: float = 1.0  # std of the independent noise coordinate in y
        random_seed: int = 0

        # Bidirectional static regression.
        regression_type: str = "lr"
        use_cv: bool = False
        cv_type: str = "kf"
        n_splits: int = 5
        shuffle: bool = True
        singular_value_rtol: float | None = None

        # Information Imbalance. Euclidean distance is appropriate for 1D spaces.
        x_RDM_metric: str = "euclidean"
        y_RDM_metric: str = "euclidean"
        k: int = 1

    # EOC

    cfg = Cfg()
    cfg
    return (cfg,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Generate $x$ and $y$

    The arrays follow the project convention `(features, samples)`. Here $x$ has
    **one** feature (the latent), $y$ has **two** features
    ($\cos(\text{latent})$ plus an independent Gaussian-noise coordinate), and
    both share the same observations.
    """)
    return


@app.cell
def _(cfg, np):
    """
    y_func
    Build the 2-D y representation from the 1-D latent x.

    INPUT:
        - x: np.ndarray -> one-dimensional latent observations
        - rng: np.random.Generator -> seeded random-number generator
        - noise_std: float -> unused here (kept for signature compatibility)
        - noise_scale: float -> standard deviation of the independent y noise axis

    OUTPUT:
        - y: np.ndarray -> shape (2, samples): [cos(x), independent Gaussian noise]
    """

    def y_func(x, rng, noise_std, noise_scale):
        y_cos = np.cos(x)  # nonlinear, non-monotone image of the latent
        y_noise = rng.normal(0, 1, size=len(x)) * noise_scale  # independent nuisance axis
        return np.vstack((y_cos, y_noise))

    # EOF

    rng = np.random.default_rng(cfg.random_seed)
    latent_u = rng.uniform(cfg.x_min, cfg.x_max, size=cfg.n_samples)
    x_values = latent_u + rng.normal(0, cfg.noise_std, size=cfg.n_samples)
    y_values = y_func(
        x_values,
        rng=rng,
        noise_std=cfg.noise_std,
        noise_scale=cfg.y_noise_scale,
    )

    # x is the single nonlinearly-informative latent; y carries its cosine
    # image plus one independent noise coordinate.
    x = x_values[np.newaxis, :]
    y = y_values

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    return latent_u, x, y


@app.cell
def _(plt, y):
    # The two y coordinates: the cosine signal against the independent noise axis.
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(y[0], y[1], s=8, alpha=0.5)
    _ax.set_xlabel(r"$y_1 = \cos(x)$")
    _ax.set_ylabel(r"$y_2$ = independent noise")
    _ax.set_title("The two y coordinates")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(plt, x, y):
    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(x[0], y[0], s=8, alpha=0.5)
    _ax.set_xlabel("x (latent, model representation)")
    _ax.set_ylabel(r"$y_1 = \cos(x)$")
    _ax.set_title("Nonlinear latent -> cosine relation")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fit the two static linear projections

    `predicted_x` is the $y\rightarrow x$ prediction and `predicted_y` is the
    $x\rightarrow y$ prediction. When CV is enabled, both directions use
    identical fold assignments and all reported predictions are out of fold.
    """)
    return


@app.cell
def _(cfg, fit_static_projection, linear_encoding, np, x, y):
    if cfg.regression_type not in {"lr", "ridge", "lasso", "en"}:
        raise ValueError(f"Unsupported regression_type: {cfg.regression_type}")
    # end if cfg.regression_type not in supported types
    if cfg.use_cv and cfg.cv_type not in {"loo", "kf"}:
        raise ValueError("Use 'loo' or 'kf' when use_cv=True.")
    # end if cfg.use_cv and cfg.cv_type not in supported types

    active_cv_type = cfg.cv_type if cfg.use_cv else "same"
    y_to_x_encoding = linear_encoding(
        regression_type=cfg.regression_type,
        cv_type=active_cv_type,
        score_type="r2",
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
    )
    x_to_y_encoding = linear_encoding(
        regression_type=cfg.regression_type,
        cv_type=active_cv_type,
        score_type="r2",
        n_splits=cfg.n_splits,
        shuffle=cfg.shuffle,
    )

    # Reset the seed before each direction so CV uses the same folds.
    np.random.seed(cfg.random_seed)
    predicted_x, x_r2 = fit_static_projection(
        y,
        x,
        y_to_x_encoding,
        use_cv=cfg.use_cv,
    )

    np.random.seed(cfg.random_seed)
    predicted_y, y_r2 = fit_static_projection(
        x,
        y,
        x_to_y_encoding,
        use_cv=cfg.use_cv,
    )
    return (
        predicted_x,
        predicted_y,
        x_r2,
        x_to_y_encoding,
        y_r2,
        y_to_x_encoding,
    )


@app.cell
def _(
    cfg,
    np,
    plt,
    predicted_x,
    predicted_y,
    x,
    x_r2,
    x_to_y_encoding,
    y,
    y_r2,
    y_to_x_encoding,
):
    encoding_mode = "out-of-fold" if cfg.use_cv else "in-sample"
    print(f"Encoding mode: {encoding_mode}")
    print(f"y -> x: R²={np.nanmean(x_r2):.4f}, prediction shape={predicted_x.shape}")
    print(f"x -> y: R²={np.nanmean(y_r2):.4f}, prediction shape={predicted_y.shape}")
    print(f"y -> x weight shape: {y_to_x_encoding.get_weights().shape}")
    print(f"x -> y weight shape: {x_to_y_encoding.get_weights().shape}")

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    _axes[0].scatter(x[0], predicted_x[0], s=8, alpha=0.5)
    _axes[0].set_xlabel("observed x")
    _axes[0].set_ylabel("predicted x from y")
    _axes[0].set_title(f"y -> x, R²={np.nanmean(x_r2):.3f}")
    _axes[1].scatter(y[0], predicted_y[0], s=8, alpha=0.5)
    _axes[1].set_xlabel(r"observed $y_1 = \cos(x)$")
    _axes[1].set_ylabel(r"predicted $y_1$ from x")
    _axes[1].set_title(f"x -> y, pooled R²={np.nanmean(y_r2):.3f}")
    _fig.suptitle(f"Static linear projections ({encoding_mode})")
    _fig.tight_layout()
    _fig
    return (encoding_mode,)


@app.cell
def _(encoding_mode, np, plt, predicted_x, predicted_y, x, x_r2, y, y_r2):
    # Predictions plotted against the *latent* axis with the true target overlaid,
    # to expose what each direction can and cannot recover.
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # x -> y_1: linear fit (blue) is a near-straight line; truth cos(x) (orange)
    # is the curve. The gap is the residual that keeps pooled R2 modest.
    _axes[0].scatter(x[0], predicted_y[0], s=8, alpha=0.5, label="predicted $y_1$ from x")
    _axes[0].scatter(x[0], y[0], s=8, alpha=0.5, label="true $y_1=\\cos(x)$")
    _axes[0].axhline(np.mean(y[0]), color="0.6", lw=0.8)
    _axes[0].set_xlabel("observed x (latent)")
    _axes[0].set_ylabel(r"$y_1$")
    _axes[0].set_title(f"x -> y, pooled R²={np.nanmean(y_r2):.3f}")
    _axes[0].legend(fontsize=8)
    # y -> x: predicted x (blue) is a line in cos(x); true x (orange) folds back
    # on itself past x=pi, so cos(x) cannot pin down x.
    _axes[1].scatter(y[0], predicted_x[0], s=8, alpha=0.5, label="predicted x from y")
    _axes[1].scatter(y[0], x[0], s=8, alpha=0.5, label="true x")
    _axes[1].set_xlabel(r"observed $y_1 = \cos(x)$")
    _axes[1].set_ylabel("x (latent)")
    _axes[1].set_title(f"y -> x, R²={np.nanmean(x_r2):.3f}")
    _axes[1].legend(fontsize=8)
    _fig.suptitle(f"linear projections ({encoding_mode})")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(np, x, y):
    print(np.var(x))
    print(np.var(y))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Information Imbalance between the predicted spaces

    The helper labels its first input as the neural space and its second input
    as the model space. Here those inputs are `predicted_y` and `predicted_x`,
    so `A2B` is predicted $y\rightarrow x$ and `B2A` is predicted
    $x\rightarrow y$.
    """)
    return


@app.cell
def _(cfg, compute_static_prediction_II, predicted_x, predicted_y):
    _, predicted_y_to_x_II, predicted_x_to_y_II = compute_static_prediction_II(
        predicted_neural=predicted_y,
        predicted_model=predicted_x,
        neural_RDM_metric=cfg.y_RDM_metric,
        model_RDM_metric=cfg.x_RDM_metric,
        k=cfg.k,
    )

    print(f"predicted y -> predicted x: II={predicted_y_to_x_II:.6f}")
    print(f"predicted x -> predicted y: II={predicted_x_to_y_II:.6f}")
    return predicted_x_to_y_II, predicted_y_to_x_II


@app.cell
def _(cfg, encoding_mode, plt, predicted_x_to_y_II, predicted_y_to_x_II):
    _fig, _ax = plt.subplots(figsize=(5, 4))
    _directions = ["pred y -> pred x", "pred x -> pred y"]
    _ii_values = [predicted_y_to_x_II, predicted_x_to_y_II]
    _ax.bar(_directions, _ii_values, color=["tab:blue", "tab:orange"])
    _ax.set_ylabel("Information Imbalance")
    _ax.set_title(f"Predicted-space II, k={cfg.k} ({encoding_mode})")
    _ax.tick_params(axis="x", rotation=12)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(predicted_x, predicted_y, x, y):
    residuals_x = x - predicted_x
    residuals_y = y - predicted_y
    return residuals_x, residuals_y


@app.cell
def _(
    InformationImbalance,
    cfg,
    get_triu_perms,
    np,
    pd,
    predicted_x,
    predicted_y,
    residuals_x,
    residuals_y,
    x,
    y,
):
    _space_names = ["x", "y", "predicted_x", "predicted_y", "residuals_x", "residuals_y"]
    _all_spaces = [x, y, predicted_x, predicted_y, residuals_x, residuals_y]
    _space_metrics = [
        cfg.x_RDM_metric,
        cfg.y_RDM_metric,
        cfg.x_RDM_metric,
        cfg.y_RDM_metric,
        cfg.x_RDM_metric,
        cfg.y_RDM_metric,
    ]

    # Rows are source spaces and columns are target spaces.
    _ii_matrix = np.full((len(_all_spaces), len(_all_spaces)), np.nan)
    _space_index_pairs = get_triu_perms(list(range(len(_all_spaces))))

    for _idx_A, _idx_B in _space_index_pairs:
        _ii_obj = InformationImbalance(
            signal_RDM_metric=_space_metrics[_idx_A],
            model_RDM_metric=_space_metrics[_idx_B],
            k=cfg.k,
        )
        _ii_obj.compute_both_RDMs(_all_spaces[_idx_A], _all_spaces[_idx_B])
        _ii_obj.compute_both_distance_ranks()
        _A_to_B_II, _B_to_A_II = _ii_obj.compute_both_II()

        # A2B is row A -> column B; B2A occupies the transposed position.
        _ii_matrix[_idx_A, _idx_B] = _A_to_B_II
        _ii_matrix[_idx_B, _idx_A] = _B_to_A_II
    # end for _idx_A, _idx_B in _space_index_pairs

    ii_table = pd.DataFrame(_ii_matrix, index=_space_names, columns=_space_names)
    ii_table.index.name = "source"
    ii_table.columns.name = "target"

    # Each cell contains II(row source -> column target); self-pairs are blank.
    ii_table.style.format(precision=3, na_rep="--").background_gradient(
        cmap="viridis",
        axis=None,
        vmin=0,
        vmax=1,
    ).set_caption("Directional Information Imbalance")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## SVD-normalize the reciprocal OLS maps

    This repeats the full-data OLS analysis used in `gaussian_ols_svd_asymmetry`.
    For each fitted coefficient matrix $B=U\Sigma V^\top$ (outputs by inputs),
    the non-zero singular values are replaced with one:

    $$B_{01}=U\Sigma_{01}V^\top, \qquad W_{01}=B_{01}^\top.$$

    The row-wise transforms are $\widetilde{x}=x^\top W_{01}^{x\to y}$ and
    $\widetilde{y}=y^\top W_{01}^{y\to x}$. Numerical null directions remain
    zero, so the fitted rank is preserved while axis-specific regression gains
    are removed. Because $x$ has one feature and $y$ has two, $\widetilde{x}$
    has two collinear features and $\widetilde{y}$ has one feature. The
    $y\to x$ map puts ~zero weight on the noise coordinate, so $\widetilde{y}$
    is essentially $\cos(x)$ with the nuisance axis discarded.
    """)
    return


@app.cell
def _(cfg, np, run_asymmetry_pipeline, summarize_asymmetry_pipeline, x, y):
    # The shared pipeline expects samples in rows, unlike this notebook's static spaces.
    svd_pipeline = run_asymmetry_pipeline(
        source=x.T,
        target=y.T,
        source_metric=cfg.x_RDM_metric,
        target_metric=cfg.y_RDM_metric,
        k=cfg.k,
        relative_tolerance=cfg.singular_value_rtol,
    )
    svd_summary = summarize_asymmetry_pipeline(svd_pipeline, "x", "y")

    # Recover the matrices displayed in the Gaussian example.
    _svd_directions = (
        (
            "x -> y",
            svd_pipeline["original_fit"]["source_to_target"],
            svd_pipeline["source_svd"],
        ),
        (
            "y -> x",
            svd_pipeline["original_fit"]["target_to_source"],
            svd_pipeline["target_svd"],
        ),
    )

    for _direction, _encoding, _svd_result in _svd_directions:
        _coefficient = np.atleast_2d(np.asarray(_encoding.get_weights(), dtype=float))
        _sigma_01 = np.diag(_svd_result["active"].astype(float))
        _row_weights = _coefficient.T
        _normalized_row_weights = _svd_result["row_transform"]

        print(f"{_direction}: rank={_svd_result['rank']}")
        print("singular values:", np.round(_svd_result["singular_values"], 6))
        print("Sigma_01:\n", _sigma_01)
        print(f"W, shape {_row_weights.shape}:\n", np.round(_row_weights, 6))
        print(
            f"W_01, shape {_normalized_row_weights.shape}:\n",
            np.round(_normalized_row_weights, 6),
        )
        print()
    # end for _direction, _encoding, _svd_result

    x_tilde = svd_pipeline["transformed_source"].T
    y_tilde = svd_pipeline["transformed_target"].T
    print(f"x_tilde shape: {x_tilde.shape}; y_tilde shape: {y_tilde.shape}")

    # The rectangular transforms exchange the output dimensionalities of the maps.
    assert x_tilde.shape == (y.shape[0], cfg.n_samples)
    assert y_tilde.shape == (x.shape[0], cfg.n_samples)
    assert svd_pipeline["source_svd"]["rank"] == np.linalg.matrix_rank(
        svd_pipeline["source_svd"]["row_transform"]
    )
    assert svd_pipeline["target_svd"]["rank"] == np.linalg.matrix_rank(
        svd_pipeline["target_svd"]["row_transform"]
    )
    return svd_summary, x_tilde, y_tilde


@app.cell
def _(svd_summary):
    svd_summary.style.format({"pooled R2": "{:.6f}", "II": "{:.6f}"})
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Geometry after singular-value binarization

    Matched observations keep the same color in every panel. The left panel is
    the original 2-D $y$ space (cosine signal vs noise axis). The middle panel
    shows the two transformed representations against one another. The right
    panel makes the rank-one geometry of the now two-dimensional $\widetilde{x}$
    explicit.
    """)
    return


@app.cell
def _(latent_u, plt, x_tilde, y, y_tilde):
    point_color = latent_u
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4.2))

    _axes[0].scatter(y[0], y[1], c=point_color, cmap="twilight", s=9, alpha=0.55)
    _axes[0].set(xlabel=r"$y_1=\cos(x)$", ylabel=r"$y_2$ (noise)", title=r"original $y$ space")

    _axes[1].scatter(
        x_tilde[0], y_tilde[0], c=point_color, cmap="twilight", s=9, alpha=0.55
    )
    _axes[1].set(
        xlabel=r"$\widetilde{x}_1$",
        ylabel=r"$\widetilde{y}_1$",
        title="transformed matched observations",
    )

    _axes[2].scatter(
        x_tilde[0], x_tilde[1], c=point_color, cmap="twilight", s=9, alpha=0.55
    )
    _axes[2].set(
        xlabel=r"$\widetilde{x}_1$",
        ylabel=r"$\widetilde{x}_2$",
        title=r"rank-one $\widetilde{x}$ space",
    )

    for _ax in _axes:
        _ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        _ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    # end for _ax in _axes

    _fig.suptitle("Cosine example before and after SVD normalization")
    _fig.tight_layout()
    _fig
    return (point_color,)


@app.cell
def _(np, plt, svd_summary):
    # Compare directional asymmetry before and after the reciprocal transforms.
    # Both directions of the original spaces sit next to each other, then both
    # directions of the Sigma_01-normalized spaces sit next to each other.
    _original_metrics = svd_summary[svd_summary["stage"] == "original"]
    _normalized_metrics = svd_summary[svd_summary["stage"] == "Sigma^-1 normalized"]
    _bar_labels = [
        r"$x \to y$",
        r"$y \to x$",
        r"$\widetilde{x} \to \widetilde{y}$",
        r"$\widetilde{y} \to \widetilde{x}$",
    ]
    # A gap separates the original pair from the normalized pair.
    _bar_positions = np.array([0.0, 1.0, 2.4, 3.4])
    _bar_colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange"]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for _ax, _metric_name, _ylabel in (
        (_axes[0], "pooled R2", r"pooled $R^2$"),
        (_axes[1], "II", "1- Information Imbalance"),
    ):
        # svd_summary rows are ordered: original x->y, original y->x,
        # normalized x->y, normalized y->x, which matches _bar_positions.
        if _metric_name == "II":
            _values = np.concatenate(
                [1 - _original_metrics[_metric_name].to_numpy(),
                 1 - _normalized_metrics[_metric_name].to_numpy()]
            )
        else:
            _values = np.concatenate(
                [_original_metrics[_metric_name].to_numpy(),
                 _normalized_metrics[_metric_name].to_numpy()]
            )
        _ax.bar(_bar_positions, _values, width=0.8, color=_bar_colors)
        _ax.set_xticks(_bar_positions, _bar_labels)
        _ax.set_ylabel(_ylabel)
        _ax.set_title(f"{_ylabel}")
    # end for _ax, _metric_name, _ylabel

    _handles = [
        plt.Rectangle((0, 0), 1, 1, color="tab:blue"),
        plt.Rectangle((0, 0), 1, 1, color="tab:orange"),
    ]
    _axes[0].legend(_handles, ["original", r"$\Sigma_{01}$ normalized"])
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The two projected spaces

    $\widetilde{y}$ is one-dimensional, so it is drawn along a horizontal line.
    $\widetilde{x}$ has two coordinates but rank one, so its observations lie
    on a line in the two-dimensional output space. Colors identify matched
    samples.
    """)
    return


@app.cell
def _(cfg, np, plt, point_color, x_tilde, y_tilde):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # x_tilde has two coordinates whose rank-one structure appears as a line.
    _axes[0].scatter(
        x_tilde[0],
        x_tilde[1],
        c=point_color,
        cmap="twilight",
        s=10,
        alpha=0.55,
    )
    _axes[0].set(
        xlabel=r"$\widetilde{x}_1$",
        ylabel=r"$\widetilde{x}_2$",
        title=r"projected $\widetilde{x}$ space (rank 1)",
    )

    # y_tilde has one coordinate; zero on the vertical axis exposes its 1D geometry.
    _axes[1].scatter(
        y_tilde[0],
        np.zeros(cfg.n_samples),
        c=point_color,
        cmap="twilight",
        s=10,
        alpha=0.55,
    )
    _axes[1].set(
        xlabel=r"$\widetilde{y}_1$",
        ylabel="constant display axis",
        title=r"projected $\widetilde{y}$ space (rank 1)",
    )
    _axes[1].set_yticks([0])

    for _ax in _axes:
        _ax.axhline(0, color="0.85", linewidth=0.8, zorder=0)
        _ax.axvline(0, color="0.85", linewidth=0.8, zorder=0)
    # end for _ax in _axes

    _fig.suptitle(r"Spaces after the reciprocal $\Sigma_{01}$ projections")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
