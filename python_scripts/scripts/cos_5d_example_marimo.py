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
    # Five-dimensional cosine and linear-mixture example

    This notebook extends `cos_example_marimo.py` to a five-dimensional model
    representation with configurable uniform ranges for the two cosine inputs,

    $$
    x_1\sim\mathcal{U}(a_1,b_1),\qquad
    x_2\sim\mathcal{U}(a_2,b_2),\qquad
    x_3,x_4,x_5\sim\mathcal{N}(\mu,\sigma^2),
    $$

    and the seven-dimensional neural representation

    $$
    y=\left[
    \cos(x_1),\;
    \cos(x_2+\pi),\;
    A z+\epsilon_A,\;
    B z+\epsilon_B,\;
    C z+\epsilon_C,\;
    \eta_1,\;
    \eta_2
    \right],
    \qquad z=[x_3,x_4,x_5]^\top.
    $$

    Here $A$, $B$, and $C$ are configurable row vectors. The three
    $\epsilon$ terms are mild independent Gaussian noise, while $\eta_1$ and
    $\eta_2$ are independent nuisance dimensions. The notebook repeats the
    original bidirectional linear-encoding, Information Imbalance, and
    singular-value-normalized analyses.
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

    # marimo saves every figure at 2 x figure.dpi and displays it at
    # figsize x 100, so lowering the dpi shrinks the stored base64 PNG
    # without changing the on-screen size.
    plt.rcParams["figure.dpi"] = 72

    # Locate the repository whether marimo starts in the root or scripts folder.
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
    from useful_stuff.general_utils.II import InformationImbalance
    from useful_stuff.general_utils.regression import linear_encoding
    from useful_stuff.general_utils.utils import get_triu_perms

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
def _(dataclass, np):
    @dataclass
    class Cfg:
        # Synthetic data.
        n_samples: int = 1000
        x1_min: float = 0.0
        x1_max: float = 4.0
        x2_min: float = 0.0
        x2_max: float = 4.0
        gaussian_x_mean: float = 0.0
        gaussian_x_std: float = 1
        linear_noise_std: float = 0 #0.10
        nuisance_noise_std: float = 1.0
        second_cos_phase: float = np.pi
        random_seed: int = 0

        # A, B, and C are row vectors acting on z = [x3, x4, x5].
        A: tuple[float, float, float] = (1.00, 0.50, -0.25)
        B: tuple[float, float, float] = (-0.40, 1.00, 0.30)
        C: tuple[float, float, float] = (0.20, -0.35, 1.00)

        # Bidirectional static regression.
        regression_type: str = "lr"
        use_cv: bool = False
        cv_type: str = "kf"
        n_splits: int = 5
        shuffle: bool = True

        # None uses NumPy's numerical-rank convention. Set a positive relative
        # cutoff to discard weak fitted directions before SVD normalization.
        singular_value_rtol: float | None = None

        # Information Imbalance.
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

    Arrays follow the project convention `(features, samples)`. Set `x1_min`,
    `x1_max`, `x2_min`, and `x2_max` in `Cfg` to select the two portions of the
    cosine curves. The remaining coordinates $x_3,x_4,x_5$ stay Gaussian.
    """)
    return


@app.cell
def _(cfg, np):
    """
    build_neural_representation
    Construct the seven-dimensional y representation from five-dimensional x.

    INPUT:
        - x: np.ndarray -> model representation, shape (5, samples)
        - projection_matrix: np.ndarray -> stacked A, B, C rows, shape (3, 3)
        - rng: np.random.Generator -> seeded random-number generator
        - linear_noise_std: float -> noise standard deviation for y3, y4, and y5
        - nuisance_noise_std: float -> noise standard deviation for y6 and y7
        - second_cos_phase: float -> phase added to x2 before applying cosine

    OUTPUT:
        - y: np.ndarray -> neural representation, shape (7, samples)
        - clean_linear_y: np.ndarray -> noiseless A/B/C projections, shape (3, samples)
    """

    def build_neural_representation(
        x,
        projection_matrix,
        rng,
        linear_noise_std,
        nuisance_noise_std,
        second_cos_phase,
    ):
        if x.ndim != 2 or x.shape[0] != 5:
            raise ValueError("x must have shape (5, samples).")
        # end if x has an invalid shape
        if projection_matrix.shape != (3, 3):
            raise ValueError("The stacked A/B/C projection matrix must be 3 x 3.")
        # end if projection_matrix has an invalid shape
        if linear_noise_std < 0 or nuisance_noise_std < 0:
            raise ValueError("Noise standard deviations must be non-negative.")
        # end if either noise standard deviation is negative

        # Cosine creates periodic, generally non-invertible coordinates.
        cosine_y = np.vstack(
            (
                np.cos(x[0]),
                np.cos(x[1] + second_cos_phase),
            )
        )

        # Three independent rows mix the final three latent coordinates.
        linear_latents = x[2:5]
        clean_linear_y = projection_matrix @ linear_latents
        linear_noise = rng.normal(
            loc=0.0,
            scale=linear_noise_std,
            size=clean_linear_y.shape,
        )

        # These last two coordinates contain no information about x.
        nuisance_y = rng.normal(
            loc=0.0,
            scale=nuisance_noise_std,
            size=(2, x.shape[1]),
        )
        y = np.vstack((cosine_y, clean_linear_y + linear_noise, nuisance_y))
        return y, clean_linear_y

    # EOF

    if cfg.n_samples < 8:
        raise ValueError("n_samples must be at least 8 for the seven-dimensional example.")
    # end if cfg.n_samples is too small
    if cfg.x1_max <= cfg.x1_min:
        raise ValueError("x1_max must be greater than x1_min.")
    # end if the x1 interval is invalid
    if cfg.x2_max <= cfg.x2_min:
        raise ValueError("x2_max must be greater than x2_min.")
    # end if the x2 interval is invalid
    if cfg.gaussian_x_std <= 0:
        raise ValueError("gaussian_x_std must be positive.")
    # end if cfg.gaussian_x_std is non-positive

    projection_matrix = np.asarray((cfg.A, cfg.B, cfg.C), dtype=float)
    if np.linalg.matrix_rank(projection_matrix) < 3:
        raise ValueError("A, B, and C must form a full-rank 3 x 3 matrix.")
    # end if projection_matrix is rank deficient

    rng = np.random.default_rng(cfg.random_seed)
    cosine_x = np.vstack(
        (
            rng.uniform(cfg.x1_min, cfg.x1_max, size=cfg.n_samples),
            rng.uniform(cfg.x2_min, cfg.x2_max, size=cfg.n_samples),
        )
    )
    gaussian_x = rng.normal(
        loc=cfg.gaussian_x_mean,
        scale=cfg.gaussian_x_std,
        size=(3, cfg.n_samples),
    )
    x = np.vstack((cosine_x, gaussian_x))
    y, clean_linear_y = build_neural_representation(
        x=x,
        projection_matrix=projection_matrix,
        rng=rng,
        linear_noise_std=cfg.linear_noise_std,
        nuisance_noise_std=cfg.nuisance_noise_std,
        second_cos_phase=cfg.second_cos_phase,
    )

    x_feature_names = [f"x{idx}" for idx in range(1, 6)]
    y_feature_names = [
        "cos(x1)",
        "cos(x2 + pi)",
        "A @ [x3,x4,x5] + noise",
        "B @ [x3,x4,x5] + noise",
        "C @ [x3,x4,x5] + noise",
        "independent noise 1",
        "independent noise 2",
    ]

    print(f"x shape: {x.shape}")
    print(f"y shape: {y.shape}")
    print(f"rank([A; B; C]): {np.linalg.matrix_rank(projection_matrix)}")
    return (
        clean_linear_y,
        projection_matrix,
        x,
        x_feature_names,
        y,
        y_feature_names,
    )


@app.cell
def _(np, pd, projection_matrix):
    projection_table = pd.DataFrame(
        projection_matrix,
        index=["A", "B", "C"],
        columns=["x3", "x4", "x5"],
    )
    print(f"det([A; B; C]) = {np.linalg.det(projection_matrix):.3f}")
    projection_table.style.format(precision=2).set_caption(
        "Linear projection vectors"
    )
    return


@app.cell
def _(clean_linear_y, plt, x, y):
    # Show the two nonlinear coordinates, the noisy linear block, and nuisances.
    _fig, _axes = plt.subplots(2, 2, figsize=(11, 8.5))

    _axes[0, 0].scatter(x[0], y[0], s=8, alpha=0.45)
    _axes[0, 0].set(
        xlabel=r"$x_1$",
        ylabel=r"$y_1$",
        title=r"$y_1=\cos(x_1)$",
    )

    _axes[0, 1].scatter(x[1], y[1], s=8, alpha=0.45, color="tab:orange")
    _axes[0, 1].set(
        xlabel=r"$x_2$",
        ylabel=r"$y_2$",
        title=r"$y_2=\cos(x_2+\pi)=-\cos(x_2)$",
    )

    for _linear_idx, _color in enumerate(("tab:blue", "tab:orange", "tab:green")):
        _axes[1, 0].scatter(
            clean_linear_y[_linear_idx],
            y[_linear_idx + 2],
            s=8,
            alpha=0.35,
            color=_color,
            label=f"y{_linear_idx + 3}",
        )
    # end for _linear_idx, _color
    _linear_limits = _axes[1, 0].get_xlim()
    _axes[1, 0].plot(_linear_limits, _linear_limits, color="black", lw=1)
    _axes[1, 0].set(
        xlabel="noiseless linear projection",
        ylabel="observed coordinate",
        title="Mildly noisy A/B/C projections",
    )
    _axes[1, 0].legend()

    _axes[1, 1].hist(y[5], bins=35, alpha=0.55, label=r"$y_6$")
    _axes[1, 1].hist(y[6], bins=35, alpha=0.55, label=r"$y_7$")
    _axes[1, 1].set(
        xlabel="value",
        ylabel="count",
        title="Independent nuisance coordinates",
    )
    _axes[1, 1].legend()

    _fig.suptitle("Structure of the seven-dimensional y representation")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fit the two static linear projections

    `predicted_x` is the $y\rightarrow\hat{x}$ prediction and `predicted_y` is
    the $x\rightarrow\hat{y}$ prediction. The cosine coordinates are
    deterministic, but how well a linear map approximates them depends on the
    selected uniform intervals. The A/B/C block is linearly recoverable up to
    mild noise, and the final two $y$ coordinates are irreducible nuisance
    noise.
    """)
    return


@app.cell
def _(cfg, fit_static_projection, linear_encoding, np, x, y):
    if cfg.regression_type not in {"lr", "ridge", "lasso", "en"}:
        raise ValueError(f"Unsupported regression_type: {cfg.regression_type}")
    # end if cfg.regression_type is unsupported
    if cfg.use_cv and cfg.cv_type not in {"loo", "kf"}:
        raise ValueError("Use 'loo' or 'kf' when use_cv=True.")
    # end if cfg.use_cv and cfg.cv_type is unsupported

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

    # Reset the seed before each direction so CV uses the same fold assignments.
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
    return predicted_x, predicted_y, x_r2, y_r2


@app.cell
def _(np, pd, x_feature_names, x_r2, y_feature_names, y_r2):
    # Per-output scores reveal which coordinates a linear map can recover.
    encoding_scores = pd.concat(
        (
            pd.DataFrame(
                {
                    "direction": "y -> x",
                    "target": x_feature_names,
                    "R2": np.asarray(x_r2),
                }
            ),
            pd.DataFrame(
                {
                    "direction": "x -> y",
                    "target": y_feature_names,
                    "R2": np.asarray(y_r2),
                }
            ),
        ),
        ignore_index=True,
    )
    encoding_scores.style.format({"R2": "{:.4f}"}).background_gradient(
        subset=["R2"],
        cmap="viridis",
        vmin=0,
        vmax=1,
    ).set_caption("Per-target linear encoding scores")
    return (encoding_scores,)


@app.cell
def _(cfg, encoding_scores, np, plt):
    encoding_mode = "out-of-fold" if cfg.use_cv else "in-sample"
    _directions = ("y -> x", "x -> y")
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for _ax, _direction in zip(_axes, _directions):
        _direction_scores = encoding_scores[
            encoding_scores["direction"] == _direction
        ]
        _positions = np.arange(len(_direction_scores))
        _ax.bar(_positions, _direction_scores["R2"], color="tab:blue")
        _ax.set_xticks(
            _positions,
            _direction_scores["target"],
            rotation=35,
            ha="right",
        )
        _ax.axhline(0, color="black", linewidth=0.8)
        _ax.set_ylabel(r"$R^2$")
        _ax.set_title(_direction)
    # end for _ax, _direction

    _fig.suptitle(f"Per-target static linear encoding ({encoding_mode})")
    _fig.tight_layout()
    _fig
    return (encoding_mode,)


@app.cell
def _(encoding_mode, plt, predicted_x, predicted_y, x, y):
    # Observed-versus-predicted plots summarize every target dimension at once.
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.8))
    _axes[0].scatter(x.ravel(), predicted_x.ravel(), s=7, alpha=0.25)
    _axes[0].set(
        xlabel="observed x coordinates",
        ylabel="predicted x coordinates",
        title=r"$y\rightarrow\hat{x}$",
    )

    _axes[1].scatter(y.ravel(), predicted_y.ravel(), s=7, alpha=0.25)
    _axes[1].set(
        xlabel="observed y coordinates",
        ylabel="predicted y coordinates",
        title=r"$x\rightarrow\hat{y}$",
    )

    for _ax in _axes:
        _minimum = min(_ax.get_xlim()[0], _ax.get_ylim()[0])
        _maximum = max(_ax.get_xlim()[1], _ax.get_ylim()[1])
        _ax.plot([_minimum, _maximum], [_minimum, _maximum], color="black", lw=1)
    # end for _ax in _axes

    _fig.suptitle(f"Observed and linearly predicted values ({encoding_mode})")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Information Imbalance between predicted spaces

    The project helper labels its first input as neural and its second input as
    model. Here those are `predicted_y` and `predicted_x`, so `A2B` corresponds
    to predicted $y\rightarrow x$, while `B2A` corresponds to predicted
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
    _fig, _ax = plt.subplots(figsize=(5.5, 4.2))
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

    ii_table.style.format(precision=3, na_rep="--").background_gradient(
        cmap="viridis",
        axis=None,
        vmin=0,
        vmax=1,
    ).set_caption("Directional Information Imbalance: row source -> column target")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Normalize the reciprocal OLS maps by their singular values

    The shared asymmetry pipeline fits both full-data OLS directions and uses a
    compact SVD to replace retained singular values with one. By default, the
    numerical rank of each fitted map is preserved. Set `singular_value_rtol`
    in `Cfg` if weak fitted directions should be removed.
    """)
    return


@app.cell
def _(cfg, np, run_asymmetry_pipeline, summarize_asymmetry_pipeline, x, y):
    # The shared pipeline expects samples in rows.
    svd_pipeline = run_asymmetry_pipeline(
        source=x.T,
        target=y.T,
        source_metric=cfg.x_RDM_metric,
        target_metric=cfg.y_RDM_metric,
        k=cfg.k,
        relative_tolerance=cfg.singular_value_rtol,
    )
    svd_summary = summarize_asymmetry_pipeline(svd_pipeline, "x", "y")

    x_tilde = svd_pipeline["transformed_source"].T
    y_tilde = svd_pipeline["transformed_target"].T

    for _label, _svd_result in (
        ("x -> y", svd_pipeline["source_svd"]),
        ("y -> x", svd_pipeline["target_svd"]),
    ):
        print(f"{_label}: retained rank={_svd_result['rank']}")
        print("singular values:", np.round(_svd_result["singular_values"], 6))
    # end for _label, _svd_result

    print(f"x_tilde shape: {x_tilde.shape}")
    print(f"y_tilde shape: {y_tilde.shape}")
    assert x_tilde.shape == (y.shape[0], cfg.n_samples)
    assert y_tilde.shape == (x.shape[0], cfg.n_samples)
    return svd_summary, x_tilde, y_tilde


@app.cell
def _(svd_summary):
    svd_summary.style.format({"pooled R2": "{:.6f}", "II": "{:.6f}"})
    return


@app.cell
def _(np, plt, svd_summary):
    # Compare directional asymmetry before and after singular-value normalization.
    _original_metrics = svd_summary[svd_summary["stage"] == "original"]
    _normalized_metrics = svd_summary[
        svd_summary["stage"] == "Sigma^-1 normalized"
    ]
    _bar_labels = [
        r"$x \to y$",
        r"$y \to x$",
        r"$\widetilde{x} \to \widetilde{y}$",
        r"$\widetilde{y} \to \widetilde{x}$",
    ]
    _bar_positions = np.array([0.0, 1.0, 2.4, 3.4])
    _bar_colors = ["tab:blue", "tab:blue", "tab:orange", "tab:orange"]

    _fig, _axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for _ax, _metric_name, _ylabel in (
        (_axes[0], "pooled R2", r"pooled $R^2$"),
        (_axes[1], "II", "1 - Information Imbalance"),
    ):
        _original_values = _original_metrics[_metric_name].to_numpy()
        _normalized_values = _normalized_metrics[_metric_name].to_numpy()
        _values = np.concatenate((_original_values, _normalized_values))
        if _metric_name == "II":
            _values = 1 - _values
        # end if _metric_name is II

        _ax.bar(_bar_positions, _values, width=0.8, color=_bar_colors)
        _ax.set_xticks(_bar_positions, _bar_labels, rotation=12)
        _ax.set_ylabel(_ylabel)
        _ax.set_title(_ylabel)
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
    ### Geometry after normalization

    The transformed spaces are multidimensional, so each is visualized through
    its first two principal directions. Points are colored by $x_1$; matched
    colors let us compare how the cosine fold appears across representations.
    """)
    return


@app.cell
def _(np, plt, x, x_tilde, y, y_tilde):
    """
    first_two_principal_coordinates
    Project a feature-by-sample representation onto its first two centered PCs.

    INPUT:
        - space: np.ndarray -> representation, shape (features, samples)

    OUTPUT:
        - coordinates: np.ndarray -> sample coordinates, shape (samples, 2)
    """

    def first_two_principal_coordinates(space):
        centered_space = space - np.mean(space, axis=1, keepdims=True)
        _, singular_values, right_vectors = np.linalg.svd(
            centered_space,
            full_matrices=False,
        )
        # V contains sample directions; multiplying by Sigma gives PC scores.
        coordinates = right_vectors[:2].T * singular_values[:2]
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack((coordinates[:, 0], np.zeros(space.shape[1])))
        # end if the space has only one principal direction
        return coordinates

    # EOF

    _spaces = (x, y, x_tilde, y_tilde)
    _titles = (r"original $x$", r"original $y$", r"$\widetilde{x}$", r"$\widetilde{y}$")
    _fig, _axes = plt.subplots(2, 2, figsize=(10, 8.5))

    for _ax, _space, _title in zip(_axes.ravel(), _spaces, _titles):
        _coordinates = first_two_principal_coordinates(_space)
        _scatter = _ax.scatter(
            _coordinates[:, 0],
            _coordinates[:, 1],
            c=x[0],
            cmap="coolwarm",
            s=9,
            alpha=0.55,
        )
        _ax.set(xlabel="PC1", ylabel="PC2", title=_title)
    # end for _ax, _space, _title

    _fig.colorbar(_scatter, ax=_axes, label=r"$x_1$", shrink=0.8)
    _fig.suptitle("Original and singular-value-normalized geometries")
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
