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
    # Two 2D Gaussians sharing the dominant coordinate of $X$

    The source space is

    $$
    X=[x_1,\;x_2],\qquad x_1\sim\mathcal{N}(0,\sigma_1^2),\quad x_2\sim\mathcal{N}(0,\sigma_2^2),
    $$

    with $\sigma_1^2$ chosen so that $x_1$ holds 90% of the variance of $X$.
    The target space reuses the *dominant* coordinate of $X$,

    $$
    Y=[y_1,\;a\,x_1],\qquad y_1\sim\mathcal{N}(0,\sigma_{y_1}^2)\ \text{independent of } X,
    $$

    and $\sigma_{y_1}^2$ is fixed by asking $y_1$ to hold 90% of the variance
    of $Y$: $\sigma_{y_1}^2=\frac{f_y}{1-f_y}a^2\sigma_1^2$.

    So $x_1$ is the only shared information, and it carries 90% of $X$ but
    only 10% of $Y$. That mismatch is what makes the two directions differ:
    almost all of $X$ is written into $Y$, yet it lands in the direction that
    barely moves $Y$. The notebook computes bidirectional OLS $R^2$ and the
    Information Imbalance in both directions.

    Note that, once the 90% constraint is imposed, $a$ only rescales the whole
    of $Y$: it changes neither $R^2$ (a ratio of sums of squares) nor II
    (a function of distance *ranks*). A cell below checks that numerically,
    and the sweep at the end varies the quantity that does matter, the
    variance share $f_y$ of $y_1$.
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

    # marimo stores every figure at 2 x figure.dpi, so a lower dpi keeps the
    # embedded PNG small without changing the on-screen size.
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
        compute_bidirectional_ii,
        fit_bidirectional_ols,
        generate_variance_split_gaussians,
    )

    return (
        compute_bidirectional_ii,
        dataclass,
        fit_bidirectional_ols,
        generate_variance_split_gaussians,
        np,
        pd,
        plt,
    )


@app.cell
def _(dataclass):
    @dataclass
    class Cfg:
        # Synthetic data.
        n_samples: int = 1000
        source_leading_share: float = 0.9  # share of var(X) carried by x1
        target_leading_share: float = 0.9  # share of var(Y) carried by y1
        shared_gain: float = 1.0  # the "a" multiplying x1 inside Y
        source_total_variance: float = 1.0
        random_seed: int = 0

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
    ## Generate $X$ and $Y$
    """)
    return


@app.cell
def _(cfg, generate_variance_split_gaussians, np, pd):
    # The pipeline convention here is (samples, features).
    X, Y, theoretical_variances = generate_variance_split_gaussians(
        n_samples=cfg.n_samples,
        source_leading_share=cfg.source_leading_share,
        target_leading_share=cfg.target_leading_share,
        shared_gain=cfg.shared_gain,
        source_total_variance=cfg.source_total_variance,
        seed=cfg.random_seed,
    )

    # Empirical shares confirm that the intended 90/10 split survived sampling.
    empirical_x_variances = X.var(axis=0, ddof=1)
    empirical_y_variances = Y.var(axis=0, ddof=1)
    variance_table = pd.DataFrame(
        {
            "coordinate": ["x1", "x2", "y1", "y2 = a*x1"],
            "theoretical var": [
                theoretical_variances["x1"],
                theoretical_variances["x2"],
                theoretical_variances["y1"],
                theoretical_variances["y2"],
            ],
            "empirical var": np.concatenate(
                (empirical_x_variances, empirical_y_variances)
            ),
            "empirical share of its space": np.concatenate(
                (
                    empirical_x_variances / empirical_x_variances.sum(),
                    empirical_y_variances / empirical_y_variances.sum(),
                )
            ),
        }
    )
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    variance_table.style.format(precision=4).set_caption(
        "Variance split of the two spaces"
    )
    return X, Y


@app.cell
def _(X, Y, cfg, plt):
    # The two clouds and the shared coordinate seen from both sides.
    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4.2))

    _axes[0].scatter(X[:, 0], X[:, 1], s=7, alpha=0.35)
    _axes[0].set(xlabel=r"$x_1$", ylabel=r"$x_2$", title="X: variance along $x_1$")
    _axes[0].set_aspect("equal", adjustable="datalim")

    _axes[1].scatter(Y[:, 0], Y[:, 1], s=7, alpha=0.35, color="tab:orange")
    _axes[1].set(xlabel=r"$y_1$", ylabel=r"$a\,x_1$", title="Y: variance along $y_1$")
    _axes[1].set_aspect("equal", adjustable="datalim")

    _axes[2].scatter(X[:, 0], Y[:, 1], s=7, alpha=0.35, color="tab:green")
    _axes[2].set(
        xlabel=r"$x_1$",
        ylabel=r"$a\,x_1$",
        title=f"shared coordinate, a={cfg.shared_gain}",
    )

    _fig.suptitle("The two Gaussians share only $x_1$: dominant in X, minor in Y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Bidirectional OLS $R^2$

    Each direction is fitted on the full data (no cross-validation): the maps
    are linear and exact, so the in-sample fit already tells the whole story.
    Per-output $R^2$ separates the recoverable coordinate from the private one,
    and the pooled $R^2$ weights them by how much variance each holds.
    """)
    return


@app.cell
def _(X, Y, fit_bidirectional_ols, pd):
    ols_fit = fit_bidirectional_ols(X, Y)

    r2_table = pd.DataFrame(
        {
            "direction": ["X -> Y", "Y -> X"],
            "R2 per output": [
                [round(float(value), 6) for value in ols_fit["r2_source_to_target"]],
                [round(float(value), 6) for value in ols_fit["r2_target_to_source"]],
            ],
            "pooled R2": [
                ols_fit["pooled_r2_source_to_target"],
                ols_fit["pooled_r2_target_to_source"],
            ],
        }
    )
    r2_table.style.format({"pooled R2": "{:.4f}"}).set_caption(
        "Full-data linear encoding, both directions"
    )
    return (ols_fit,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Information Imbalance

    `II(X -> Y)` asks how well neighbours in $X$ stay neighbours in $Y$. Here
    the two directions are not interchangeable: distances in $X$ are set by
    $x_1$, which $Y$ does contain, so $X\rightarrow Y$ is informative; but
    distances in $Y$ are set by $y_1$, which $X$ never sees, so
    $Y\rightarrow X$ stays close to 1.
    """)
    return


@app.cell
def _(X, Y, cfg, compute_bidirectional_ii):
    ii_result = compute_bidirectional_ii(
        X,
        Y,
        source_metric=cfg.x_RDM_metric,
        target_metric=cfg.y_RDM_metric,
        k=cfg.k,
    )
    print(f"II(X -> Y) = {ii_result['source_to_target']:.4f}")
    print(f"II(Y -> X) = {ii_result['target_to_source']:.4f}")
    return (ii_result,)


@app.cell
def _(cfg, ii_result, ols_fit, plt):
    _fig, _axes = plt.subplots(1, 2, figsize=(10, 4.0))
    _directions = ["X -> Y", "Y -> X"]
    _colors = ["tab:blue", "tab:orange"]

    _axes[0].bar(
        _directions,
        [
            ols_fit["pooled_r2_source_to_target"],
            ols_fit["pooled_r2_target_to_source"],
        ],
        color=_colors,
    )
    _axes[0].set(ylabel=r"pooled $R^2$", ylim=(0, 1), title="Linear predictability")

    _axes[1].bar(
        _directions,
        [1-ii_result["source_to_target"], 1-ii_result["target_to_source"]],
        color=_colors,
    )
    _axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
    _axes[1].set(
        ylabel="1-Information Imbalance",
        ylim=(0, 1.15),
        title=f"II, k={cfg.k}",
    )

    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Does $a$ matter?

    With the 90% constraint in place, changing $a$ multiplies both coordinates
    of $Y$ by the same factor, so it is a pure global rescaling of $Y$. Both
    measures should be blind to it.
    """)
    return


@app.cell
def _(
    cfg,
    compute_bidirectional_ii,
    fit_bidirectional_ols,
    generate_variance_split_gaussians,
    pd,
):
    _gain_rows = []
    for _gain in (0.1, 1.0, 10.0):
        # The seed is reused, so the underlying draws are identical every time.
        _X, _Y, _ = generate_variance_split_gaussians(
            n_samples=cfg.n_samples,
            source_leading_share=cfg.source_leading_share,
            target_leading_share=cfg.target_leading_share,
            shared_gain=_gain,
            source_total_variance=cfg.source_total_variance,
            seed=cfg.random_seed,
        )
        _fit = fit_bidirectional_ols(_X, _Y)
        _ii = compute_bidirectional_ii(
            _X, _Y, cfg.x_RDM_metric, cfg.y_RDM_metric, cfg.k
        )
        _gain_rows.append(
            {
                "a": _gain,
                "pooled R2 X -> Y": _fit["pooled_r2_source_to_target"],
                "pooled R2 Y -> X": _fit["pooled_r2_target_to_source"],
                "II X -> Y": _ii["source_to_target"],
                "II Y -> X": _ii["target_to_source"],
            }
        )
    # end for _gain in the gain grid

    gain_table = pd.DataFrame(_gain_rows)
    gain_table.style.format(precision=6).set_caption(
        "Both measures are invariant to the shared gain a"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Sweep the variance share of $y_1$

    What does change the picture is *how much* of $Y$ the private coordinate
    $y_1$ occupies. The $Y\rightarrow X$ side is pinned: $x_1$ is always
    recovered exactly and $x_2$ never is, so pooled $R^2$ stays at the variance
    share of $x_1$. The $X\rightarrow Y$ side instead degrades as $y_1$ grows,
    because the shared coordinate explains a shrinking part of $Y$.
    """)
    return


@app.cell
def _(
    cfg,
    compute_bidirectional_ii,
    fit_bidirectional_ols,
    generate_variance_split_gaussians,
    np,
    pd,
):
    _share_rows = []
    for _share in np.round(np.arange(0.1, 0.95, 0.1), 2):
        _X, _Y, _ = generate_variance_split_gaussians(
            n_samples=cfg.n_samples,
            source_leading_share=cfg.source_leading_share,
            target_leading_share=float(_share),
            shared_gain=cfg.shared_gain,
            source_total_variance=cfg.source_total_variance,
            seed=cfg.random_seed,
        )
        _fit = fit_bidirectional_ols(_X, _Y)
        _ii = compute_bidirectional_ii(
            _X, _Y, cfg.x_RDM_metric, cfg.y_RDM_metric, cfg.k
        )
        _share_rows.append(
            {
                "var share of y1": float(_share),
                "pooled R2 X -> Y": _fit["pooled_r2_source_to_target"],
                "pooled R2 Y -> X": _fit["pooled_r2_target_to_source"],
                "II X -> Y": _ii["source_to_target"],
                "II Y -> X": _ii["target_to_source"],
            }
        )
    # end for _share in the sweep grid

    share_table = pd.DataFrame(_share_rows)
    share_table.style.format(precision=4).set_caption(
        "Sweep over the variance share of the private coordinate y1"
    )
    return (share_table,)


@app.cell
def _(plt, share_table):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _shares = share_table["var share of y1"]

    _axes[0].plot(_shares, share_table["pooled R2 X -> Y"], "o-", label="X -> Y")
    _axes[0].plot(_shares, share_table["pooled R2 Y -> X"], "s-", label="Y -> X")
    _axes[0].set(xlabel=r"variance share of $y_1$", ylabel=r"pooled $R^2$", ylim=(0, 1))
    _axes[0].legend()

    _axes[1].plot(1 - _shares, share_table["II X -> Y"], "o-", label="X -> Y")
    _axes[1].plot(1 - _shares, share_table["II Y -> X"], "s-", label="Y -> X")
    _axes[1].axhline(1.0, color="black", lw=0.8, ls="--")
    _axes[1].set(xlabel=r"variance share of $y_1$", ylabel="1-Information Imbalance")
    _axes[1].legend()

    _fig.suptitle("The two spaces weight the shared $x_1$ differently")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
