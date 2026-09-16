import numpy as np
import pytest

from II_analyses.asymmetry_pipeline import (
    compute_pc_coupling_matrix,
    compute_variance_asymmetry,
    decompose_pooled_r2_in_pc_basis,
    fit_bidirectional_ols,
    generate_gaussian_with_orthogonal_noise,
    generate_parabola_branch,
    pooled_r2,
    run_asymmetry_pipeline,
    sigma_inverse_row_transform,
)


def test_synthetic_generators_return_sample_first_spaces():
    parabola_source, parabola_target = generate_parabola_branch(20)
    gaussian_source, gaussian_target = generate_gaussian_with_orthogonal_noise(
        20, orthogonal_noise_scale=0.5
    )

    assert parabola_source.shape == (20, 1)
    assert parabola_target.shape == (20, 2)
    assert gaussian_source.shape == (20, 2)
    assert gaussian_target.shape == (20, 3)
# EOF


def test_pooled_r2_for_perfect_prediction():
    observed = np.array([[0.0, 2.0], [1.0, 4.0], [2.0, 8.0]])

    assert pooled_r2(observed, observed) == pytest.approx(1.0)
# EOF


def test_sigma_inverse_transform_sets_active_singular_values_to_one():
    rng = np.random.default_rng(4)
    source = rng.normal(size=(80, 2))
    target = source @ np.diag([4.0, 0.25])
    fit_result = fit_bidirectional_ols(source, target)

    transform_result = sigma_inverse_row_transform(
        fit_result["source_to_target"]
    )
    transformed_singular_values = np.linalg.svd(
        transform_result["row_transform"], compute_uv=False
    )

    np.testing.assert_allclose(transformed_singular_values, np.ones(2))
    assert transform_result["rank"] == 2
# EOF


def test_pipeline_returns_both_metric_stages():
    rng = np.random.default_rng(8)
    source = rng.normal(size=(60, 2))
    target = np.column_stack((source[:, 0], rng.normal(size=60)))

    result = run_asymmetry_pipeline(source, target, k=1)

    assert result["transformed_source"].shape == (60, 2)
    assert result["transformed_target"].shape == (60, 2)
    assert np.isfinite(result["transformed_fit"]["pooled_r2_source_to_target"])
    assert np.isfinite(result["transformed_ii"]["source_to_target"])
# EOF


def test_pc_pairwise_coupling_matches_exact_in_sample_ols_decomposition():
    rng = np.random.default_rng(12)
    shared = rng.normal(size=400)
    X = np.column_stack((3.0 * shared, rng.normal(size=400)))
    Y = np.column_stack((0.4 * shared, 2.0 * rng.normal(size=400)))

    pc_result = compute_pc_coupling_matrix(X, Y)
    decomposition = decompose_pooled_r2_in_pc_basis(
        pc_result["scores_X"],
        pc_result["scores_Y"],
        pc_result["variance_ratio_X"],
        pc_result["variance_ratio_Y"],
    )
    raw_fit = fit_bidirectional_ols(X, Y)
    coupling = pc_result["coupling"]
    pairwise_Y_to_X = np.sum(
        pc_result["variance_ratio_X"] * coupling.sum(axis=1)
    )
    pairwise_X_to_Y = np.sum(
        pc_result["variance_ratio_Y"] * coupling.sum(axis=0)
    )

    assert pairwise_Y_to_X == pytest.approx(
        decomposition["pooled_r2_Y_to_X"], abs=1e-10
    )
    assert pairwise_X_to_Y == pytest.approx(
        decomposition["pooled_r2_X_to_Y"], abs=1e-10
    )
    assert decomposition["pooled_r2_Y_to_X"] == pytest.approx(
        raw_fit["pooled_r2_target_to_source"], abs=1e-10
    )
    assert decomposition["pooled_r2_X_to_Y"] == pytest.approx(
        raw_fit["pooled_r2_source_to_target"], abs=1e-10
    )
# EOF


def test_signed_and_unsigned_variance_mismatch_can_separate():
    coupling = np.eye(2)
    variance_X = np.array([0.8, 0.2])
    variance_Y = np.array([0.2, 0.8])

    result = compute_variance_asymmetry(coupling, variance_X, variance_Y)

    assert result["A_var"] == pytest.approx(0.0)
    assert result["A_var_norm"] == pytest.approx(0.0)
    assert result["M_var"] == pytest.approx(0.6)
# EOF
