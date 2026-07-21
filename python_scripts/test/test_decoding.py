import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from II_analyses.decoding import (  # noqa: E402
    compute_centroid_trial_metrics,
    crossvalidate_decoding_over_repetitions,
    find_margin_only_differences,
    paired_normalization_statistics,
)


class TestCentroidTrialMetrics(unittest.TestCase):
    def test_signed_margin_normalized_margin_and_rank(self):
        distances = np.array(
            [
                [0.25, 2.25, 9.25],
                [3.24, 0.04, 12.24],
            ]
        )
        metrics = compute_centroid_trial_metrics(
            distances,
            targets=np.array([0, 0]),
            classes=np.array([0, 1, 2]),
        )

        np.testing.assert_allclose(metrics["d_correct"], [0.25, 3.24])
        np.testing.assert_allclose(metrics["d_second"], [2.25, 0.04])
        np.testing.assert_allclose(metrics["margin"], [2.0, -3.2])
        np.testing.assert_allclose(metrics["margin_norm"], [0.8, -3.2 / 3.28])
        np.testing.assert_array_equal(metrics["correct_rank"], [1, 2])

    def test_zero_distance_denominator_is_stable(self):
        metrics = compute_centroid_trial_metrics(
            np.zeros((1, 2)),
            targets=np.array([0]),
            classes=np.array([0, 1]),
        )

        np.testing.assert_array_equal(metrics["margin_norm"], [0.0])
        np.testing.assert_array_equal(metrics["correct_rank"], [1])


class TestRepetitionDecoding(unittest.TestCase):
    def test_crossvalidation_retains_outputs_and_old_accuracy_keys(self):
        # Three compact clusters remain separated in every repetition and time bin.
        X_rep = np.array(
            [
                [[[0.0, 0.1], [0.0, 0.1]], [[0.1, 0.0], [0.0, 0.1]], [[-0.1, 0.0], [0.0, -0.1]]],
                [[[3.0, 3.1], [0.0, 0.1]], [[3.1, 3.0], [0.0, -0.1]], [[2.9, 3.0], [0.0, 0.0]]],
                [[[0.0, 0.1], [3.0, 3.1]], [[0.1, 0.0], [3.1, 3.0]], [[-0.1, 0.0], [2.9, 3.0]]],
            ],
            dtype=float,
        )
        result = crossvalidate_decoding_over_repetitions(
            X_rep,
            preprocessing=dict(
                normalize=False,
                feature_center=False,
                mean_center=False,
            ),
        )

        self.assertEqual(result["predictions"].shape, (3, 2, 3))
        self.assertEqual(result["centroid_distances"].shape, (3, 2, 3, 3))
        self.assertEqual(result["margin"].shape, (3, 2, 3))
        self.assertEqual(result["correct_rank"].shape, (3, 2, 3))
        np.testing.assert_allclose(result["mean_accuracy"], 1.0)
        np.testing.assert_array_equal(result["correct_rank"], 1)
        np.testing.assert_allclose(
            result["mean_margin"],
            result["outcome_summary"]["correct"]["mean_margin"],
        )
        np.testing.assert_array_equal(
            result["outcome_summary"]["incorrect"]["n_margin_trials"],
            0,
        )

        # A shared feature translation must preserve all Euclidean distances.
        centered_result = crossvalidate_decoding_over_repetitions(
            X_rep,
            preprocessing=dict(
                normalize=False,
                feature_center=True,
                mean_center=False,
            ),
        )
        np.testing.assert_allclose(
            centered_result["centroid_distances"],
            result["centroid_distances"],
        )
        np.testing.assert_allclose(centered_result["margin"], result["margin"])


class TestPairedStatistics(unittest.TestCase):
    def test_flags_margin_difference_when_accuracy_is_identical(self):
        n_folds = 8
        n_time = 2
        n_samples = 3
        correct = np.ones((n_folds, n_time, n_samples), dtype=bool)
        common = {
            "fold_indices": np.arange(n_folds),
            "targets": np.arange(n_samples),
            "correct": correct,
            "fold_accuracies": np.ones((n_folds, n_time)),
        }
        results = {
            "raw": {
                **common,
                "fold_mean_margin_norm": np.zeros((n_folds, n_time)),
            },
            "norm": {
                **common,
                "fold_mean_margin_norm": np.ones((n_folds, n_time)) * 0.2,
            },
        }

        comparisons = paired_normalization_statistics(
            results,
            baseline="raw",
            metrics=("accuracy", "margin_norm"),
        )
        findings = find_margin_only_differences(
            comparisons, time_s=np.array([0.0, 0.01])
        )

        np.testing.assert_array_equal(
            comparisons["norm"]["accuracy_identical"], True
        )
        np.testing.assert_array_equal(
            comparisons["norm"]["accuracy_outcomes_identical"], True
        )
        self.assertEqual(len(findings), 2)
        self.assertTrue(all(finding["qvalue"] < 0.05 for finding in findings))


if __name__ == "__main__":
    unittest.main()
