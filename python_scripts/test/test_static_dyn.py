import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "python_scripts" / "src"))

from II_analyses.static_dyn import (  # noqa: E402
    compute_static_dCKA_subsampled,
    compute_static_dRSA_subsampled,
    init_static_dCKA,
    init_static_dRSA,
    static_dCKA_save_name,
    static_dRSA_save_name,
)
from useful_stuff.general_utils.utils import TimeSeries  # noqa: E402


class TestStaticDRSASubsampling(unittest.TestCase):
    def test_save_name_preserves_base_and_appends_subsampling(self):
        paths = {"data_path": "/tmp/data"}

        base_name = static_dRSA_save_name(
            paths,
            "cosine",
            "euclidean",
            "three0",
            "250313",
            "AIT",
            "vit_l_16",
            384,
            "blocks.0.mlp.fc2",
            100,
        )
        subsampled_name = static_dRSA_save_name(
            paths,
            "cosine",
            "euclidean",
            "three0",
            "250313",
            "AIT",
            "vit_l_16",
            384,
            "blocks.0.mlp.fc2",
            100,
            subsamples_size=4,
            n_iterations=3,
        )

        self.assertTrue(base_name.endswith("_100Hz.npz"))
        self.assertTrue(
            subsampled_name.endswith("_100Hz_4subsamples_3iterations.npz")
        )

    def test_computation_matches_mean_of_seeded_subsets(self):
        random_seed = 7
        subsamples_size = 4
        n_iterations = 3
        fs = 100
        layer_name = "layer0"
        raster_array = np.random.default_rng(1).normal(size=(3, 2, 6))
        features = np.random.default_rng(2).normal(size=(4, 6))
        raster = TimeSeries(raster_array, fs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory)
            (data_path / "models").mkdir()
            (data_path / "results").mkdir()
            np.savez_compressed(
                data_path / "models" / "stimuli_model_32_layer0_features_meanpool.npz",
                features,
            )

            result = compute_static_dRSA_subsampled(
                {"data_path": str(data_path)},
                1,
                layer_name,
                raster,
                np.arange(features.shape[1]),
                "euclidean",
                "euclidean",
                "monkey",
                "date",
                "AIT",
                "stimuli",
                "model",
                32,
                "mean",
                subsamples_size,
                n_iterations,
                random_seed,
            )

            # Independently replay the seeded subsets and average their curves.
            rng = np.random.default_rng(random_seed)
            expected_iterations = []
            for _ in range(n_iterations):
                subset = rng.choice(
                    raster_array.shape[2],
                    size=subsamples_size,
                    replace=False,
                )
                subset_raster = TimeSeries(raster_array[:, :, subset], fs)
                drsa_obj = init_static_dRSA(
                    subset_raster,
                    "euclidean",
                    "euclidean",
                )
                drsa_obj.compute_RDM(features[:, subset], "model")
                expected_iterations.append(
                    drsa_obj.compute_static_dRSA().get_array()
                )
            # end for _ in range(n_iterations)

            expected = np.mean(expected_iterations, axis=0)
            np.testing.assert_allclose(result.get_array(), expected)

            save_name = static_dRSA_save_name(
                {"data_path": str(data_path)},
                "euclidean",
                "euclidean",
                "monkey",
                "date",
                "AIT",
                "model",
                32,
                layer_name,
                fs,
                subsamples_size=subsamples_size,
                n_iterations=n_iterations,
            )
            np.testing.assert_allclose(np.load(save_name)["arr_0"], expected)
        # end with tempfile.TemporaryDirectory()


class TestStaticDCKASubsampling(unittest.TestCase):
    def test_computation_matches_mean_of_seeded_subsets(self):
        random_seed = 11
        subsamples_size = 4
        n_iterations = 3
        fs = 100
        layer_name = "layer0"
        raster_array = np.random.default_rng(3).normal(size=(3, 2, 6))
        features = np.random.default_rng(4).normal(size=(4, 6))
        raster = TimeSeries(raster_array, fs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            data_path = Path(temporary_directory)
            (data_path / "models").mkdir()
            (data_path / "results").mkdir()
            np.savez_compressed(
                data_path / "models" / "stimuli_model_32_layer0_features_meanpool.npz",
                features,
            )

            result = compute_static_dCKA_subsampled(
                {"data_path": str(data_path)},
                1,
                layer_name,
                raster,
                np.arange(features.shape[1]),
                "linear",
                "linear",
                "biased",
                "kernel",
                "kernel",
                "monkey",
                "date",
                "AIT",
                "stimuli",
                "model",
                32,
                "mean",
                subsamples_size,
                n_iterations,
                random_seed,
            )

            # Independently replay the seeded subsets and average their curves.
            rng = np.random.default_rng(random_seed)
            expected_iterations = []
            for _ in range(n_iterations):
                subset = rng.choice(
                    raster_array.shape[2],
                    size=subsamples_size,
                    replace=False,
                )
                subset_raster = TimeSeries(raster_array[:, :, subset], fs)
                dcka_obj = init_static_dCKA(
                    subset_raster,
                    "linear",
                    "linear",
                    "biased",
                    "kernel",
                    "kernel",
                )
                dcka_obj.compute_static_model_gram(features[:, subset])
                expected_iterations.append(
                    dcka_obj.compute_static_dCKA().get_array()
                )
            # end for _ in range(n_iterations)

            expected = np.mean(expected_iterations, axis=0)
            np.testing.assert_allclose(result.get_array(), expected)

            save_name = static_dCKA_save_name(
                {"data_path": str(data_path)},
                "linear",
                "linear",
                "biased",
                "kernel",
                "kernel",
                "monkey",
                "date",
                "AIT",
                "model",
                32,
                layer_name,
                fs,
                subsamples_size=subsamples_size,
                n_iterations=n_iterations,
            )
            self.assertTrue(
                save_name.endswith(
                    "static_dCKA_biased_kernel-linear_kernel-linear_"
                    "monkey_date_AIT_model_32_layer0_100Hz_"
                    "4subsamples_3iterations.npz"
                )
            )
            np.testing.assert_allclose(np.load(save_name)["arr_0"], expected)
        # end with tempfile.TemporaryDirectory()


if __name__ == "__main__":
    unittest.main()
