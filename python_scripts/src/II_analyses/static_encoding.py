import numpy as np

from useful_stuff.general_utils.II import InformationImbalance
from useful_stuff.general_utils.regression import linear_encoding


"""
validate_static_spaces
Validate two feature-by-sample matrices used for static encoding or comparison.

INPUT:
    - source: np.ndarray -> source space with shape (features, samples)
    - target: np.ndarray -> target space with shape (features, samples)

OUTPUT:
    - source: np.ndarray -> validated floating-point source matrix
    - target: np.ndarray -> validated floating-point target matrix
"""
def validate_static_spaces(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    # Static encoding treats columns as matched image samples.
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError(
            "Static spaces must be 2D arrays with shape (features, samples)."
        )
    # end if source.ndim != 2 or target.ndim != 2
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            "Source and target spaces must contain the same number of samples: "
            f"{source.shape[1]} and {target.shape[1]}."
        )
    # end if source.shape[1] != target.shape[1]
    if source.shape[1] < 2:
        raise ValueError("At least two matched samples are required.")
    # end if source.shape[1] < 2
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Source and target spaces must contain only finite values.")
    # end if not np.isfinite(source).all() or not np.isfinite(target).all()
    return source, target
# EOF


"""
compute_participation_ratio
Compute the effective dimensionality of a centered static response space from
its covariance eigenvalue spectrum.

INPUT:
    - space: np.ndarray -> response space with shape (features, samples)

OUTPUT:
    - participation_ratio: float -> covariance-spectrum participation ratio
"""
def compute_participation_ratio(space: np.ndarray) -> float:
    space = np.asarray(space, dtype=float)

    # Images are samples, so center every feature across image presentations.
    if space.ndim != 2:
        raise ValueError(
            "Participation ratio requires a 2D array with shape "
            "(features, samples)."
        )
    # end if space.ndim != 2
    if space.shape[1] < 2:
        raise ValueError("Participation ratio requires at least two samples.")
    # end if space.shape[1] < 2
    if not np.isfinite(space).all():
        raise ValueError(
            "Participation ratio requires only finite response values."
        )
    # end if not np.isfinite(space).all()

    centered_space = space - np.mean(space, axis=1, keepdims=True)
    singular_values = np.linalg.svd(centered_space, compute_uv=False)
    covariance_eigenvalues = singular_values**2 / (space.shape[1] - 1)
    denominator = np.sum(covariance_eigenvalues**2)

    # A constant response occupies no dimensions.
    if denominator == 0:
        return 0.0
    # end if denominator == 0
    participation_ratio = (
        np.sum(covariance_eigenvalues) ** 2 / denominator
    )
    return float(participation_ratio)
# EOF


"""
fit_static_projection
Fit one static linear projection and return predictions and per-output scores.
When cross-validation is enabled, predictions are assembled out of fold before
scoring. The encoder is then refit on all samples so its final weights describe
the full-data projection.

INPUT:
    - source: np.ndarray -> source space with shape (features, samples)
    - target: np.ndarray -> target space with shape (output_features, samples)
    - encoding: linear_encoding -> configured static encoding object
    - use_cv: bool -> whether to produce out-of-fold predictions

OUTPUT:
    - prediction: np.ndarray -> predicted target space, shaped like target
    - scores: np.ndarray -> score for every target feature
"""
def fit_static_projection(
    source: np.ndarray,
    target: np.ndarray,
    encoding: linear_encoding,
    use_cv: bool = False,
):
    source, target = validate_static_spaces(source, target)

    if use_cv:
        prediction = np.full(target.shape, np.nan, dtype=float)

        # Each test fold is predicted by a projection fitted without those samples.
        for train_idx, test_idx in encoding.get_cv_obj().split(source.T):
            encoding.fit(source[:, train_idx], target[:, train_idx])
            prediction[:, test_idx] = encoding.predict(source[:, test_idx])
        # end for train_idx, test_idx in encoding.get_cv_obj().split(source.T)

        if np.isnan(prediction).any():
            raise RuntimeError(
                "Cross-validation did not generate a prediction for every sample."
            )
        # end if np.isnan(prediction).any()

        # Score the complete out-of-fold prediction against its matched targets.
        scores = encoding.score(source, target, y_hat=prediction)

        # Retain a full-data fit so weights and intercepts remain inspectable.
        encoding.fit(source, target)
    else:
        # The default analysis fits and evaluates the projection on all images.
        encoding.fit(source, target)
        prediction = encoding.predict(source)
        scores = encoding.score(source, target, y_hat=prediction)
    # end if use_cv
    return prediction, scores
# EOF


"""
compute_static_prediction_II
Compute static Information Imbalance between predicted neural and model spaces.

INPUT:
    - predicted_neural: np.ndarray -> predicted neural space (features, samples)
    - predicted_model: np.ndarray -> predicted model space (features, samples)
    - neural_RDM_metric: str -> distance metric for predicted neural responses
    - model_RDM_metric: str -> distance metric for predicted model features
    - k: int -> number of nearest neighbors used by Information Imbalance

OUTPUT:
    - ii_obj: InformationImbalance -> fully computed static II object
    - neural_to_model_II: float -> II from predicted neural to predicted model
    - model_to_neural_II: float -> II from predicted model to predicted neural
"""
def compute_static_prediction_II(
    predicted_neural: np.ndarray,
    predicted_model: np.ndarray,
    neural_RDM_metric: str,
    model_RDM_metric: str,
    k: int = 1,
):
    predicted_neural, predicted_model = validate_static_spaces(
        predicted_neural, predicted_model
    )
    if k < 1 or k >= predicted_neural.shape[1]:
        raise ValueError(
            f"k must be between 1 and n_samples - 1; received {k}."
        )
    # end if k < 1 or k >= predicted_neural.shape[1]

    # The static II class labels predicted neural as signal and model as model.
    ii_obj = InformationImbalance(neural_RDM_metric, model_RDM_metric, k=k)
    ii_obj.compute_both_RDMs(predicted_neural, predicted_model)
    ii_obj.compute_both_distance_ranks()
    neural_to_model_II, model_to_neural_II = ii_obj.compute_both_II()
    return ii_obj, neural_to_model_II, model_to_neural_II
# EOF
