import numpy as np
from scipy import stats


PREPROCESSING_CONFIGS = {
    "raw": dict(normalize=False, feature_center=False, mean_center=False),
    "norm": dict(normalize=True, feature_center=False, mean_center=False),
    "feature_center": dict(normalize=False, feature_center=True, mean_center=False),
    "feature_center_norm": dict(
        normalize=True, feature_center=True, mean_center=False
    ),
    "mean_center": dict(normalize=False, feature_center=False, mean_center=True),
    "mean_center_norm": dict(
        normalize=True, feature_center=False, mean_center=True
    ),
}


"""
select_balanced_images
Selects image classes with enough repetitions for balanced cross-validation.

INPUT:
    - name_to_indices: dict -> trial indices grouped by image name
    - n_images: int -> maximum number of image classes to select
    - min_repetitions: int -> minimum number of trials required per image
    - random_state: int or None -> seed used to sample image classes
    - image_names: iterable or None -> optional fixed image-name ordering

OUTPUT:
    - selected: list -> selected image names
    - n_repetitions: int -> repetitions available for every selected image
"""
def select_balanced_images(
    name_to_indices,
    n_images,
    min_repetitions=2,
    random_state=None,
    image_names=None,
):
    rng = np.random.default_rng(random_state)
    eligible = np.array(
        [
            name
            for name, trial_indices in name_to_indices.items()
            if len(trial_indices) >= min_repetitions
        ],
        dtype=object,
    )
    if len(eligible) == 0:
        raise ValueError(f"No images have at least {min_repetitions} repetitions.")
    # end if len(eligible) == 0

    if image_names is None:
        n_selected = min(n_images, len(eligible))
        selected = rng.choice(
            eligible, size=n_selected, replace=False
        ).tolist()
    else:
        eligible_set = set(eligible)
        selected = [name for name in image_names if name in eligible_set]
        if not selected:
            raise ValueError(
                "None of image_names are eligible with enough repetitions."
            )
        # end if not selected
    # end if image_names is None

    n_repetitions = min(len(name_to_indices[name]) for name in selected)
    return selected, n_repetitions
# EOF


"""
make_repetition_tensor
Builds a balanced image-by-repetition neural response tensor.

INPUT:
    - rasters: np.ndarray -> neural data shaped (neurons, time, trials)
    - name_to_indices: dict -> trial indices grouped by image name
    - selected_names: iterable -> image classes to include
    - n_repetitions: int or None -> repetitions retained for every image
    - random_state: int or None -> seed used to shuffle repetitions

OUTPUT:
    - X_rep: np.ndarray -> data shaped (images, repetitions, neurons, time)
    - labels: np.ndarray -> integer image labels
    - selected_trial_indices: dict -> source trial indices retained per image
"""
def make_repetition_tensor(
    rasters,
    name_to_indices,
    selected_names,
    n_repetitions=None,
    random_state=None,
):
    rasters = np.asarray(rasters)
    if rasters.ndim != 3:
        raise ValueError(
            "rasters must have shape (neurons, time, trials), "
            f"got {rasters.shape}."
        )
    # end if rasters.ndim != 3

    selected_names = list(selected_names)
    if len(selected_names) == 0:
        raise ValueError("selected_names cannot be empty.")
    # end if len(selected_names) == 0

    rng = np.random.default_rng(random_state)
    if n_repetitions is None:
        n_repetitions = min(
            len(name_to_indices[name]) for name in selected_names
        )
    # end if n_repetitions is None
    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required for decoding.")
    # end if n_repetitions < 2

    image_trials = []
    selected_trial_indices = {}
    n_total_trials = rasters.shape[2]
    for name in selected_names:
        trial_indices = np.asarray(name_to_indices[name], dtype=int)
        if len(trial_indices) < n_repetitions:
            raise ValueError(
                f"Image {name!r} has {len(trial_indices)} trials, fewer than "
                f"the requested {n_repetitions}."
            )
        # end if len(trial_indices) < n_repetitions
        if np.any((trial_indices < 0) | (trial_indices >= n_total_trials)):
            raise IndexError(
                f"Trial indices for {name!r} exceed raster trial axis length "
                f"{n_total_trials}."
            )
        # end if invalid trial indices

        trial_indices = rng.permutation(trial_indices)[:n_repetitions]
        selected_trial_indices[name] = trial_indices.tolist()
        image_trials.append(rasters[:, :, trial_indices])
    # end for name in selected_names

    # Move the source trial axis next to image so folds index repetitions directly.
    X_rep = np.stack(image_trials, axis=0).transpose(0, 3, 1, 2)
    labels = np.arange(len(selected_names))
    return X_rep, labels, selected_trial_indices
# EOF


"""
fit_preprocess
Fits training-dependent population-vector preprocessing parameters.

INPUT:
    - X_train: np.ndarray -> training samples shaped (samples, features)
    - normalize: bool -> divide every population vector by its L2 norm
    - feature_center: bool -> subtract the training mean of every feature
    - mean_center: bool -> subtract each population vector's feature mean
    - eps: float -> lower bound used for stable L2 normalization

OUTPUT:
    - params: dict -> fitted preprocessing parameters
"""
def fit_preprocess(
    X_train,
    *,
    normalize=False,
    feature_center=False,
    mean_center=False,
    eps=1e-12,
):
    X_train = np.asarray(X_train, dtype=float)
    if X_train.ndim != 2:
        raise ValueError(
            f"X_train must be a samples-by-features array, got {X_train.shape}."
        )
    # end if X_train.ndim != 2

    params = {
        "normalize": normalize,
        "feature_center": feature_center,
        "mean_center": mean_center,
        "eps": eps,
        "feature_mean": None,
    }
    if feature_center:
        # Fit feature centering on training trials only to avoid test leakage.
        params["feature_mean"] = X_train.mean(axis=0, keepdims=True)
    # end if feature_center
    return params
# EOF


"""
transform_preprocess
Applies fitted population-vector preprocessing to samples.

INPUT:
    - X: np.ndarray -> samples shaped (samples, features)
    - params: dict -> output of fit_preprocess

OUTPUT:
    - X_processed: np.ndarray -> preprocessed samples
"""
def transform_preprocess(X, params):
    X_processed = np.asarray(X, dtype=float).copy()
    if X_processed.ndim != 2:
        raise ValueError(
            f"X must be a samples-by-features array, got {X_processed.shape}."
        )
    # end if X_processed.ndim != 2

    if params["feature_center"]:
        X_processed -= params["feature_mean"]
    # end if params["feature_center"]
    if params["mean_center"]:
        X_processed -= X_processed.mean(axis=1, keepdims=True)
    # end if params["mean_center"]
    if params["normalize"]:
        norms = np.linalg.norm(X_processed, axis=1, keepdims=True)
        X_processed /= np.maximum(norms, params["eps"])
    # end if params["normalize"]
    return X_processed
# EOF


"""
fit_nearest_centroid
Fits one mean population vector per class.

INPUT:
    - X_train: np.ndarray -> training samples shaped (samples, features)
    - y_train: np.ndarray -> class label for every training sample

OUTPUT:
    - model: dict -> class labels and centroid population vectors
"""
def fit_nearest_centroid(X_train, y_train):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    if X_train.ndim != 2 or y_train.ndim != 1:
        raise ValueError("X_train must be 2D and y_train must be 1D.")
    # end if invalid input dimensions
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must contain the same samples.")
    # end if sample counts differ

    classes = np.unique(y_train)
    centroids = np.stack(
        [X_train[y_train == class_label].mean(axis=0) for class_label in classes],
        axis=0,
    )
    return {"classes": classes, "centroids": centroids}
# EOF


"""
nearest_centroid_distances
Computes the squared Euclidean distances used by the existing decoder.

INPUT:
    - model: dict -> output of fit_nearest_centroid
    - X_test: np.ndarray -> held-out samples shaped (samples, features)

OUTPUT:
    - distances: np.ndarray -> squared distances shaped (samples, classes)
"""
def nearest_centroid_distances(model, X_test):
    X_test = np.asarray(X_test, dtype=float)
    if X_test.ndim != 2:
        raise ValueError(f"X_test must be 2D, got {X_test.shape}.")
    # end if X_test.ndim != 2

    differences = X_test[:, None, :] - model["centroids"][None, :, :]
    return np.sum(differences ** 2, axis=2)
# EOF


"""
predict_nearest_centroid
Predicts the class of every sample from its nearest centroid.

INPUT:
    - model: dict -> output of fit_nearest_centroid
    - X_test: np.ndarray -> held-out samples shaped (samples, features)
    - return_distances: bool -> also return the full centroid-distance matrix

OUTPUT:
    - predictions: np.ndarray -> predicted class labels
    - distances: np.ndarray, optional -> squared distance to every centroid
"""
def predict_nearest_centroid(model, X_test, return_distances=False):
    distances = nearest_centroid_distances(model, X_test)
    predictions = model["classes"][np.argmin(distances, axis=1)]
    if return_distances:
        return predictions, distances
    # end if return_distances
    return predictions
# EOF


"""
fit_classifier
Fits a supported classifier while preserving the notebook's prior interface.

INPUT:
    - X_train: np.ndarray -> training samples shaped (samples, features)
    - y_train: np.ndarray -> class labels
    - classifier: str -> "nearest_centroid" or "linear_svc"

OUTPUT:
    - model: object -> fitted classifier
"""
def fit_classifier(X_train, y_train, classifier="nearest_centroid"):
    if classifier == "nearest_centroid":
        return fit_nearest_centroid(X_train, y_train)
    # end if classifier == "nearest_centroid"
    if classifier == "linear_svc":
        from sklearn.svm import LinearSVC

        model = LinearSVC(C=1.0, dual="auto", max_iter=5000)
        model.fit(X_train, y_train)
        return model
    # end if classifier == "linear_svc"
    raise ValueError(f"Unknown classifier: {classifier}")
# EOF


"""
predict_classifier
Predicts held-out labels and optionally returns centroid distances.

INPUT:
    - model: object -> fitted classifier
    - X_test: np.ndarray -> held-out samples shaped (samples, features)
    - classifier: str -> "nearest_centroid" or "linear_svc"
    - return_distances: bool -> request distances for nearest-centroid decoding

OUTPUT:
    - predictions: np.ndarray -> predicted labels
    - distances: np.ndarray or None, optional -> full centroid distances
"""
def predict_classifier(
    model,
    X_test,
    classifier="nearest_centroid",
    return_distances=False,
):
    if classifier == "nearest_centroid":
        return predict_nearest_centroid(
            model, X_test, return_distances=return_distances
        )
    # end if classifier == "nearest_centroid"
    if classifier == "linear_svc":
        predictions = model.predict(X_test)
        if return_distances:
            return predictions, None
        # end if return_distances
        return predictions
    # end if classifier == "linear_svc"
    raise ValueError(f"Unknown classifier: {classifier}")
# EOF


"""
compute_repetition_decoder_outputs
Runs leave-one-repetition-out decoding and retains every held-out output.

INPUT:
    - X_rep: np.ndarray -> data shaped (images, repetitions, neurons, time)
    - preprocessing: dict -> preprocessing flags passed to fit_preprocess
    - classifier: str -> "nearest_centroid" or "linear_svc"
    - repetition_folds: iterable or None -> repetition indices held out

OUTPUT:
    - outputs: dict -> predictions, targets, folds, and full centroid distances
"""
def compute_repetition_decoder_outputs(
    X_rep,
    preprocessing,
    classifier="nearest_centroid",
    repetition_folds=None,
):
    X_rep = np.asarray(X_rep)
    if X_rep.ndim != 4:
        raise ValueError(
            "X_rep must have shape (images, repetitions, neurons, time), "
            f"got {X_rep.shape}."
        )
    # end if X_rep.ndim != 4

    n_images, n_repetitions, n_neurons, n_time = X_rep.shape
    if n_images < 2:
        raise ValueError("At least two image classes are required for margins.")
    # end if n_images < 2
    if n_repetitions < 2:
        raise ValueError("At least two repetitions are required for decoding.")
    # end if n_repetitions < 2

    labels_by_image = np.arange(n_images)
    if repetition_folds is None:
        repetition_folds = np.arange(n_repetitions, dtype=int)
    else:
        repetition_folds = np.asarray(list(repetition_folds), dtype=int)
    # end if repetition_folds is None
    if len(repetition_folds) == 0:
        raise ValueError("repetition_folds cannot be empty.")
    # end if len(repetition_folds) == 0
    if np.any((repetition_folds < 0) | (repetition_folds >= n_repetitions)):
        raise IndexError("repetition_folds contains an invalid repetition index.")
    # end if invalid repetition fold

    n_folds = len(repetition_folds)
    predictions = np.empty((n_folds, n_time, n_images), dtype=int)
    centroid_distances = None
    if classifier == "nearest_centroid":
        centroid_distances = np.empty(
            (n_folds, n_time, n_images, n_images), dtype=float
        )
    # end if classifier == "nearest_centroid"

    for fold_idx, held_out_rep in enumerate(repetition_folds):
        train_repetitions = [
            repetition
            for repetition in range(n_repetitions)
            if repetition != held_out_rep
        ]
        X_train_all = X_rep[:, train_repetitions, :, :]
        X_test_all = X_rep[:, held_out_rep, :, :]
        y_train = np.repeat(labels_by_image, len(train_repetitions))

        for time_idx in range(n_time):
            # Flatten image and repetition only; neurons remain the feature axis.
            X_train_time = X_train_all[:, :, :, time_idx].reshape(
                n_images * len(train_repetitions), n_neurons
            )
            X_test_time = X_test_all[:, :, time_idx]

            preprocess_params = fit_preprocess(
                X_train_time, **preprocessing
            )
            X_train_time = transform_preprocess(
                X_train_time, preprocess_params
            )
            X_test_time = transform_preprocess(X_test_time, preprocess_params)

            model = fit_classifier(
                X_train_time, y_train, classifier=classifier
            )
            time_predictions, time_distances = predict_classifier(
                model,
                X_test_time,
                classifier=classifier,
                return_distances=True,
            )
            predictions[fold_idx, time_idx] = time_predictions
            if centroid_distances is not None:
                centroid_distances[fold_idx, time_idx] = time_distances
            # end if centroid_distances is not None
        # end for time_idx
    # end for fold_idx

    outputs = {
        "predictions": predictions,
        "targets": labels_by_image,
        "fold_indices": repetition_folds,
        "classes": labels_by_image,
        "classifier": classifier,
    }
    if centroid_distances is not None:
        # These distances are intentionally retained for later confusion analyses.
        outputs["centroid_distances"] = centroid_distances
    # end if centroid_distances is not None
    return outputs
# EOF


"""
compute_centroid_trial_metrics
Derives signed margins and correct-class ranks from full centroid distances.

INPUT:
    - centroid_distances: np.ndarray -> distances shaped (..., samples, classes)
    - targets: np.ndarray -> correct label for every sample
    - classes: np.ndarray -> class labels matching the final distance axis
    - eps: float -> minimum meaningful normalized-margin denominator

OUTPUT:
    - trial_metrics: dict -> per-trial distances, margins, and ranks
"""
def compute_centroid_trial_metrics(
    centroid_distances,
    targets,
    classes,
    eps=1e-12,
):
    centroid_distances = np.asarray(centroid_distances, dtype=float)
    targets = np.asarray(targets)
    classes = np.asarray(classes)
    if centroid_distances.ndim < 2:
        raise ValueError("centroid_distances must have sample and class axes.")
    # end if centroid_distances.ndim < 2
    if centroid_distances.shape[-2] != len(targets):
        raise ValueError("The sample distance axis must match targets.")
    # end if sample count differs
    if centroid_distances.shape[-1] != len(classes):
        raise ValueError("The class distance axis must match classes.")
    # end if class count differs
    if len(classes) < 2:
        raise ValueError("At least two classes are required for a margin.")
    # end if len(classes) < 2

    class_lookup = {class_label: idx for idx, class_label in enumerate(classes)}
    try:
        target_class_indices = np.array(
            [class_lookup[target] for target in targets], dtype=int
        )
    except KeyError as error:
        raise ValueError(f"Target class {error.args[0]!r} is missing from classes.")
    # end try

    index_shape = (1,) * (centroid_distances.ndim - 2) + (-1, 1)
    target_indices = target_class_indices.reshape(index_shape)
    d_correct = np.take_along_axis(
        centroid_distances, target_indices, axis=-1
    )[..., 0]

    # Mask the true centroid so the minimum is the strongest competing class.
    incorrect_distances = centroid_distances.copy()
    np.put_along_axis(incorrect_distances, target_indices, np.inf, axis=-1)
    d_second = np.min(incorrect_distances, axis=-1)

    # The signed margin is positive exactly when the correct centroid wins.
    margin = d_second - d_correct
    margin_denominator = d_second + d_correct
    margin_norm = np.zeros_like(margin)
    np.divide(
        margin,
        margin_denominator,
        out=margin_norm,
        where=margin_denominator > eps,
    )

    # Rank one means no centroid is strictly closer than the correct centroid.
    correct_rank = 1 + np.sum(
        centroid_distances < d_correct[..., None], axis=-1
    )
    return {
        "d_correct": d_correct,
        "d_second": d_second,
        "margin": margin,
        "margin_norm": margin_norm,
        "correct_rank": correct_rank,
    }
# EOF


def _fold_time_mean(values, mask):
    """Average samples within every fold and time bin without empty warnings."""
    valid = mask & np.isfinite(values)
    counts = valid.sum(axis=2)
    sums = np.where(valid, values, 0.0).sum(axis=2)
    means = np.full(counts.shape, np.nan, dtype=float)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means
# EOF


def _time_summary(values, mask):
    """Summarize a fold-by-time-by-sample array at every time bin."""
    valid = mask & np.isfinite(values)
    distributions = np.where(valid, values, np.nan)
    n_time = values.shape[1]
    means = np.full(n_time, np.nan, dtype=float)
    medians = np.full(n_time, np.nan, dtype=float)
    counts = np.zeros(n_time, dtype=int)

    for time_idx in range(n_time):
        time_values = values[:, time_idx, :][valid[:, time_idx, :]]
        counts[time_idx] = len(time_values)
        if len(time_values) > 0:
            means[time_idx] = np.mean(time_values)
            medians[time_idx] = np.median(time_values)
        # end if len(time_values) > 0
    # end for time_idx

    fold_means = _fold_time_mean(values, valid)
    sem = np.full(n_time, np.nan, dtype=float)
    for time_idx in range(n_time):
        valid_fold_means = fold_means[:, time_idx]
        valid_fold_means = valid_fold_means[np.isfinite(valid_fold_means)]
        if len(valid_fold_means) > 1:
            sem[time_idx] = np.std(valid_fold_means, ddof=1) / np.sqrt(
                len(valid_fold_means)
            )
        elif len(valid_fold_means) == 1:
            sem[time_idx] = 0.0
        # end if valid fold count
    # end for time_idx

    return {
        "mean": means,
        "median": medians,
        "distribution": distributions,
        "fold_means": fold_means,
        "sem": sem,
        "n_trials": counts,
    }
# EOF


"""
summarize_decoder_outputs
Computes accuracy and continuous margin summaries from held-out outputs.

INPUT:
    - outputs: dict -> output of compute_repetition_decoder_outputs
    - eps: float -> stability threshold for normalized margins

OUTPUT:
    - summary: dict -> backwards-compatible accuracy and margin summaries
"""
def summarize_decoder_outputs(outputs, eps=1e-12):
    predictions = np.asarray(outputs["predictions"])
    targets = np.asarray(outputs["targets"])
    if predictions.ndim != 3 or predictions.shape[2] != len(targets):
        raise ValueError(
            "predictions must have shape (folds, time, samples) matching targets."
        )
    # end if prediction shape is invalid

    correct = predictions == targets[None, None, :]
    fold_accuracies = correct.mean(axis=2)
    n_folds = fold_accuracies.shape[0]
    sem_accuracy = (
        fold_accuracies.std(axis=0, ddof=1) / np.sqrt(n_folds)
        if n_folds > 1
        else np.zeros(fold_accuracies.shape[1])
    )
    summary = {
        "correct": correct,
        "fold_accuracies": fold_accuracies,
        "mean_accuracy": fold_accuracies.mean(axis=0),
        "std_accuracy": fold_accuracies.std(axis=0),
        "sem_accuracy": sem_accuracy,
        "chance": 1 / len(outputs["classes"]),
    }

    if "centroid_distances" not in outputs:
        # Linear-SVC accuracy remains available, but centroid geometry does not.
        return summary
    # end if no centroid_distances

    trial_metrics = compute_centroid_trial_metrics(
        outputs["centroid_distances"],
        targets,
        outputs["classes"],
        eps=eps,
    )
    summary.update(trial_metrics)
    summary["margin_distribution"] = trial_metrics["margin"]
    summary["margin_norm_distribution"] = trial_metrics["margin_norm"]

    outcome_masks = {
        "all": np.ones(correct.shape, dtype=bool),
        "correct": correct,
        "incorrect": ~correct,
    }
    outcome_summary = {}
    for outcome, outcome_mask in outcome_masks.items():
        outcome_values = {}
        for metric_name, metric_values in trial_metrics.items():
            metric_summary = _time_summary(metric_values, outcome_mask)
            outcome_values[f"mean_{metric_name}"] = metric_summary["mean"]
            outcome_values[f"median_{metric_name}"] = metric_summary["median"]
            outcome_values[
                f"{metric_name}_distribution"
            ] = metric_summary["distribution"]
            outcome_values[
                f"fold_mean_{metric_name}"
            ] = metric_summary["fold_means"]
            outcome_values[f"sem_{metric_name}"] = metric_summary["sem"]
            outcome_values[
                f"n_{metric_name}_trials"
            ] = metric_summary["n_trials"]
        # end for metric_name
        outcome_summary[outcome] = outcome_values
    # end for outcome

    summary["outcome_summary"] = outcome_summary
    summary.update(outcome_summary["all"])
    for outcome in ("correct", "incorrect"):
        for metric_name, values in outcome_summary[outcome].items():
            summary[f"{metric_name}_{outcome}"] = values
        # end for metric_name
    # end for outcome
    return summary
# EOF


"""
crossvalidate_decoding_over_repetitions
Runs decoding once, then adds summary metrics without discarding trial outputs.

INPUT:
    - X_rep: np.ndarray -> data shaped (images, repetitions, neurons, time)
    - preprocessing: dict -> preprocessing flags passed to fit_preprocess
    - classifier: str -> "nearest_centroid" or "linear_svc"
    - repetition_folds: iterable or None -> repetition indices held out
    - margin_eps: float -> stability threshold for normalized margins

OUTPUT:
    - result: dict -> decoder outputs plus backwards-compatible summaries
"""
def crossvalidate_decoding_over_repetitions(
    X_rep,
    preprocessing,
    classifier="nearest_centroid",
    repetition_folds=None,
    margin_eps=1e-12,
):
    outputs = compute_repetition_decoder_outputs(
        X_rep,
        preprocessing=preprocessing,
        classifier=classifier,
        repetition_folds=repetition_folds,
    )
    summary = summarize_decoder_outputs(outputs, eps=margin_eps)
    return {**outputs, **summary}
# EOF


def _false_discovery_rate(pvalues):
    """Apply Benjamini-Hochberg correction while retaining NaN positions."""
    pvalues = np.asarray(pvalues, dtype=float)
    qvalues = np.full(pvalues.shape, np.nan, dtype=float)
    valid_indices = np.flatnonzero(np.isfinite(pvalues))
    if len(valid_indices) == 0:
        return qvalues
    # end if no valid p-values

    valid_pvalues = pvalues[valid_indices]
    order = np.argsort(valid_pvalues)
    ranked_pvalues = valid_pvalues[order]
    ranks = np.arange(1, len(ranked_pvalues) + 1)
    adjusted = ranked_pvalues * len(ranked_pvalues) / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    qvalues[valid_indices[order]] = adjusted
    return qvalues
# EOF


def _paired_wilcoxon(reference, candidate):
    """Run a paired signed-rank test with stable handling of zero differences."""
    valid = np.isfinite(reference) & np.isfinite(candidate)
    reference = reference[valid]
    candidate = candidate[valid]
    differences = candidate - reference
    if len(differences) < 2:
        return np.nan, np.nan, len(differences)
    # end if too few pairs
    if np.allclose(differences, 0.0):
        return 0.0, 1.0, len(differences)
    # end if every difference is zero

    test = stats.wilcoxon(
        candidate,
        reference,
        alternative="two-sided",
        method="auto",
    )
    return float(test.statistic), float(test.pvalue), len(differences)
# EOF


"""
paired_normalization_statistics
Compares normalization timecourses using paired held-out repetition folds.

INPUT:
    - results: dict -> decoding result for every normalization
    - baseline: str -> reference normalization label
    - metrics: iterable -> summary metrics compared across folds
    - alpha: float -> FDR-corrected significance threshold

OUTPUT:
    - comparisons: dict -> paired effects, p-values, q-values, and flags
"""
def paired_normalization_statistics(
    results,
    baseline="raw",
    metrics=("accuracy", "margin", "margin_norm", "correct_rank"),
    alpha=0.05,
):
    if baseline not in results:
        raise KeyError(f"Baseline {baseline!r} is missing from results.")
    # end if baseline missing

    metric_keys = {
        "accuracy": "fold_accuracies",
        "margin": "fold_mean_margin",
        "margin_norm": "fold_mean_margin_norm",
        "correct_rank": "fold_mean_correct_rank",
    }
    unknown_metrics = set(metrics) - set(metric_keys)
    if unknown_metrics:
        raise ValueError(f"Unknown paired metrics: {sorted(unknown_metrics)}")
    # end if unknown_metrics

    reference = results[baseline]
    comparisons = {}
    for label, candidate in results.items():
        if label == baseline:
            continue
        # end if label == baseline
        if not np.array_equal(
            candidate["fold_indices"], reference["fold_indices"]
        ):
            raise ValueError(f"Fold indices do not match for {label!r}.")
        # end if folds do not match
        if not np.array_equal(candidate["targets"], reference["targets"]):
            raise ValueError(f"Targets do not match for {label!r}.")
        # end if targets do not match

        reference_mean_accuracy = reference.get(
            "mean_accuracy", np.mean(reference["fold_accuracies"], axis=0)
        )
        candidate_mean_accuracy = candidate.get(
            "mean_accuracy", np.mean(candidate["fold_accuracies"], axis=0)
        )
        # Match the displayed accuracy curves, while retaining exact trial agreement.
        accuracy_identical = np.isclose(
            candidate_mean_accuracy,
            reference_mean_accuracy,
            rtol=0.0,
            atol=1e-12,
        )
        accuracy_outcomes_identical = np.all(
            candidate["correct"] == reference["correct"], axis=(0, 2)
        )
        label_comparison = {
            "baseline": baseline,
            "accuracy_identical": accuracy_identical,
            "accuracy_outcomes_identical": accuracy_outcomes_identical,
            "metrics": {},
        }
        for metric in metrics:
            summary_key = metric_keys[metric]
            if summary_key not in candidate or summary_key not in reference:
                continue
            # end if summary key unavailable

            reference_values = np.asarray(reference[summary_key], dtype=float)
            candidate_values = np.asarray(candidate[summary_key], dtype=float)
            if reference_values.shape != candidate_values.shape:
                raise ValueError(
                    f"Paired {metric!r} shapes do not match for {label!r}."
                )
            # end if shapes do not match

            n_time = reference_values.shape[1]
            statistics = np.full(n_time, np.nan, dtype=float)
            pvalues = np.full(n_time, np.nan, dtype=float)
            n_pairs = np.zeros(n_time, dtype=int)
            mean_differences = np.nanmean(
                candidate_values - reference_values, axis=0
            )
            median_differences = np.nanmedian(
                candidate_values - reference_values, axis=0
            )
            for time_idx in range(n_time):
                statistic, pvalue, n_pair = _paired_wilcoxon(
                    reference_values[:, time_idx],
                    candidate_values[:, time_idx],
                )
                statistics[time_idx] = statistic
                pvalues[time_idx] = pvalue
                n_pairs[time_idx] = n_pair
            # end for time_idx

            qvalues = _false_discovery_rate(pvalues)
            label_comparison["metrics"][metric] = {
                "statistic": statistics,
                "pvalue": pvalues,
                "qvalue": qvalues,
                "mean_difference": mean_differences,
                "median_difference": median_differences,
                "n_pairs": n_pairs,
                "significant": qvalues < alpha,
            }
        # end for metric

        for margin_metric in ("margin", "margin_norm"):
            if margin_metric in label_comparison["metrics"]:
                label_comparison[
                    f"{margin_metric}_only_significant"
                ] = (
                    accuracy_identical
                    & label_comparison["metrics"][margin_metric]["significant"]
                )
            # end if margin metric available
        # end for margin_metric
        comparisons[label] = label_comparison
    # end for label
    return comparisons
# EOF


"""
find_margin_only_differences
Lists time bins where accuracy is identical but a margin differs significantly.

INPUT:
    - comparisons: dict -> output of paired_normalization_statistics
    - time_s: np.ndarray -> time value for every tested bin
    - metric: str -> "margin" or "margin_norm"

OUTPUT:
    - findings: list -> normalization, time, effect, and FDR q-value records
"""
def find_margin_only_differences(comparisons, time_s, metric="margin_norm"):
    if metric not in ("margin", "margin_norm"):
        raise ValueError("metric must be 'margin' or 'margin_norm'.")
    # end if invalid metric

    time_s = np.asarray(time_s, dtype=float)
    findings = []
    flag_key = f"{metric}_only_significant"
    for label, comparison in comparisons.items():
        if flag_key not in comparison:
            continue
        # end if metric unavailable
        metric_results = comparison["metrics"][metric]
        if len(metric_results["qvalue"]) != len(time_s):
            raise ValueError("time_s does not match the number of tested bins.")
        # end if time length differs

        for time_idx in np.flatnonzero(comparison[flag_key]):
            findings.append(
                {
                    "normalization": label,
                    "baseline": comparison["baseline"],
                    "time_idx": int(time_idx),
                    "time_s": float(time_s[time_idx]),
                    "metric": metric,
                    "mean_difference": float(
                        metric_results["mean_difference"][time_idx]
                    ),
                    "qvalue": float(metric_results["qvalue"][time_idx]),
                }
            )
        # end for time_idx
    # end for label
    return findings
# EOF
