# Representational margins for repeated-trial decoding

## What the quantities measure

The analysis uses exactly the centroids and squared Euclidean distances used by the existing leave-one-repetition-out nearest-centroid decoder. For a held-out population vector, `d_correct` is its squared distance to the centroid estimated from the other repetitions of the same image. `d_second` is its distance to the closest centroid belonging to another image.

The signed nearest-centroid margin is

```text
d_second - d_correct
```

A positive value means that the held-out response lies inside the correct class's nearest-centroid decision region. Its magnitude is the clearance, in squared-distance units, from the closest decision competitor. A negative value means that an incorrect centroid is closer; larger negative magnitudes indicate more decisive errors. The sign therefore contains the accuracy decision, while the magnitude retains information that accuracy discards.

The normalized margin is

```text
(d_second - d_correct) / (d_second + d_correct)
```

It is bounded by `[-1, 1]` for non-negative distances and is unchanged if all distances for a trial are multiplied by the same positive constant. This makes comparisons less sensitive to the very different scales induced by raw, centered, and unit-normalized population vectors. When both distances are numerically zero, the implementation returns zero because the relative separation is undefined and neither centroid has clearance.

The correct-class rank is one plus the number of centroids strictly closer than the correct centroid. Rank one is a nearest-centroid success, rank two says that only one competing class is closer, and so on. Rank is invariant to every monotonic transformation of distance, so it is useful when even normalized distance scales are difficult to compare. Tied centroids receive the best shared rank; exact ties are retained in the full distance vectors and should be inspected if they occur often.

The complete centroid-distance vector is stored for every fold, time bin, and held-out image. Unlike a single margin, it retains which classes compete and how the rest of the class geometry is arranged. It supports later class-by-class confusion profiles, distance-profile similarity, entropy-like summaries, and tests of whether two normalizations preserve the same competitor structure.

## Why margins can be more sensitive than accuracy

Accuracy changes only when a sample crosses a decision boundary. A normalization can move the correct response substantially closer to or farther from that boundary without changing which centroid is nearest. The margin records that movement continuously. Rank is coarser than a distance margin but still distinguishes near misses from errors in which many class centroids are closer.

This sensitivity is not automatically scientific evidence. Trial margins from the same session share training data, neurons, stimuli, and temporal autocorrelation. The notebook therefore averages images within each held-out repetition and uses repetitions as paired units for exploratory within-session Wilcoxon tests. Benjamini-Hochberg correction is applied across time bins for each normalization and metric comparison. Its explicit “accuracy identical, margin different” flag refers to equal mean accuracy at a time bin; the saved statistics also report whether every individual correct/incorrect outcome is identical. A manuscript-level population claim should repeat the analysis independently across sessions and use one paired session-level effect per normalization, ideally with monkey represented as a grouping factor when enough animals and sessions are available.

Correct and incorrect trials are also summarized separately. Under this signed definition, correct trials normally have non-negative margins and incorrect trials have non-positive margins. The separate curves should therefore be read mainly as confidence conditional on the outcome, not as two independent effects. Conditioning on correctness can also select different trials under different normalizations, so inferential comparisons use all paired trials rather than treating the outcome subsets as directly paired samples.

## Recommended manuscript measure

Use the normalized signed margin as the primary continuous decoder measure, accompanied by accuracy and the correct-class rank. It is the closest match to the scientific question—relative clearance of the correct class from its strongest competitor—while reducing the arbitrary scale dependence of the raw squared-distance margin. Report the raw margin only as a decoder-unit diagnostic because preprocessing deliberately changes distance scale. Rank is a valuable scale-free robustness analysis, but it discards within-rank changes and is therefore less sensitive than the normalized margin.

For inference, compute each session's mean normalized-margin difference between two normalizations in a pre-registered time window or with a time-resolved cluster procedure, then test those session-level paired differences. Avoid presenting fold-level p-values from a single session as evidence that generalizes across sessions or monkeys.

## What these margins do—and do not—say about geometry

The margins measure geometry only through the assumptions of a nearest-centroid decoder. They privilege compact, approximately spherical classes, use one centroid per image, weight all neural dimensions equally after preprocessing, and consider only the closest competitor. A normalization can improve this margin because it better matches those assumptions even if other aspects of the representational geometry become less stable or less biologically meaningful. Conversely, multimodal or anisotropic class structure can be informative but poorly summarized by a centroid.

There is an additional metric-specific detail: the existing decoder minimizes squared Euclidean distance. Unit L2 normalization makes this monotonic with cosine distance for individual vectors, but a centroid of unit vectors is generally not itself unit length. The procedure is therefore best described as nearest-centroid decoding after the stated population-vector transformation, rather than as a pure cosine nearest-centroid classifier.

Feature centering without normalization cannot change this decoder at all (apart from floating-point error). It subtracts one training-derived translation vector from every training and test response; the fitted centroids translate by exactly the same vector, so every Euclidean centroid distance is preserved. Consequently, `raw` and `feature_center` should have identical accuracy, margins, ranks, and full distance vectors. The relevant centered-cosine-like comparison is `feature_center_norm`, where the subsequent sample-wise normalization can change angles and centroid geometry. This invariance is a mathematical property of the pipeline, not a failure of margin sensitivity.

Two extensions are worth considering:

1. **Full competitor-profile analysis.** Compare the complete held-out centroid-distance vectors across normalizations, or aggregate them into class-by-class confusion-distance matrices. This reveals whether a normalization changes only the nearest competitor or reorganizes the broader confusion geometry. The current implementation already retains everything required, so this should be the next analysis and does not require rerunning decoding.

2. **Cross-validated probabilistic scores.** A temperature-calibrated softmax over negative centroid distances would provide held-out log loss or true-class probability and use all competitors. It should be implemented only if the temperature is fitted inside each training fold; otherwise changes in raw distance scale masquerade as confidence. It adds a calibration assumption, so it is a useful secondary analysis rather than a replacement for the normalized margin.

For a decoder-independent geometry analysis, compare cross-validated representational dissimilarity matrices or crossnobis distances across repetitions. Crossnobis can estimate unbiased separation under an explicit noise model, while RDM reliability assesses whether normalization changes reproducible pairwise structure. These analyses answer a broader question than decoder confidence and are preferable if the manuscript claim is about representational geometry itself rather than nearest-centroid separability.
