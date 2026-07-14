# Metrics II: project handoff

Last inspected: 2026-07-14

This document is the context handoff for continuing the project from a different ChatGPT/Codex account. Read it together with `CODE_GENERATION_COMMANDMENTS.md` before changing code.

## Project in one paragraph

`metrics_II` is a scientific Python project for comparing the representational geometry of visual neural responses with representations from artificial neural networks (ANNs). The neural data are time-resolved population responses to natural images. The ANN data are static layer activations for the same images. The project aligns the image orders, builds representational dissimilarity matrices (RDMs), and studies their relationship over neural time and model depth using dynamic RSA (dRSA), directed Information Imbalance (II), metric-to-metric comparisons, layer-to-layer comparisons, and cross-validated linear encoding. Most reusable computation lives in `python_scripts/src`; experiment launchers and exploratory/visualization notebooks live in `python_scripts/scripts`.

## The central research questions

The project currently asks several related questions:

1. How well does each ANN layer's static image geometry match the monkey neural geometry at each timepoint?
2. Is the relationship directional? In other words, does the neighborhood structure in the neural representation predict the ANN distance ranks better than the reverse direction?
3. How much do conclusions depend on the RDM metric: Euclidean, cosine, centered cosine variants, or correlation distance?
4. How similar are different ANN layers to one another under different metrics?
5. Are results stable when computed on repeated random subsets of images?
6. Can static ANN features linearly encode dynamic neural activity?
7. What changes when neural population vectors are normalized or centered?
8. What temporal structure appears with delay embeddings, neural-to-neural auto-II, split-half reliability, decoding, and dimensionality measures?

## Mental model and data flow

```text
Natural-image dataset
    -> ANN model
    -> one feature matrix per layer: (features, images)

Monkey natraster MAT file
    -> float32, transpose, optional brain-area slice, resample
    -> TimeSeries: (neurons, timepoints, images/trials)

allimages/uniqueImage MAT file + ImageFolder filenames
    -> mapping index
    -> reorder ANN columns to match neural image order

Aligned ANN layer features + neural raster
    -> RDMs / distance ranks / regression
    -> dRSA, directed II, metric comparison, layer comparison, encoding
    -> compressed NPZ results
    -> visualization and exploratory notebooks
```

The basic shape convention is important:

- Static model features: `(features, images)`.
- Neural raster: `(neurons, timepoints, images_or_trials)`.
- MATLAB natraster files are transposed into that neural convention by `load_img_natraster`.
- `TimeSeries` comes from the separate `useful_stuff` repository and carries the sampling frequency.
- Most saved arrays use NumPy's default NPZ key, `arr_0`.

## Core scientific concepts

### RDMs and metrics

An RDM describes pairwise distances between image representations. The same underlying population vectors can produce different RDMs depending on preprocessing and metric. Metrics used throughout the project include `euclidean`, `cosine`, `cosine_cnt`, `correlation`, and some notebook experiments with additional centered-cosine variants.

The normalization experiments make these distinctions explicit:

- `normalize`: divide every population vector by its L2 norm; this removes vector magnitude.
- `feature_center`: center every feature across images; this is associated with `cosine_cnt` in the current code.
- `mean_center`: center each population vector across its features; this corresponds to the centering used by correlation distance.

### Static dRSA

For each neural timepoint, compute the neural RDM and compare it with a static ANN-layer RDM. The result is one similarity timecourse per model layer. The main implementation uses `useful_stuff.general_utils.RSA.dRSA`; a notebook also computes Spearman rather than the default Pearson-style RDM-vector comparison.

### Static-dynamic Information Imbalance

Information Imbalance compares neighborhood/rank information in two spaces and is directional.

- `A2B`: condition on nearest neighbors in A, then evaluate ranks in B.
- `B2A`: the reverse direction.
- In the static neural-model analyses, A is normally the neural signal and B is the ANN model representation.
- Lower raw II means less information imbalance. Several plotting notebooks display `1 - II`, so verify whether a plot shows raw imbalance or transformed similarity/informativeness before interpreting it.
- `k` controls the number of nearest neighbors used for conditioning.

### Metric comparisons

These analyses apply two distance metrics to the same neural population data, then compare the resulting geometries with either directed II or RSA. They are used to determine which metric retains information about another metric and whether the relationship changes over time.

### Layer-distance comparisons

These analyses compare every ordered pair of ANN layers, producing `n_layers x n_layers` directed-II matrices. Both full-dataset and random-subsample versions exist.

### Static linear encoding

Each ANN layer is used to predict neural responses at every timepoint. The current code supports linear regression, ridge, lasso, and elastic net; correlation or R-squared scoring; and same-data, leave-one-out, or k-fold cross-validation. The notebook warns that cross-validated R-squared is not defined for one held-out sample, so use k-fold CV for meaningful R-squared.

## Repository layout

The Git repository root is `/Users/tizianocausin/Desktop/metrics_II`. This handoff lives in its `python_scripts` directory.

```text
metrics_II/
├── config.yaml                  environment-specific paths
├── brain_areas.yaml             monkey-specific channel ranges
├── pyproject.toml               Python 3.14 project and dependencies
├── uv.lock                      locked Python environment
├── README.md                    currently empty
├── bash_scripts/                Dipsen Slurm launchers
└── python_scripts/
    ├── CODE_GENERATION_COMMANDMENTS.md
    ├── PROJECT_HANDOFF.md        this file
    ├── src/
    │   ├── II_analyses/
    │   │   ├── static_dyn.py
    │   │   └── metrics_comparison.py
    │   └── project_specific_utils/
    │       ├── dataloader.py
    │       └── feature_extraction.py
    └── scripts/
        ├── run_*.py              command-line/MPI computations
        └── *.ipynb               analysis, development, and visualization
```

## Important source modules

### `src/project_specific_utils/dataloader.py`

- `decode_matlab_strings`: decodes MATLAB v7.3 HDF5 char-array references.
- `load_img_natraster`: loads `<monkey>_natraster<date>.mat`, changes axes to `(neurons, time, trials)`, wraps it in `TimeSeries` at 1000 Hz, optionally selects a brain area, and optionally resamples.
- `BrainAreas`: loads named channel ranges from `brain_areas.yaml` and concatenates the selected slices.
- `map_image_order_from_ann_to_monkey`: maps `ImageFolder` filenames onto image names from `<monkey>_allimages<date>.mat` or `uniqueImage`.
- `rename_talia_dataset`: adapts legacy Talia stimulus filenames by inserting an underscore before the first number and removing spaces.

Image ordering must be treated as a correctness constraint. ANN features are extracted in deterministic `ImageFolder` order and then reordered by filename before neural-model analyses. The current mapping code sorts the unique monkey image names lexicographically and requires an exact filename match.

### `src/project_specific_utils/feature_extraction.py`

This is active, currently untracked work. It adds serial and MPI feature extraction for Hugging Face, timm, and torchvision models.

- `FeatureExtractionCfg`: dataclass containing dataset, model, package, image size, batch, pooling, dtype, layers, paths, and overwrite/cache options.
- Known Hugging Face aliases currently include DINOv3-L and I-JEPA ViT-H/14.
- Hugging Face models use `AutoImageProcessor`; other packages use the usual project transform.
- Intermediate layers are captured through the `useful_stuff.image_processing.computational_models.imgANN` wrapper.
- Activations are pooled by the wrapper, concatenated over batches, transposed to `(features, images)`, and saved one NPZ per layer.
- With MPI, rank 0 coordinates; worker ranks each load the model and dataset, and contiguous groups of layers are distributed among workers.
- `--prepare_only` downloads/prepares Hugging Face model and processor assets without running extraction.

### `src/II_analyses/static_dyn.py`

- Initializes neural RDM timecourses once for static dRSA or static-dynamic II.
- Computes and saves one dRSA or two directional-II timecourses per ANN layer.
- Implements random image-subsample static-dynamic II.
- Implements population-vector preprocessing and cross-validated static linear encoding.
- Reads per-layer feature files and writes result files under `paths["data_path"]`.

### `src/II_analyses/metrics_comparison.py`

- Compares two RDM metrics on the same data with II or RSA.
- Computes dynamic metric-comparison timecourses over a neural raster.
- Provides random-subsample variants.
- Compares all ANN layer pairs and writes directional matrices.
- Centralizes most metric/layer result filename conventions.

### The external `useful_stuff` repository

This project is not standalone. It appends `paths["useful_stuff_path"]` to `sys.path` and relies on another local repository, normally:

```text
/Users/tizianocausin/Desktop/useful_stuff/python_scripts/src/useful_stuff
```

The most important external APIs are:

- `TimeSeries` and `print_wise` from `general_utils.utils`.
- `InformationImbalance` and `dynInformationImbalance` from `general_utils.II`.
- `RSA` and `dRSA` from `general_utils.RSA`.
- `dyn_linear_encoding` from `general_utils.regression`.
- `imgANN` and `get_relevant_output_layers` from `image_processing.computational_models`.
- `master_workers_queue` and `parallel_setup` from `parallel.parallel_funcs`.

Before modifying these interactions, inspect the live `useful_stuff` implementations rather than guessing their API or array conventions.

## Command-line scripts

| Script | Purpose |
|---|---|
| `run_ann_feature_extraction.py` | Extract and save layer activations; active untracked work. |
| `run_static_dRSA.py` | Compare each static ANN layer RDM to the neural RDM timecourse. |
| `run_static_dynII.py` | Compute full-dataset A2B and B2A static-dynamic II per layer. |
| `run_static_dynII_subsampled.py` | Average static-dynamic II over random image subsets. |
| `run_static_linear_encoding.py` | Cross-validate ANN-to-neural linear encoding at every timepoint. |
| `run_metrics_comparison.py` | Dynamic directed-II comparison between neural distance metrics. |
| `run_metrics_comparison_subsampled.py` | Subsampled version of the metric-II comparison. |
| `run_RSA_metrics_comparison.py` | Dynamic RSA comparison between metric-defined neural RDMs. |
| `run_RSA_metrics_comparison_subsampled.py` | Subsampled metric-RSA comparison. |
| `run_layer_distance_comparison.py` | Directed II for all ANN layer pairs and metric pairs. |
| `run_layer_distance_comparison_subsampled.py` | Subsampled layer-pair comparison. |

Most computations use MPI and `master_workers_queue`. Conventionally, rank 0 is the coordinator and ranks 1..N are workers. Existing examples use five MPI ranks.

## Main notebooks

The notebooks fall into five groups.

### Primary result visualization

- `viz_static_dRSA.ipynb`: layerwise dRSA timecourses plus peak, onset-latency, and temporal-centroid analyses. It is currently modified for DINOv3-L experiments.
- `viz_static_dRSA_metric_difference.ipynb`: compare two metric choices layer by layer and plot their difference.
- `viz_static_dynII.ipynb` and `viz_static_dynII_subsampled.ipynb`: directional II across time and layer depth.
- `viz_static_linear_encoding.ipynb`: load and visualize encoding scores.
- `viz_layer_distance_comparison.ipynb`: visualize layer-by-layer directed-II matrices.
- `viz_metric_comparison_k.ipynb`: study metric-comparison timecourses as `k` changes and compare with metric RSA.
- `viz_graph_metrics_comparison.ipynb`: graph of directed relationships among distance metrics, including session aggregation and direction tests.
- `viz_graph_neural-model.ipynb` and `viz_graph_neural-model_subsampled-II.ipynb`: graph-based summaries of neural-model relationships.
- `figs_summary_metrics_II.ipynb`: combined figure-generation notebook for metric, dRSA, and dynII summaries.

### Delay-embedding and temporal-structure prototypes

- `static_dRSA_delay_embeddings.ipynb`: static dRSA after delay-embedding each neural trial; results remain in memory.
- `static_dynII_delay_embeddings.ipynb`: static-dynamic II after delay embedding, optionally with subsampling; results remain in memory.
- `auto_dynII_delay_embeddings.ipynb`: neural-to-neural directed II between every pair of timepoints, with lag summaries and optional delay embedding.

These are prototypes. If they become part of a production workflow, move reusable helpers into `src/` and add command-line scripts.

### Reliability, decoding, and normalization

- `self_consistency_dev.ipynb`: split-half dRSA and dynII reliability over repeated presentations and image subsets.
- `repetition_decoding_normalization_dev.ipynb`: image decoding across repeated trials under normalization/centering variants.
- `norm_angle_contributions.ipynb`: separate population magnitude and angular contributions to RDM structure.
- `effect_of_k.ipynb`: exploratory effect of nearest-neighbor count on II.
- `subsampling_blocks_dataset_dev.ipynb`: older development notebook for image-subset II.

### Neural-response characterization

- `participation_ratio_timecourse.ipynb`: PCA participation ratio of neural population activity over time.
- `most_exciting_images_from_natraster.ipynb`: display images producing the largest mean response in a configured time window.
- `viz_baby1_240816to26_responses.ipynb`: session/area response inspection; despite its filename, its current configuration has also been used for `frosty`.

### Historical development

- `static_dRSA_and_dynII_cluster_dev.ipynb`: older combined prototype with functions later moved into `src`; imports and paths are partly stale.

## Data, stimuli, models, and results

The code expects the following logical layout under the environment-specific paths in `config.yaml`:

```text
paths["data_path"]/
├── data/
│   ├── <monkey>_natraster<date>.mat
│   └── <monkey>_allimages<date>.mat
├── models/
│   └── <folder>_<model>_<img_size>_<layer>_features_<pooling>pool.npz
└── results/
    └── analysis result NPZ files

paths["livingstone_lab"]/
└── Stimuli/
    └── <folder_name>/               torchvision ImageFolder dataset
```

The stimulus directory must satisfy `ImageFolder`: images need to be inside class subdirectories, even when class labels are not analytically important.

Data, model weights/features, figures, and results are intentionally not in Git. The `.gitignore` excludes common image/PDF/model output types, and the large neural/stimulus data live outside the repository. A new machine or account therefore needs access to the data directories and model caches in addition to the Git repository.

## Environment configuration

`config.yaml` currently defines four environments:

- `tiziano_mac_mini`: primary local setup; includes `data_path` and local/NAS paths.
- `tiziano_local`: another local setup, but its `src_path` currently contains `metric_II` (singular) and it lacks `data_path`; verify before use.
- `o2_cluster`: O2/HMS paths; currently lacks `data_path`, although several computations require it.
- `dipsen_hpc`: Dipsen cluster paths with NAS-backed `data_path`.

Scripts select the block with:

```bash
export MY_ENV=tiziano_mac_mini
```

The default is normally `tiziano_mac_mini`. Paths are machine-specific and are not portable secrets/configuration; update or add an environment block on a new machine instead of hard-coding paths in scripts.

## Python environment and dependencies

The project uses `uv`, with `.venv` at the repository root. `pyproject.toml` currently pins Python exactly to 3.14 and includes NumPy/SciPy/scikit-learn, PyTorch/torchvision, timm, Transformers/Hugging Face, h5py, mpi4py, matplotlib, numba, UMAP, OpenCV, and Jupyter.

Typical setup from the repository root:

```bash
cd /Users/tizianocausin/Desktop/metrics_II
uv sync
export MY_ENV=tiziano_mac_mini
export MPLCONFIGDIR=/tmp/matplotlib
```

The strict Python 3.14 pin and very recent scientific-package versions may make cluster installation difficult. Keep `uv.lock` synchronized with `pyproject.toml` and avoid adding dependencies unless necessary.

## Typical workflows

Run scripts from `python_scripts/scripts` unless and until all remaining cwd-relative paths are fixed.

### 1. Extract ANN layer features

Serial DINOv3-L example:

```bash
cd /Users/tizianocausin/Desktop/metrics_II/python_scripts/scripts
MY_ENV=tiziano_mac_mini ../../.venv/bin/python run_ann_feature_extraction.py \
  --folder_name talia_20each_tizi \
  --model_name dino_v3_l \
  --pkg hf \
  --img_size 224 \
  --batch_size 8 \
  --pooling mean
```

Use `--layers layer.path ...` if `get_relevant_output_layers` does not yet know the model. Use `--prepare_only` while network access is available to populate the Hugging Face cache before offline cluster runs. Use `--overwrite` only deliberately.

### 2. Run static dRSA

```bash
mpiexec -np 5 ../../.venv/bin/python run_static_dRSA.py \
  --monkey_name three0 --date 250313 --brain_area AIT \
  --folder_name talia_20each_tizi \
  --signal_RDM_metric cosine_cnt --model_RDM_metric cosine_cnt \
  --model_name vit_l_16 --img_size 384 --pooling mean \
  --new_fs 100 --pkg timm
```

### 3. Run static-dynamic II

```bash
mpiexec -np 5 ../../.venv/bin/python run_static_dynII.py \
  --monkey_name three0 --date 250313 --brain_area AIT \
  --folder_name talia_20each_tizi \
  --signal_RDM_metric cosine_cnt --model_RDM_metric cosine_cnt \
  --model_name vit_l_16 --img_size 384 --pooling mean \
  --new_fs 100 --pkg timm --k 10
```

The subsampled script adds `--subsamples_size`, `--n_iterations`, and `--random_seed`.

### 4. Run metric or layer comparisons

```bash
mpiexec -np 5 ../../.venv/bin/python run_RSA_metrics_comparison.py \
  --monkey_name three0 --date 250313 --brain_area AIT --new_fs 100 \
  --metrics cosine_cnt cosine correlation euclidean

mpiexec -np 5 ../../.venv/bin/python run_layer_distance_comparison.py \
  --folder_name talia_20each_tizi --model_name vit_l_16 \
  --img_size 384 --pooling mean --pkg timm --k 1 \
  --metrics cosine correlation euclidean
```

### 5. Run static linear encoding

```bash
mpiexec -np 5 ../../.venv/bin/python run_static_linear_encoding.py \
  --monkey_name three0 --date 250313 --brain_area AIT \
  --folder_name talia_20each_tizi --model_name vit_l_16 \
  --img_size 384 --pooling mean --new_fs 100 --pkg timm \
  --regression_type ridge --score_type corr --cv_type loo \
  --alpha_min 1e-6 --alpha_max 1e3 --n_alphas 10 \
  --normalize --feature_center
```

### Cluster execution

The three Dipsen Slurm scripts launch static dRSA, static dynII, and subsampled static dynII. They request one node, five MPI tasks, 50 GB, and 12 hours; activate the repository virtual environment; set `MY_ENV=dipsen_hpc`; restrict BLAS/OpenMP threads; disable HDF5 locking; and run Hugging Face/Transformers offline. They expect experiment parameters to be supplied as exported Slurm environment variables.

## Output filename conventions

Names encode most experimental parameters so that notebooks can reconstruct paths without metadata files.

- ANN features: `<folder>_<model>_<size>_<layer>_features_<pooling>pool.npz`
- Static dRSA: `static_dRSA_<signal_metric>-<model_metric>_<monkey>_<date>_<area>_<model>_<size>_<layer>_<fs>Hz.npz`
- Static dynII: `dynII_<A2B|B2A>_k<k>_<signal_metric>-<model_metric>_<monkey>_<date>_<area>_<model>_<size>_<layer>_<fs>Hz[...].npz`
- Metric II: `metric_comparison_k<k>_<from>-<to>_<monkey>_<date>_<area>_<fs>Hz[...].npz`
- Metric RSA: `RSA_metric_comparison_<metricA>-<metricB>_<monkey>_<date>_<area>_<fs>Hz[...].npz`
- Layer II: `layer_distance_comparison_<A2B|B2A>_k<k>_<metricA>-<metricB>_<folder>_<model>_<size>[...].npz`
- Encoding: `static_linear_encoding_<regression>_<score>_<cv>_alpha..._<preprocessing>_<monkey>_<date>_<area>_<model>_<size>_<layer>_<fs>Hz.npz`

`[...]` may include subsample size and iteration count. The current subsampled filenames do not include `random_seed`, so changing only the seed does not create a new result filename.

## Brain-area configuration

`brain_areas.yaml` contains channel ranges for `paul`, `three0`, `baby1`, `og`, `octavius`, `friday`, `baby5`, `red`, `frosty`, and `casper`. Available named regions vary by monkey and include V1, V2, V3, PIT, CIT, and AIT.

Two details need care:

- Ranges are consumed by normal Python slicing, whose end index is exclusive. Confirm that each YAML range was authored with that convention before changing mappings.
- Some brain areas consist of multiple disjoint ranges and are concatenated.

## Coding contract for future development

The intended project contract is:

1. Prefer readable, interpretable scientific code over clever abstractions.
2. Match the style in this project's `src/` and in `useful_stuff` before introducing a pattern.
3. Keep functions and control flow compact when possible.
4. Reuse `TimeSeries`, `print_wise`, RDM/RSA/II, plotting, model, and parallel utilities instead of duplicating them.
5. Resolve paths through `config.yaml`, `MY_ENV`, and `Path(__file__)`; do not add local absolute paths to source.
6. Use the established `function_name / explanation / INPUT / OUTPUT` documentation block for non-trivial functions.
7. Use concise `# end ...`, `# EOF`, and `# EOC` comments where nesting or file structure benefits from them.
8. Be modular without over-engineering.
9. Put reusable functions in the owning `src/` module; keep only genuinely one-off logic in scripts/notebooks.
10. Preserve repository structure and naming/save conventions.
11. Make array shapes, assumptions, and scientific transformations easy to inspect.
12. Avoid new dependencies unless necessary.
13. Surface any conflict between a request and these rules.
14. Put parameters in a dataclass configuration object where appropriate and expose executable-script parameters through `argparse` in the established style.

The checked-in `CODE_GENERATION_COMMANDMENTS.md` is the canonical local reference, but the latest handoff instructions supplied by the project owner also emphasize rules 9 and 14 above.

## Current Git and working state

As inspected on 2026-07-14:

- Branch: `main`, tracking `origin/main`.
- Tracked branch state: 0 commits ahead and 0 behind the locally known upstream.
- Latest commit: `0b13e0b` (`Added code generation commandments`, 2026-07-09).
- Modified tracked file: `scripts/viz_static_dRSA.ipynb`.
- Untracked files: `scripts/run_ann_feature_extraction.py` and `src/project_specific_utils/feature_extraction.py`.
- All Python files passed AST parsing.
- The core `II_analyses`, dataloader, and feature-extraction imports succeeded in the project virtual environment when `MY_ENV=tiziano_mac_mini` was set.

Do not discard these changes when changing accounts. The notebook modification switches the active dRSA visualization toward `dino_v3_l` at 224 px and adds/changes latency/centroid analysis, but the notebook currently contains at least one recorded `NameError` output (`annotate_spearman` not defined at that execution point). Re-run it top-to-bottom after resolving configuration and model-layer lookup.

## Known issues and portability risks

These are observations, not all confirmed bugs:

1. `README.md` is empty, which is why this handoff is currently the best project overview.
2. `config.yaml` is machine-specific. `tiziano_local.src_path` appears to contain a `metric_II`/`metrics_II` typo, and some environments lack the `data_path` required by analysis code.
3. `run_metrics_comparison.py` is older than the other launchers: it defaults to a nonexistent-looking `MY_ENV=dev` and opens `../../config.yaml` relative to the current working directory.
4. `BrainAreas` also opens `../../brain_areas.yaml` relative to the current working directory. This works from `python_scripts/scripts` but is fragile from elsewhere.
5. Some compute functions assume the results directory already exists, while the metric-comparison and feature-extraction code create output directories. A fresh data tree may therefore fail on the first save.
6. There is no automated test suite. Correctness is currently checked mainly through notebooks, saved-output inspection, parsing/imports, and scientific sanity checks.
7. The DINOv3 feature-extraction work uses `pkg=hf`, while the currently modified `viz_static_dRSA.ipynb` has recorded configuration with `model_name=dino_v3_l` and `pkg=timm`. Confirm the package used for layer lookup and filenames before trusting the notebook.
8. The model layer list is owned by `useful_stuff.get_relevant_output_layers`. New architectures may require an update there or an explicit `--layers` list.
9. Subsample result names omit the random seed, and most functions skip computation when the target file already exists. Record seeds externally or expand the filename convention before systematic seed comparisons.
10. Image matching is filename-sensitive. Missing names, duplicate naming variants, or different ImageFolder ordering can invalidate or stop alignment.
11. The Dipsen scripts force Hugging Face and Transformers offline, so required weights/processors must already be cached.
12. Importing the scientific stack in a restricted environment produced Matplotlib/font-cache warnings and a non-fatal MPI network-bind warning. Setting `MPLCONFIGDIR` to a writable location helps; MPI should be tested in the actual local or cluster runtime.
13. The local Git object database is unusually large: about 4.5 GiB of loose objects plus reported temporary garbage at inspection time, while tracked notebooks total about 29 MiB. Before transferring/cloning extensively, consider a careful Git housekeeping pass after protecting all uncommitted work.

## Suggested next development priorities

1. Protect the current three working-tree changes with a deliberate commit or backup.
2. Finish and validate DINOv3/I-JEPA feature extraction on a tiny image subset and a small explicit layer list, then on the full dataset.
3. Confirm DINOv3 layer-name registration in `useful_stuff`, package selection (`hf` versus `timm`), feature shapes, and exact downstream filenames.
4. Re-run `viz_static_dRSA.ipynb` top-to-bottom, clear stale/error outputs, and validate the new latency/centroid analyses.
5. Normalize config loading and path resolution across all scripts/modules, especially `run_metrics_comparison.py` and `BrainAreas`.
6. Add small tests for image-order mapping, brain-area slicing, save-name generation, preprocessing, subsampling reproducibility, and synthetic RDM/II shapes.
7. Decide which delay-embedding, auto-II, reliability, and normalization notebook helpers are mature enough to move into `src/`.
8. Add metadata to outputs, or a sidecar experiment manifest, if filename-only provenance becomes too fragile.
9. After saving all work, clean notebook outputs and inspect Git storage growth before running Git maintenance.

## What must move with the account or machine

A Git clone alone is insufficient. Preserve or recreate:

- This repository, including the current uncommitted files.
- The separate `useful_stuff` repository at a path configured in `config.yaml`.
- Neural MAT files and stimulus ImageFolder datasets.
- Pre-extracted ANN layer NPZ files and analysis-result NPZ files.
- Hugging Face/model caches needed for offline execution.
- The correct environment-specific path block in `config.yaml`.
- Python 3.14 plus the `uv` environment, or a deliberate dependency-porting plan.
- Cluster credentials/modules and MPI/Slurm runtime configuration, when relevant.

Do not put private data, credentials, or large neural/model outputs into this Markdown file or into Git.

## Recommended first prompt in the new ChatGPT account

Copy this after opening the repository in the new account:

> Read `python_scripts/PROJECT_HANDOFF.md` and `python_scripts/CODE_GENERATION_COMMANDMENTS.md` completely before acting. Then inspect `git status`, `config.yaml`, the relevant modules in `python_scripts/src`, and the matching APIs in the separate `useful_stuff` repository. Preserve all existing uncommitted work. Summarize your understanding of the requested change, the files you expect to touch, data-shape assumptions, and the verification you will run; then implement the task unless a genuinely consequential choice is missing.

## Quick orientation checklist for a future assistant

- What monkey, date/session, brain area, stimulus folder, model, package, and image size are in scope?
- Which RDM metrics and which RDM-to-RDM comparison metric are intended?
- Is the plotted quantity raw II or `1 - II`?
- What are the neural and model array shapes at every boundary?
- Has ANN image order been mapped to the monkey image order?
- Does the selected `MY_ENV` provide every required path?
- Are model features/weights already available, especially on an offline cluster?
- Is the run full-dataset or subsampled, and is its random seed recorded?
- Does the output filename uniquely identify the experiment?
- Is reusable logic going into `src/` rather than a notebook/script?
- Has existing output been protected from silent skip or accidental overwrite?
- Were parsing/imports, a small synthetic/smoke run, and final result shapes checked?

# EOF
