PD Speech Classification — LOSO + SMOTE + Multi-view (with full visualizations)

Short description
This repository contains code to run a reproducible analysis pipeline for Parkinson’s disease (PD) classification from speech features using a Leave-One-Subject-Out (LOSO) cross-validation scheme. The script implements multi-view modeling (original features, PCA view, optional LDA view), imbalance handling with SMOTE (or class weights), ensemble soft-voting across base classifiers, feature importance analysis and a comprehensive set of visualizations (t-SNE, PCA, LDA, ROC, PR, confusion matrices, feature importance).

Table of contents
Project overview
Requirements
Repository structure
Input data expectations
Quick start — run the script
Key configuration options
What the script does (step-by-step)
Outputs and filenames
Functions & modules explained
Tips, caveats and reproducibility
License
Project overview

This project was developed to explore and compare machine-learning approaches for early PD detection from voice/speech features. The intent is to evaluate realistic subject-level generalization using LOSO, mitigate label imbalance (SMOTE or class weighting), and combine information from multiple “views” of the feature space (original, PCA, optional LDA) using an ensemble of standard classifiers (Logistic Regression, SVM, Random Forest).

Primary goals:

Subject-wise generalization assessment (LOSO).

Robust treatment of class imbalance.

Multi-view ensembling to combine complementary feature representations.

Produce publication-quality diagnostic plots and an interpretable feature importance ranking.

Requirements

Recommended Python environment (tested with Python 3.9+). Install with:

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt


Example requirements.txt (create this file or install packages manually):

numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
tqdm
imblearn


Note: The script forces matplotlib.use('Agg') so it runs in headless environments (servers, CI) without a GUI.

Repository structure (suggested)
/.
├── README.md
├── requirements.txt
├── run_loso_pd.py          # the main script (the code you posted)
├── UNIFIED_RECONSTRUCTED.csv   # expected input CSV (not included)
├── outputs/                # generated images & logs (auto-created)
└── results/                # saved numeric outputs (optional)

Input data expectations

A CSV file (default path configured in the script):
INPUT_PATH = r"C:\Users\luesp\Downloads\parkinson disease article\UNIFIED_RECONSTRUCTED.csv"

The script will automatically locate the label/target column and the group/subject id column by scanning for common names:

Target candidates: ["label","class","target","y","pd","diagnosis"]

Group candidates: ["subject_id","subject","id","speaker","patient_id","name","recording","filename"]

The code expects the target to be binary and converts it to int. If your label column is named differently, either rename it in the CSV or change TARGET_CANDIDATES in the script.

Non-numeric columns will be attempted to convert to numeric; if a column is largely non-numeric it will be treated as categorical and removed from feature set (unless it can be encoded).

The script drops a hardcoded list of metadata/label columns (to avoid leakage). Review meta_and_label_cols_to_drop inside load_data() and adapt if needed.

Quick start — run the script

Place the dataset CSV at the path set by INPUT_PATH or edit the path in the script.

(Optional) edit configuration constants at the top of the script (see next section).

Run:

python run_loso_pd.py


Outputs (plots and logs) will be saved to the working directory as PNG files (see Outputs and filenames).

Key configuration options (top of the script)

Edit the following variables to change behavior:

INPUT_PATH — path to CSV.

RANDOM_STATE — reproducibility seed.

BALANCE_MODE — "smote" (default) or "weights" or other.

"smote" uses SMOTE oversampling on training folds.

"weights" uses class_weight="balanced" in classifiers.

SMOTE_RATIO — ratio of minority to majority after SMOTE.

MAX_SMOTE_K — maximum k neighbors for SMOTE (safe fallback is implemented).

USE_ORIGINAL_VIEW, USE_PCA_VIEW, USE_LDA_VIEW — enable/disable views.

PLOT_TSNE, PLOT_PCA, PLOT_LDA, PLOT_FEATURE_IMPORTANCE, PLOT_CONFUSION_MATRIX, PLOT_ROC_CURVES — toggles for plots.

LIMIT_SUBJECTS — integer to run on a subset of subjects (useful for debugging).

N_JOBS — parallel jobs for LOSO (-1 uses all cores).

VERBOSE_EVERY — how frequently the script prints fold progress.

What the script does (overview)

Load & clean data

Reads CSV, removes rows with missing labels, detects target and group columns, converts suitable object columns to numeric, removes all-NaN columns and obvious leakage columns.

Exploratory visualizations

Imputes & scales features for visualization then saves: t-SNE, PCA scatter + PCA explained variance, and optional LDA.

LOSO cross-validation (parallel)

Uses LeaveOneGroupOut where each unique subject is left out as the test fold.

For each fold:

Preprocess numeric and low-cardinality categorical columns with median imputation + scaling and one-hot encoding respectively.

Variance threshold feature selection (default threshold = 0.01).

Create multiple “views”:

Original features (possibly SMOTE balanced)

PCA view (reduced to explain large part of variance)

LDA view (optional)

For each view, trains base classifiers (Logistic Regression, SVM, Random Forest) and obtains predicted probabilities on the test set.

Soft-vote across base classifiers in each view, then soft-vote across views to produce final probability for each sample.

Parallelizes fold processing using joblib.Parallel.

Evaluation

Collects predictions across folds and computes accuracy, weighted F1, ROC AUC, recalls per class.

Produces confusion matrix PNG, ROC & precision-recall curves.

Feature importance

Trains RandomForest on entire dataset (after preprocessing) to produce a feature importance ranking and saves a bar plot.

Outputs and filenames

Example output files created in the working directory:

tsne_t-sne_do_dataset_completo.png — t-SNE visualization (sampled if >1000 rows).

pca_pca_do_dataset_completo.png — PCA scatter.

pca_explained_variance.png — cumulative explained variance by PCA.

lda_lda_do_dataset_completo.png — LDA scatter (if enabled).

confusion_matrix_matriz_de_confusao_–_loso.png — LOSO confusion matrix.

roc_curve_curva_roc_–_loso.png — ROC curve for the LOSO ensemble.

precision_recall_curve_curva_precisao-recall_–_loso.png — Precision-Recall curve.

feature_importance_feature_importance.png — top-N feature importances.

Tip: redirect stdout/stderr to a log file to retain printed fold progress and final evaluation summary.

Functions & modules explained (quick reference)

load_data() — read CSV, detect columns, clean and return DataFrame plus target/group column names.

choose_column(cands, cols) — helper to find best match among candidate names.

safe_smote_fit_resample(X, y) — robust SMOTE wrapper with safe fallback for tiny minority counts.

probs_softvote(list_of_prob_arrays, weights=None) — average (or weighted average) of probability arrays.

evaluate(y_true, y_prob, threshold=0.5, label="Model") — prints and returns basic metrics.

plot_* — plotting helpers that save PNGs (confusion matrix, ROC, PR, t-SNE, PCA, LDA, feature importance).

make_base_models(class_weight=None) — returns dictionary of base classifiers used in the ensemble.

process_fold(fold_data) — main per-fold processing pipeline that trains models and returns test labels and predicted probabilities for that fold.

run_loso_parallel(df, y_col, g_col) — builds fold list and dispatches process_fold in parallel, then collects results.

analyze_feature_importance(df, y_col, g_col) — trains RF on full processed data and saves importance plot.

Tips, caveats and reproducibility

Avoid data leakage: the script removes columns flagged as leak-prone. If you have columns that directly encode labels or are derived from the label (e.g., train/test flags), add them to meta_and_label_cols_to_drop.

SMOTE is applied only on the training fold. LOSO ensures the test subject is unseen during training and balancing.

Small minority class: safe_smote_fit_resample() reduces the SMOTE k parameter when the minority class is very small; if the minority class has fewer than 2 samples in a fold, SMOTE is skipped.

Parallelism: N_JOBS = -1 uses all cores; adjust if you hit memory limits.

Large datasets: t-SNE is expensive; the script samples at most 1000 examples for t-SNE by default.

Reproducibility: set RANDOM_STATE to a fixed integer.

Troubleshooting

If the script cannot find label or subject columns, set TARGET_CANDIDATES/GROUP_CANDIDATES manually or rename columns in the CSV.

If OOM occurs during parallel execution, reduce N_JOBS or set LIMIT_SUBJECTS for debug runs.

If an object column should be used as a feature but was dropped, check the automatic conversion in load_data() and move that column into numeric form (e.g., map categories to integers) before running.
