"""
ENSTA - Machine Learning - Practical session 5: feature selection (filter and wrapper methods)
Breast Cancer Wisconsin (Diagnostic), scikit-learn.

Run:  python Work_session_5_Feature_selection.py
Needs: numpy, pandas, scipy, matplotlib, seaborn, scikit-learn (recent version).
Figures are saved in ./figures, results are printed. Total run time: about 7-8 minutes
(set RUN_ROBUSTNESS = False in part D.1 to skip the extra checks and save about 4 minutes).
The written answers are in the comments (# ...) after each part.
"""
import matplotlib
matplotlib.use("Agg")          # figures are saved to ./figures instead of opening windows


# ==============================================================================
# MACHINE LEARNING – PRACTICAL SESSION 5
# ==============================================================================
# ==============================================================================
# FEATURE SELECTION: FILTER AND WRAPPER METHODS
# ==============================================================================
#
# Data: Breast Cancer Wisconsin (Diagnostic), `load_breast_cancer(as_frame=True)`: 569 observations, 30 numerical features, target 0 = malignant, 1 = benign.
# Classifier: `LogisticRegression(max_iter=1000)` unless stated otherwise.
#
# How the ground rules are applied in this notebook
#
# * One fixed seed (`SEED = 42`) everywhere, and one stratified 70 % / 30 % split reused in every part.
# * Every scaling step (and, from Part D, every selection step) is inside a scikit-learn `Pipeline`, so it is re-fitted on the training part of every fold and never sees test data.
# * Every choice (threshold, `k`, number of features) is made with `StratifiedKFold(5, shuffle=True, random_state=42)` on the training set only.
# * The test set is used to report final results. Two places show a test score before Part D because the exercise asks for it explicitly (the baseline in A.4 and the table of B.2); in both, the number is only displayed, never used to take a decision. The real test evaluation of every method is the table of D.3.
# * F1-score is computed for the positive class `1` (benign), the default of `f1_score`.

import itertools, time
from math import comb
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, binomtest

import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, RFECV, SelectKBest, SequentialFeatureSelector,
                                       VarianceThreshold, f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

print("scikit-learn", sklearn.__version__)

SEED = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)   # validation scheme for every choice
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 170)
pd.set_option("display.max_colwidth", 140)
pd.set_option("display.max_columns", 30)


def make_pipe(clf=None):
    """StandardScaler + classifier (default: LogisticRegression). The scaler is re-fitted on each training fold."""
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", clf if clf is not None else LogisticRegression(max_iter=1000))])


def cv_acc(model, X, y):
    """Mean and std of the 5-fold stratified cross-validated accuracy."""
    s = cross_val_score(model, X, y, cv=CV, scoring="accuracy")
    return s.mean(), s.std()


def smallest_k_within_1sd(ks, means, stds):
    """Smallest size whose accuracy is within one standard deviation of the best one."""
    means, stds = np.asarray(means), np.asarray(stds)
    best = int(np.argmax(means))
    cutoff = means[best] - stds[best]
    return int(min(k for k, m in zip(ks, means) if m >= cutoff))


import os
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)
_n_fig = [0]

def show():
    """Save the current figure in ./figures and close it (replaces plt.show())."""
    _n_fig[0] += 1
    path = os.path.join(FIG_DIR, f"figure_{_n_fig[0]:02d}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    print(f"[figure saved: {path}]")

def display(obj):
    """Print a DataFrame / Series (replaces IPython's display)."""
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


# ------------------------------------------------------------------------------
# ==============================================================================
# PART A. DATA INSPECTION AND BASELINE
# ==============================================================================
# ---- A.1 Dimensions, types, descriptive statistics, missing values, duplicates ----

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
X, y = df.drop(columns="target"), df["target"]
FEATURES = X.columns.tolist()

print("shape:", df.shape, "-> 569 observations, 30 features + target")
print("dtypes:", df.dtypes.astype(str).value_counts().to_dict())
print("missing values:", int(df.isna().sum().sum()), "| duplicated rows:", int(df.duplicated().sum()))
display(df.describe().T.round(3))


# ---- A.2 Distribution of the target and DummyClassifier baseline ----

counts = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(4.6, 3.2))
bars = ax.bar(["malignant (0)", "benign (1)"], counts.values, color=["#c0504d", "#4f81bd"])
for b, c in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, c + 4, f"{c} ({c / len(y):.1%})", ha="center")
ax.set_ylabel("number of observations"); ax.set_title("Distribution of the target"); ax.set_ylim(0, 420)
plt.tight_layout(); show()

dummy_full = DummyClassifier(strategy="most_frequent").fit(X, y)
print(f"DummyClassifier(most_frequent) accuracy on the full data: {dummy_full.score(X, y):.3f}")


# ---- A.3 Stratified 70 % / 30 % split and standardization (scaler fitted on the training set only) ----

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)

scaler = StandardScaler().fit(X_train)            # fitted on the training set only
X_train_s = pd.DataFrame(scaler.transform(X_train), columns=FEATURES, index=X_train.index)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=FEATURES, index=X_test.index)

print("train:", X_train.shape, "| test:", X_test.shape)
print("share of benign  train: %.3f  test: %.3f" % (y_train.mean(), y_test.mean()))


# ---- A.4 Reference model: logistic regression on all 30 features ----

dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

base = make_pipe()                                  # scaler + LogisticRegression
t0 = time.perf_counter(); base.fit(X_train, y_train); base_time = time.perf_counter() - t0

baseline_table = pd.DataFrame([
    dict(model="DummyClassifier (most_frequent)", n_features=0,
         train_acc=accuracy_score(y_train, dummy.predict(X_train)), test_acc=accuracy_score(y_test, dummy.predict(X_test)),
         test_f1=f1_score(y_test, dummy.predict(X_test)), train_time_s=np.nan),
    dict(model="LogisticRegression, all features", n_features=30,
         train_acc=base.score(X_train, y_train), test_acc=base.score(X_test, y_test),
         test_f1=f1_score(y_test, base.predict(X_test)), train_time_s=base_time),
]).set_index("model")
display(baseline_table.round(4))


# ---- A.5 Correlation matrix of the training features ----

corr_train = X_train.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_train, cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.2,
            cbar_kws={"shrink": 0.7, "label": "Pearson correlation"}, ax=ax)
ax.set_title("Correlation matrix of the 30 training features")
plt.tight_layout(); show()


def redundancy_groups(corr, thr):
    """Connected groups of features linked by |correlation| > thr."""
    names = corr.columns.tolist()
    parent = {n: n for n in names}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if abs(corr.loc[a, b]) > thr:
                parent[find(a)] = find(b)
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    return [g for g in groups.values() if len(g) > 1]


upper = np.triu(np.ones(corr_train.shape, dtype=bool), k=1)
for thr in (0.95, 0.90):
    n_pairs = int((corr_train.abs().values[upper] > thr).sum())
    print(f"\n|corr| > {thr}: {n_pairs} pairs out of {upper.sum()}")
    for g in redundancy_groups(corr_train, thr):
        print("  -", g)
REDUNDANT_GROUPS = redundancy_groups(corr_train, 0.95)


# Baseline row. 30 features, test accuracy 0.9883 (169 of 171 patients), test F1 0.9907, training time about 0.01 s (training accuracy 0.9874). The `DummyClassifier("most_frequent")` only reaches 0.627 (0.626 on the test set, F1 0.770), so the model really learns the task.
#
# Why this dataset is a good candidate for feature selection
# * Strong redundancy. 15 of the 435 pairs of features have |correlation| > 0.95 and 23 have |correlation| > 0.90. Mean radius, perimeter and area and the worst radius, perimeter and area form one block of 6 near-copies of the same measurement; the three "error" versions form a second block, and concavity / concave points form a third one (threshold 0.90).
# * Many features carry almost no information alone. `texture error`, `symmetry error`, `fractal dimension error` and `mean fractal dimension` have F-scores close to 0 (B.3), so a much smaller model may be enough.
# * A reference to compare with. A simple linear model already reaches 98.8 % (against 62.7 % for the dummy), so we can measure exactly what is lost or gained with fewer features; fewer features also give a model that is easier to interpret and cheaper to measure.


# ------------------------------------------------------------------------------
# ==============================================================================
# PART B. FILTER METHODS
# ==============================================================================
# ---- B.1 Variance threshold ----

var_train = X_train.var(ddof=0).sort_values()          # variance used by VarianceThreshold (ddof = 0)
display(var_train.to_frame("variance (unscaled)").T.round(6).T)

rows = []
for thr in (0, 0.01, 0.1):
    vt = VarianceThreshold(threshold=thr).fit(X_train)
    removed = [f for f, keep in zip(FEATURES, vt.get_support()) if not keep]
    rows.append(dict(threshold=thr, n_removed=len(removed), removed=", ".join(removed) if removed else "-"))
display(pd.DataFrame(rows).set_index("threshold"))

# the same filter after standardization: every variance is 1, so nothing can be removed
print("variances after standardization (min, max):", X_train_s.var(ddof=0).min().round(6), X_train_s.var(ddof=0).max().round(6))
for thr in (0, 0.01, 0.1):
    n_rem = int((~VarianceThreshold(threshold=thr).fit(X_train_s).get_support()).sum())
    print(f"  threshold {thr}: removed {n_rem} features after standardization")


# Result (unscaled training data). Threshold 0 removes 0 features (there is no constant feature), threshold 0.01 removes 16, threshold 0.1 removes 19.
#
# Why it is meaningless after standardization. `StandardScaler` gives every feature a variance of exactly 1 (checked above: minimum = maximum = 1), so no threshold below 1 can remove anything: the filter would only compare identical numbers.
#
# Why it is of limited use on this dataset, even before scaling. The variance depends on the *units* of a feature, not on its information. `worst area` has a variance of about 3·10⁵ and `fractal dimension error` about 8·10⁻⁶ simply because areas are measured in hundreds and fractal dimensions in hundredths. Thresholds 0.01 and 0.1 just delete every feature measured on a small scale (smoothness, compactness, symmetry, fractal dimension, ...) and keep all large-scale ones, including the six near-identical size features. Some deleted features are useful: `worst smoothness` (variance 0.0005) is selected by 6 of the 7 methods of D.3. The only safe use of this filter is threshold 0 (constant columns), and there are none here.


# ---- B.2 Removing correlated features ----

def remove_correlated(X, threshold, y):
    """Greedy correlation filter.
    For every pair of features with |Pearson correlation| > threshold (strongest pairs first), drop the feature of the
    pair that is LESS correlated with the target (so the more informative one is kept). Returns the list of kept columns."""
    X = pd.DataFrame(X).reset_index(drop=True)
    y = np.asarray(y)
    corr = X.corr().abs()
    target_corr = X.apply(lambda col: abs(np.corrcoef(col.values, y)[0, 1]))
    cols = list(X.columns)
    pairs = [(corr.loc[a, b], a, b) for i, a in enumerate(cols) for b in cols[i + 1:] if corr.loc[a, b] > threshold]
    pairs.sort(key=lambda t: -t[0])
    dropped = set()
    for _, a, b in pairs:
        if a in dropped or b in dropped:
            continue                                   # the pair is already broken
        dropped.add(a if target_corr[a] < target_corr[b] else b)
    return [c for c in cols if c not in dropped]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper around remove_correlated, so that the filter can live inside a Pipeline (fitted per fold)."""
    def __init__(self, threshold=0.9):
        self.threshold = threshold
    def fit(self, X, y):
        X = np.asarray(X)
        kept = remove_correlated(pd.DataFrame(X), self.threshold, y)
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[kept] = True
        return self
    def transform(self, X):
        return np.asarray(X)[:, self.support_]
    def get_support(self):
        return self.support_


rows = []
for thr in (0.95, 0.90, 0.80):
    kept = remove_correlated(X_train, thr, y_train)
    removed = [f for f in FEATURES if f not in kept]
    pipe = Pipeline([("scaler", StandardScaler()), ("sel", CorrelationFilter(thr)),
                     ("clf", LogisticRegression(max_iter=1000))])
    m, s = cv_acc(pipe, X_train.values, y_train.values)             # selection repeated inside every fold
    pipe.fit(X_train.values, y_train.values)
    rows.append(dict(threshold=thr, n_kept=len(kept), n_removed=len(removed), cv_acc=m, cv_std=s,
                     test_acc_display_only=pipe.score(X_test.values, y_test.values), removed=", ".join(removed)))
b2 = pd.DataFrame(rows).set_index("threshold")
display(b2.drop(columns="removed").round(4))
display(b2[["removed"]])

cv_all = cv_acc(make_pipe(), X_train.values, y_train.values)
print(f"reference, all 30 features: CV accuracy {cv_all[0]:.4f} +/- {cv_all[1]:.4f}")

best_row = max(rows, key=lambda r: (round(r["cv_acc"], 4), -r["n_kept"]))      # best CV accuracy, then fewest features
B2_THRESHOLD = best_row["threshold"]
print("threshold kept for the final comparison (best CV accuracy, ties -> fewer features):", B2_THRESHOLD,
      "->", best_row["n_kept"], "features")


# Result (selection repeated inside every fold; the test accuracy column is displayed only because the exercise asks for it and was not used to choose anything):
#
# | threshold | features kept | CV accuracy (training set) | test accuracy (display only) |
# |---|---|---|---|
# | none | 30 | 0.9824 ± 0.0189 | 0.9883 |
# | 0.95 | 23 | 0.9824 ± 0.0189 | 0.9825 |
# | 0.90 | 20 | 0.9824 ± 0.0189 | 0.9766 |
# | 0.80 | 12 | 0.9447 ± 0.0220 | 0.9357 |
#
# Is removing redundancy enough to improve the accuracy? No. Down to 20 features (threshold 0.90) the accuracy is the same as with 30: useful for simplicity, but not an improvement. At 0.80 the filter becomes harmful (CV accuracy falls by almost 4 points): the rule only compares features pairwise, so it chains deletions and removes eight of the ten "mean" features (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points) together with their informative neighbours. In addition, logistic regression with its L2 penalty already copes reasonably with correlated inputs, so there is little to gain from deleting copies. Threshold 0.90 (20 features; it ties with 0.95 on CV accuracy and keeps fewer features) is used in D.3.


# ---- B.3 Univariate scores and SelectKBest ----

F, pval = f_classif(X_train, y_train)
MI = mutual_info_classif(X_train, y_train, random_state=SEED)
scores = pd.DataFrame({"F_score": F, "p_value": pval, "mutual_info": MI}, index=FEATURES)
display(scores.sort_values("F_score", ascending=False).round(4).head(10))

fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
for ax, col, color, title in [(axes[0], "F_score", "#4f81bd", "ANOVA F-score"),
                              (axes[1], "mutual_info", "#9bbb59", "Mutual information")]:
    s = scores[col].sort_values()
    ax.barh(s.index, s.values, color=color)
    ax.set_title(title + " (training set)"); ax.tick_params(axis="y", labelsize=8)
plt.tight_layout(); show()

rho, _ = spearmanr(scores["F_score"], scores["mutual_info"])
top = lambda col, n=10: set(scores[col].sort_values(ascending=False).index[:n])
print(f"Spearman correlation between the two rankings: {rho:.3f}")
print(f"overlap of the two top-10 lists: {len(top('F_score') & top('mutual_info'))} features")
top10_F = scores["F_score"].sort_values(ascending=False).index[:10].tolist()
print("top-10 by F-score:", top10_F)
print("... of which belong to the radius/perimeter/area family:",
      [f for f in top10_F if any(w in f for w in ("radius", "perimeter", "area"))])

KS = [2, 5, 10, 15, 20, 25, 30]
score_funcs = {"F-score": f_classif, "Mutual information": partial(mutual_info_classif, random_state=SEED)}

kb_rows = []
for name, fn in score_funcs.items():
    for k in KS:
        pipe = Pipeline([("scaler", StandardScaler()), ("sel", SelectKBest(fn, k=k)),
                         ("clf", LogisticRegression(max_iter=1000))])
        m, s = cv_acc(pipe, X_train.values, y_train.values)
        kb_rows.append(dict(score=name, k=k, cv_mean=m, cv_std=s))
kb = pd.DataFrame(kb_rows)
display(kb.pivot(index="k", columns="score", values="cv_mean").round(4))

best_k = {}
fig, ax = plt.subplots(figsize=(7.5, 4.3))
for (name, color) in [("F-score", "#4f81bd"), ("Mutual information", "#9bbb59")]:
    d = kb[kb.score == name]
    ax.errorbar(d.k, d.cv_mean, yerr=d.cv_std, marker="o", capsize=3, label=name, color=color)
    best_k[name] = smallest_k_within_1sd(d.k, d.cv_mean, d.cv_std)
    ax.axvline(best_k[name], ls="--", lw=1, color=color)
ax.set_xlabel("k (number of selected features)"); ax.set_ylabel("5-fold CV accuracy (training set)")
ax.set_title("SelectKBest: accuracy against k (dashed: smallest k within 1 sd of the best)"); ax.legend()
plt.tight_layout(); show()
K_F, K_MI = best_k["F-score"], best_k["Mutual information"]
print("chosen k -> F-score:", K_F, "| mutual information:", K_MI)


# Comment on the two rankings. They are almost the same: Spearman correlation 0.966, and the two top-10 lists contain exactly the same 10 features (worst concave points, worst perimeter, worst radius, worst area, mean concave points, mean perimeter, mean radius, mean area, mean concavity, worst concavity). Both give almost zero to `texture error`, `symmetry error`, `fractal dimension error` and `mean fractal dimension`. Mutual information could capture non-linear dependences, but here the classes are separated mostly monotonically, so it brings nothing more than the F-test.
#
# Comment on accuracy against k (5-fold CV, selection inside the pipeline):
#
# | k | 2 | 5 | 10 | 15 | 20 | 25 | 30 |
# |---|---|---|---|---|---|---|---|
# | F-score | 0.935 | 0.952 | 0.952 | 0.960 | 0.987 | 0.982 | 0.982 |
# | mutual information | 0.937 | 0.955 | 0.952 | 0.960 | 0.980 | 0.982 | 0.982 |
#
# The accuracy is stuck around 0.95–0.96 up to k = 15 and jumps at k = 20. The smallest k within one standard deviation of the best is k = 20 for both scores (k = 15 is more than one standard deviation below the best).
#
# Why can a univariate filter select nearly identical features? Each feature is scored alone against the target. `radius`, `perimeter` and `area` of the same nucleus are almost the same measurement (correlation > 0.95), so they all get a high score together, and the filter cannot know that after keeping one of them the others add nothing. Six of the 10 best F-score features belong to this family. The slots of a small k are therefore filled with copies, which is why the accuracy stays flat until k = 20. The complementary features that really add information (`worst texture`, `worst smoothness`) have only a mediocre univariate score (ranks 16 and 18) and enter much later.


# ------------------------------------------------------------------------------
# ==============================================================================
# PART C. WRAPPER METHODS
# ==============================================================================
# All wrappers use the unscaled training matrix with `make_pipe()` as estimator (scaler + logistic regression), so the data are standardized inside each cross-validation fold; this is equivalent to using standardized training data without letting the validation folds influence the scaling.

Xtr, ytr = X_train.values, y_train.values
Xte, yte = X_test.values, y_test.values
fnames = lambda idx: [FEATURES[i] for i in sorted(idx)]


# ---- C.1 Forward selection ----

# C.1.1 scikit-learn
t0 = time.perf_counter()
sfs_fwd = SequentialFeatureSelector(make_pipe(), direction="forward", n_features_to_select=10,
                                    scoring="accuracy", cv=CV).fit(Xtr, ytr)
t_sfs_fwd = time.perf_counter() - t0
fwd_sk = np.flatnonzero(sfs_fwd.get_support())
print(f"SequentialFeatureSelector(forward, 10) - {t_sfs_fwd:.2f} s")
print(fnames(fwd_sk))

# C.1.2 own implementation
def forward_selection(X, y, estimator, cv, max_features=10, eps=0.001):
    """Greedy forward selection. At each step add the feature maximizing the cross-validated accuracy.
    Stops when max_features is reached or when the gain is lower than eps (eps=None: never stop on the gain)."""
    t0 = time.perf_counter()
    selected, remaining, history, n_models = [], list(range(X.shape[1])), [], 0
    stopped_on_gain = False
    while remaining and len(selected) < max_features:
        cand = [cross_val_score(estimator, X[:, selected + [j]], y, cv=cv, scoring="accuracy") for j in remaining]
        n_models += len(remaining) * cv.get_n_splits()
        means = [c.mean() for c in cand]
        b = int(np.argmax(means))                               # first maximum (same tie rule as scikit-learn)
        gain = means[b] - (history[-1]["cv_mean"] if history else 0.0)
        if eps is not None and history and gain < eps:
            stopped_on_gain = True
            break
        selected.append(remaining.pop(b))
        history.append(dict(n_features=len(selected), added=FEATURES[selected[-1]],
                            cv_mean=means[b], cv_std=cand[b].std(), gain=gain))
    return dict(selected=selected, history=pd.DataFrame(history), n_models=n_models,
                elapsed=time.perf_counter() - t0, stopped_on_gain=stopped_on_gain)


fwd_eps = forward_selection(Xtr, ytr, make_pipe(), CV, max_features=10, eps=0.001)   # stopping rule of the exercise
fwd_10 = forward_selection(Xtr, ytr, make_pipe(), CV, max_features=10, eps=None)      # always 10 features (to compare with sklearn)

display(fwd_10["history"].round(4).set_index("n_features"))
print("stopping rule (gain < 0.001):",
      f"stopped after {len(fwd_eps['selected'])} features" if fwd_eps["stopped_on_gain"] else "never triggered before 10 features")
N_FWD = len(fwd_eps["selected"])

# C.1.3 plot and comparison with scikit-learn
h = fwd_10["history"]
fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(h.n_features, h.cv_mean, yerr=h.cv_std, marker="o", capsize=3, color="#4f81bd")
if fwd_eps["stopped_on_gain"]:
    ax.axvline(N_FWD, ls="--", color="grey", label=f"stop rule (gain < 0.001): {N_FWD} features"); ax.legend()
ax.set_xlabel("number of selected features"); ax.set_ylabel("5-fold CV accuracy (training set)")
ax.set_title("Forward selection (own implementation)"); plt.tight_layout(); show()

same = set(fwd_10["selected"]) == set(fwd_sk)
print("own implementation selects the same 10 features as scikit-learn:", same)
if not same:
    print("  only own:", fnames(set(fwd_10["selected"]) - set(fwd_sk)), "| only sklearn:", fnames(set(fwd_sk) - set(fwd_10["selected"])))
print("features added, in order:", h.added.tolist())


# Comments. The two implementations select the same 10 features (mean radius, mean texture, mean perimeter, mean compactness, texture error, smoothness error, worst radius, worst texture, worst perimeter, worst smoothness). This is expected: same estimator, same folds and the same tie rule (the first best candidate wins), so the greedy path is identical.
#
# The CV accuracy rises quickly: 0.922 with 1 feature (`worst radius`), 0.967 with 2 (`worst smoothness`), 0.982 with 3 (`worst texture`), 0.985 with 4 (`mean radius`), and then stays on a plateau (0.985–0.990) up to 10 features. With the exercise's stopping rule (gain < ε = 0.001) the algorithm stops after 4 features (worst radius, worst smoothness, worst texture, mean radius), because the 5th feature brings a gain of 0. As the CV accuracy is an average over 398 patients, one more correct patient changes it by 0.0025, so ε = 0.001 means "stop as soon as no additional patient is gained". The cost of this rule is that it ignores the small gains that come later (+0.0025 at the 8th and the 10th feature).
#
# It is also instructive that the second feature chosen is `worst smoothness`, which has a poor univariate score (rank 18 in B.3): it is selected because it is *complementary* to `worst radius`, which a filter cannot see.


# ---- C.2 Backward elimination ----

# C.2.1 scikit-learn
t0 = time.perf_counter()
sfs_bwd = SequentialFeatureSelector(make_pipe(), direction="backward", n_features_to_select=10,
                                    scoring="accuracy", cv=CV).fit(Xtr, ytr)
t_sfs_bwd = time.perf_counter() - t0
bwd_sk = np.flatnonzero(sfs_bwd.get_support())
print(f"SequentialFeatureSelector(backward, 10) - {t_sfs_bwd:.2f} s")
print(fnames(bwd_sk))

# C.2.2 own implementation (down to a single feature, to draw the whole curve)
def backward_elimination(X, y, estimator, cv, min_features=1):
    """Greedy backward elimination. Start from all features; at each step remove the feature whose removal gives the
    highest (= least degraded) cross-validated accuracy."""
    t0 = time.perf_counter()
    current = list(range(X.shape[1]))
    s0 = cross_val_score(estimator, X, y, cv=cv, scoring="accuracy")
    history = [dict(n_features=len(current), removed="-", cv_mean=s0.mean(), cv_std=s0.std())]
    subsets, n_models = {len(current): list(current)}, cv.get_n_splits()
    while len(current) > min_features:
        cand = [cross_val_score(estimator, X[:, [f for f in current if f != j]], y, cv=cv, scoring="accuracy") for j in current]
        n_models += len(current) * cv.get_n_splits()
        means = [c.mean() for c in cand]
        b = int(np.argmax(means))
        removed = current.pop(b)
        subsets[len(current)] = list(current)
        history.append(dict(n_features=len(current), removed=FEATURES[removed], cv_mean=means[b], cv_std=cand[b].std()))
    return dict(history=pd.DataFrame(history), subsets=subsets, n_models=n_models, elapsed=time.perf_counter() - t0)


bwd = backward_elimination(Xtr, ytr, make_pipe(), CV, min_features=1)
hb = bwd["history"]
display(hb[hb.n_features <= 30].round(4).set_index("n_features").head(22))

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(hb.n_features, hb.cv_mean, yerr=hb.cv_std, marker="o", ms=4, capsize=2, color="#c0504d")
ax.invert_xaxis(); ax.set_xlabel("number of remaining features"); ax.set_ylabel("5-fold CV accuracy (training set)")
ax.set_title("Backward elimination (own implementation)"); plt.tight_layout(); show()

same = set(bwd["subsets"][10]) == set(bwd_sk)
print("own implementation retains the same 10 features as scikit-learn:", same)
if not same:
    print("  only own:", fnames(set(bwd["subsets"][10]) - set(bwd_sk)), "| only sklearn:", fnames(set(bwd_sk) - set(bwd["subsets"][10])))
N_BWD = smallest_k_within_1sd(hb.n_features, hb.cv_mean, hb.cv_std)
print("number of features chosen on validation data (smallest within 1 sd of the best):", N_BWD)

# C.2.3 forward versus backward
F10, B10 = set(fwd_10["selected"]), set(bwd["subsets"][10])
print("common to both  :", fnames(F10 & B10), f"({len(F10 & B10)})")
print("only forward    :", fnames(F10 - B10))
print("only backward   :", fnames(B10 - F10))
print(f"Jaccard similarity: {len(F10 & B10) / len(F10 | B10):.2f}")
cv_f10 = cv_acc(make_pipe(), Xtr[:, sorted(F10)], ytr)
cv_b10 = cv_acc(make_pipe(), Xtr[:, sorted(B10)], ytr)
print(f"CV accuracy with the forward subset : {cv_f10[0]:.4f} +/- {cv_f10[1]:.4f}")
print(f"CV accuracy with the backward subset: {cv_b10[0]:.4f} +/- {cv_b10[1]:.4f}")


# Comments (C.2.2). Again scikit-learn and my implementation retain the same 10 features (area error, smoothness error, fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst concavity, worst symmetry). The curve starts at 0.9824 with 30 features, rises to 0.9899 when the redundant features are removed (24 down to 13 features), and then falls slowly (0.9874 with 12 and 11, 0.9849 with 10 features). The smallest size within one standard deviation of the best (0.9899 ± 0.0095) is 3 features (worst texture, worst perimeter, worst smoothness).
#
# Forward versus backward (C.2.3). They share only 5 of 10 features (smoothness error, worst radius, worst texture, worst perimeter, worst smoothness; Jaccard similarity 0.33). The CV accuracy of the two subsets is 0.9899 ± 0.0095 for forward and 0.9849 ± 0.0123 for backward.
#
# Why are they not identical? Both algorithms are greedy and they follow different paths. Forward starts from nothing and keeps forever what it added first: it takes one member of the size family (`worst radius`) and then adds complementary features (worst smoothness, worst texture, mean radius, ...). Backward starts with all 30 features and first discards those that are redundant *given the others* (mean radius, mean perimeter, mean texture are the first three to go), and a feature that has been removed can never come back. Moreover the accuracy moves in coarse steps of 0.0025, so many candidate moves tie and the first one in column order is taken; many different subsets of 10 features have almost the same CV accuracy, so the "best" subset is not unique.


# ---- C.3 Recursive feature elimination (RFE) ----

# C.3.1 RFE with logistic regression on the standardized training data
rfe_lr = RFE(LogisticRegression(max_iter=1000), n_features_to_select=10).fit(X_train_s, y_train)
rfe_table = (pd.DataFrame({"feature": FEATURES, "selected": rfe_lr.support_, "ranking": rfe_lr.ranking_})
             .sort_values(["ranking", "feature"]).reset_index(drop=True))
display(rfe_table)

# C.3.2 RFECV (5-fold, accuracy); the estimator is a pipeline so that scaling is redone inside every fold
rfecv = RFECV(make_pipe(), step=1, cv=CV, scoring="accuracy",
              importance_getter="named_steps.clf.coef_").fit(Xtr, ytr)
res = rfecv.cv_results_
mean_s, std_s = np.asarray(res["mean_test_score"]), np.asarray(res["std_test_score"])
ns = np.arange(1, len(mean_s) + 1)

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.errorbar(ns, mean_s, yerr=std_s, marker="o", ms=4, capsize=2, color="#8064a2")
ax.axvline(rfecv.n_features_, ls="--", color="grey", label=f"optimal: {rfecv.n_features_} features"); ax.legend()
ax.set_xlabel("number of features kept by RFE"); ax.set_ylabel("5-fold CV accuracy (training set)")
ax.set_title("RFECV (logistic regression)"); plt.tight_layout(); show()
rfecv_sel = np.flatnonzero(rfecv.support_)
print("optimal number of features:", rfecv.n_features_,
      f"| CV accuracy there: {mean_s[ns == rfecv.n_features_][0]:.4f} +/- {std_s[ns == rfecv.n_features_][0]:.4f}",
      f"| with all 30: {mean_s[ns == 30][0]:.4f} +/- {std_s[ns == 30][0]:.4f}")
print(fnames(rfecv_sel))

# C.3.3 RFE with other estimators (standardized training data)
rfe_svc = RFE(SVC(kernel="linear"), n_features_to_select=10).fit(X_train_s, y_train)
rfe_rf = RFE(RandomForestClassifier(n_estimators=200, random_state=SEED), n_features_to_select=10).fit(X_train_s, y_train)
sets = {"RFE-LogReg": set(np.flatnonzero(rfe_lr.support_)), "RFE-SVC(linear)": set(np.flatnonzero(rfe_svc.support_)),
        "RFE-RandomForest": set(np.flatnonzero(rfe_rf.support_))}
for k_, v_ in sets.items():
    print(f"{k_:18s}", fnames(v_))
overlap = pd.DataFrame({a: {b: len(sets[a] & sets[b]) for b in sets} for a in sets})
print("\nnumber of common features (out of 10):"); display(overlap)
print("selected by all three:", fnames(set.intersection(*sets.values())))


# RFE with logistic regression (C.3.1). The 10 selected features (ranking 1) are mean area, mean concave points, area error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst symmetry and worst concave points. The next ones are mean compactness (2), mean radius (3), radius error (4), worst concavity (5), and the first eliminated are `worst compactness` (21), `mean fractal dimension` (20), `mean symmetry` (19) and `worst fractal dimension` (18). The ranking keeps several members of the size family (worst radius, perimeter, area, mean area, area error): the L2-penalized model spreads the weight over correlated features, so none of them gets a weight small enough to be eliminated early. `worst compactness`, which has a high univariate score, is eliminated first, probably because its information is already carried by correlated features present in the model.
#
# RFECV (C.3.2). The optimal number of features is 8 (mean area, area error, worst radius, worst texture, worst perimeter, worst area, worst smoothness, worst concave points) with a CV accuracy of about 0.985. The curve is flat (about 0.982) above 15 features, irregular between 9 and 14 and falls below 8; since the standard deviations are 0.015–0.03, the differences between 8 and 30 features are smaller than one standard deviation, so the optimum is only weakly determined.
#
# Other estimators (C.3.3). The selections are not the same. The linear SVC shares 8 of its 10 features with logistic regression, the random forest only 6 (and 5 with the SVC). Only 4 features are chosen by all three: worst radius, worst perimeter, worst area, worst concave points. RFE eliminates according to the importance given by the estimator: linear models use |weight| on standardized features, so complementary variables such as worst texture or worst smoothness are kept; the random forest uses impurity-based importance, which is dominated by the strongest single features (size and concavity) and keeps mean radius, mean perimeter, mean concavity, but not texture or smoothness. A RFE selection is therefore relative to the model used.
#
# RFE versus backward elimination (C.3.4). RFE removes the feature with the smallest importance (|coefficient|) of one fitted model, without cross-validation: for 30 → 10 features it trains about 21 models (20 eliminations + the final fit). Backward elimination removes the feature whose *removal gives the best cross-validated accuracy*: at each step it trains one model per candidate and per fold, 5 × (30 + 29 + ... + 11) = 2050 models for the same 30 → 10 path, about 100 times more. RFE is much cheaper but relies on a proxy (the weights) instead of the real validation performance; the two 10-feature subsets share 7 features.


# ---- C.4 Cost of the wrapper search ----

p, k, folds = 30, 10, CV.get_n_splits()
n_fwd_formula = folds * sum(p - i for i in range(k))           # forward: step i evaluates p - i candidates
n_all_subsets = (2 ** p - 1) * folds                            # exhaustive search over every non-empty subset
n_size_k = comb(p, k) * folds                                   # exhaustive search over the subsets of exactly k features

# time of one model (fit + predict on one fold), averaged over 200 fits
t0 = time.perf_counter(); reps = 40
for _ in range(reps):
    for tr, va in CV.split(Xtr, ytr):
        make_pipe().fit(Xtr[tr][:, :10], ytr[tr]).score(Xtr[va][:, :10], ytr[va])
t_model = (time.perf_counter() - t0) / (reps * folds)

def human(seconds):
    for unit, size in (("years", 365.25 * 86400), ("days", 86400), ("hours", 3600), ("minutes", 60)):
        if seconds >= size:
            return f"{seconds / size:,.1f} {unit}"
    return f"{seconds:.1f} s"

print(f"Forward selection, p={p}, k={k}, {folds}-fold CV:")
print(f"   formula {folds} x (30+29+...+21) = {n_fwd_formula} models | counted in my implementation: {fwd_10['n_models']}")
print(f"   one model (fit + score) = {t_model * 1000:.2f} ms  ->  predicted time {n_fwd_formula * t_model:.1f} s"
      f" | measured: own implementation {fwd_10['elapsed']:.1f} s, sklearn SFS {t_sfs_fwd:.1f} s")
print(f"Exhaustive search over all subsets : {2 ** p - 1:,} subsets x {folds} = {n_all_subsets:,} models  ~ {human(n_all_subsets * t_model)}")
print(f"Exhaustive search, exactly {k} features: C({p},{k}) = {comb(p, k):,} subsets x {folds} = {n_size_k:,} models  ~ {human(n_size_k * t_model)}")
print(f"Backward elimination 30 -> 10: {folds * sum(range(11, 31))} models (sklearn) | my full path 30 -> 1: {bwd['n_models'] - folds} candidate models + {folds} for the start")


# Forward selection with p = 30, k = 10 and 5-fold cross-validation: at step i = 0, ..., 9 the algorithm evaluates the 30 − i remaining features, each with 5 models, so
#
# 5 × (30 + 29 + ... + 21) = 5 × 255 = 1275 models (255 candidate subsets).
#
# Exhaustive search: all non-empty subsets, (2³⁰ − 1) × 5 = 1 073 741 823 × 5 = 5 368 709 115 models; only the subsets of exactly 10 features, C(30,10) × 5 = 30 045 015 × 5 = 150 225 075 models.
#
# Check against the measured time. My implementation counts exactly 1275 models. One model (fit + score on a fold) takes about 4 ms, which predicts about 5 s for the forward search; the measured time is about 5–6 s (own implementation and scikit-learn), so the count is consistent (the small excess is overhead). With the same time per model, the exhaustive search would need about 250 days for all subsets and about 7 days for the subsets of exactly 10 features on one core, which is why wrappers are always greedy.


# ---- C.5 (extra check) Is the greedy result optimal? Exhaustive search for 3 features ----
# With 30 features, all `C(30,3) = 4060` subsets of 3 features can be scored, which lets us compare the greedy paths with the true optimum for that size.

def score_subset(cols):
    return cross_val_score(make_pipe(), Xtr[:, list(cols)], ytr, cv=CV, scoring="accuracy").mean()

combos = list(itertools.combinations(range(30), 3))
t0 = time.perf_counter()
ex_scores = np.array([score_subset(c) for c in combos])
print(f"{len(combos)} subsets scored in {time.perf_counter() - t0:.0f} s")

best = int(np.argmax(ex_scores))
fwd3 = tuple(sorted(fwd_10["selected"][:3])); bwd3 = tuple(sorted(bwd["subsets"][3]))
rank = lambda c: int((ex_scores > ex_scores[combos.index(c)] + 1e-12).sum()) + 1
print(f"best subset of 3 (exhaustive): {fnames(combos[best])} -> CV accuracy {ex_scores[best]:.4f}")
print(f"forward  at 3 features        : {fnames(fwd3)} -> {ex_scores[combos.index(fwd3)]:.4f} (rank {rank(fwd3)} of {len(combos)})")
print(f"backward at 3 features        : {fnames(bwd3)} -> {ex_scores[combos.index(bwd3)]:.4f} (rank {rank(bwd3)} of {len(combos)})")
print(f"number of 3-feature subsets reaching the best score: {int((ex_scores >= ex_scores[best] - 1e-12).sum())}")


# With 3 features the exhaustive search is feasible (4060 subsets). The best CV accuracy is 0.9824, reached by {worst radius, worst texture, worst smoothness}: this is exactly the subset obtained by forward selection at step 3 (rank 1 of 4060). Backward elimination at 3 features gives {worst texture, worst perimeter, worst smoothness}, which also reaches 0.9824 (two subsets tie for the best). So on this dataset the greedy searches find an optimal subset for k = 3. This is an observation, not a guarantee (see question 4 of the final discussion).


# ---- Summary of the features selected by the wrapper methods ----

wrapper_table = pd.DataFrame([
    ("Forward (sklearn, 10)", len(fwd_sk), ", ".join(fnames(fwd_sk))),
    ("Forward (own, 10)", 10, ", ".join(fnames(fwd_10["selected"]))),
    (f"Forward (own, gain rule, {N_FWD})", N_FWD, ", ".join(fnames(fwd_eps["selected"]))),
    ("Backward (sklearn, 10)", len(bwd_sk), ", ".join(fnames(bwd_sk))),
    ("Backward (own, 10)", 10, ", ".join(fnames(bwd["subsets"][10]))),
    (f"Backward (own, 1-sd rule, {N_BWD})", N_BWD, ", ".join(fnames(bwd["subsets"][N_BWD]))),
    ("RFE logistic regression (10)", 10, ", ".join(fnames(np.flatnonzero(rfe_lr.support_)))),
    (f"RFECV ({rfecv.n_features_})", rfecv.n_features_, ", ".join(fnames(rfecv_sel))),
    ("RFE SVC linear (10)", 10, ", ".join(fnames(np.flatnonzero(rfe_svc.support_)))),
    ("RFE random forest (10)", 10, ", ".join(fnames(np.flatnonzero(rfe_rf.support_)))),
], columns=["method", "n_features", "selected features"]).set_index("method")
display(wrapper_table)


# ------------------------------------------------------------------------------
# ==============================================================================
# PART D. NOISE INJECTION, LEAKAGE AND FINAL COMPARISON
# ==============================================================================
# All selection steps below are put in a `Pipeline` (scaler + selector [+ classifier]). The helper `selector_pipe` builds the pipeline of each method; the wrappers use a `make_pipe()` estimator internally so that the standardization is refitted in every inner fold.

def selector_pipe(kind, n=10, thr=None, with_clf=False):
    """Pipeline(scaler, selector [, logistic regression]) for one selection method."""
    if kind == "kbest_f":
        sel = SelectKBest(f_classif, k=n)
    elif kind == "kbest_mi":
        sel = SelectKBest(partial(mutual_info_classif, random_state=SEED), k=n)
    elif kind == "corr":
        sel = CorrelationFilter(thr)
    elif kind == "forward":
        sel = SequentialFeatureSelector(make_pipe(), n_features_to_select=n, direction="forward", scoring="accuracy", cv=CV)
    elif kind == "backward":
        sel = SequentialFeatureSelector(make_pipe(), n_features_to_select=n, direction="backward", scoring="accuracy", cv=CV)
    elif kind == "rfe":
        sel = RFE(LogisticRegression(max_iter=1000), n_features_to_select=n)
    elif kind == "rfecv":
        sel = RFECV(make_pipe(), step=1, cv=CV, scoring="accuracy", importance_getter="named_steps.clf.coef_")
    else:
        raise ValueError(kind)
    steps = [("scaler", StandardScaler()), ("sel", sel)]
    if with_clf:
        steps.append(("clf", LogisticRegression(max_iter=1000)))
    return Pipeline(steps)


# ---- D.1 Can the methods find the noise? ----

NOISE_COLS = [f"noise_{i:02d}" for i in range(20)]

def make_noisy(noise_seed, shuffle_columns=False):
    """30 original features + 20 N(0,1) noise features (independent of the target), indexed like X.
    shuffle_columns=True puts the 50 columns in a random order (used only for the tie-breaking check below)."""
    rng = np.random.default_rng(noise_seed)
    noise = pd.DataFrame(rng.standard_normal((len(X), 20)), columns=NOISE_COLS, index=X.index)
    out = pd.concat([X, noise], axis=1)
    if shuffle_columns:
        out = out.iloc[:, np.random.default_rng(1000 + noise_seed).permutation(out.shape[1])]
    return out

X50 = make_noisy(SEED)
X50_train, X50_test = X50.loc[X_train.index], X50.loc[X_test.index]        # same split as before
print("50-feature data:", X50_train.shape, X50_test.shape)

NOISE_METHODS = {"SelectKBest (F-score)": "kbest_f", "Forward Selection": "forward",
                 "Backward Elimination": "backward", "RFE": "rfe"}

def run_noise_experiment(Xnoisy_train, y_tr):
    out = {}
    for name, kind in NOISE_METHODS.items():
        t0 = time.perf_counter()
        pipe = selector_pipe(kind, n=10).fit(Xnoisy_train.values, y_tr.values)
        chosen = Xnoisy_train.columns[pipe.named_steps["sel"].get_support()].tolist()
        out[name] = dict(selected=chosen, n_noise=sum(c.startswith("noise_") for c in chosen), seconds=time.perf_counter() - t0)
    return out

noise_res = run_noise_experiment(X50_train, y_train)
display(pd.DataFrame({m: dict(n_noise_in_10=r["n_noise"], real_features=", ".join(c for c in r["selected"] if not c.startswith("noise_")),
                              noise_features=", ".join(c for c in r["selected"] if c.startswith("noise_")) or "-")
                      for m, r in noise_res.items()}).T)

# Robustness checks (not required; set RUN_ROBUSTNESS = False to skip, they take a few minutes)
RUN_ROBUSTNESS = True
if RUN_ROBUSTNESS:
    # (1) the same experiment with three other noise draws
    rob = {f"seed {SEED}": {m: r["n_noise"] for m, r in noise_res.items()}}
    for s in (1, 2, 3):
        Xs = make_noisy(s).loc[X_train.index]
        rob[f"seed {s}"] = {m: r["n_noise"] for m, r in run_noise_experiment(Xs, y_train).items()}
    rob_df = pd.DataFrame(rob)
    rob_df["mean"] = rob_df.mean(axis=1)
    print("(1) number of noise features among the 10 selected (out of 20 available), noise columns placed LAST:")
    display(rob_df)

    # (2) tie-breaking check: accuracy moves in steps of 0.25 points, so many candidates tie and scikit-learn takes the
    #     lowest column index. With the noise columns placed last this favours real features in forward selection and
    #     penalizes them in backward elimination. Same experiment with the 50 columns in random order:
    shuf = {}
    for s in (SEED, 1, 2):
        Xs = make_noisy(s, shuffle_columns=True).loc[X_train.index]
        shuf[f"seed {s}"] = {m: r["n_noise"] for m, r in run_noise_experiment(Xs, y_train).items()}
    shuf_df = pd.DataFrame(shuf)
    shuf_df["mean"] = shuf_df.mean(axis=1)
    print("(2) same, with the 50 columns in random order:")
    display(shuf_df)


# Answer D.1 (noise columns appended after the 30 real ones, 10 features kept by each method):
# * seed 42: SelectKBest (F-score) 0 noise features, RFE 0, Forward Selection 2 (noise_00, noise_05), Backward Elimination 4 (noise_00, noise_12, noise_17, noise_18).
# * three other noise draws (seeds 1, 2, 3): F-score 0, 0, 0 | RFE 0, 0, 0 | Forward 2, 0, 2 | Backward 3, 3, 5.
# * Most robust: the F-score filter and RFE with logistic regression. Most likely fooled: the wrappers that rely on the CV accuracy as criterion, and the most in this experiment is Backward Elimination (3 to 5 noise features out of 10).
# * Why: a noise feature has an F-score close to 1 (real features: tens to hundreds) and a weight close to 0 in the logistic regression, so univariate scoring and weight-based elimination rank it last. The CV accuracy used by forward/backward is a noisy and coarse criterion (steps of 0.0025 on a plateau around 0.98): adding or keeping a noise feature can gain one patient by chance and is then accepted, and nothing in the criterion penalizes useless features.
# * Caveat: when several candidates tie, scikit-learn takes the lowest column index. Here the noise columns come last, which favours real features in forward selection and penalizes them in backward elimination. The second table printed by the robustness cell repeats the experiment with the 50 columns in random order, so you can see how much of the backward result is due to this.


# ---- D.2 Leakage in feature selection ----

def leakage_compare(Xm, y_, cv, k=10):
    """Accuracy of SelectKBest(f_classif, k) + logistic regression, estimated the incorrect and the correct way."""
    sel = SelectKBest(f_classif, k=k).fit(Xm, y_)                     # INCORRECT: selection sees all rows (also the future validation folds)
    wrong = cross_val_score(make_pipe(), sel.transform(Xm), y_, cv=cv, scoring="accuracy").mean()
    pipe = Pipeline([("scaler", StandardScaler()), ("sel", SelectKBest(f_classif, k=k)),
                     ("clf", LogisticRegression(max_iter=1000))])     # CORRECT: selection refitted inside every fold
    right = cross_val_score(pipe, Xm, y_, cv=cv, scoring="accuracy").mean()
    return wrong, right

A50, a50y = X50_train.values, y_train.values
wrong, right = leakage_compare(A50, a50y, CV)
print(f"50 features, 5-fold CV (seed 42):  incorrect = {wrong:.4f} | correct (Pipeline) = {right:.4f} | gap = {wrong - right:+.4f}")

gaps = []
for s in range(30):                                                   # the gap on this dataset is tiny: average over 30 different fold assignments
    w, r = leakage_compare(A50, a50y, StratifiedKFold(5, shuffle=True, random_state=s))
    gaps.append((w, r))
gaps = np.array(gaps)
print(f"average over 30 CV shuffles:        incorrect = {gaps[:, 0].mean():.4f} | correct = {gaps[:, 1].mean():.4f} | "
      f"mean gap = {np.mean(gaps[:, 0] - gaps[:, 1]):+.4f} (sd {np.std(gaps[:, 0] - gaps[:, 1]):.4f})")

# When the leakage becomes large: few observations, many features, NO real signal (labels are random)
rng = np.random.default_rng(0)
demo = []
for r in range(30):
    Z = rng.standard_normal((60, 2000))                     # 60 observations, 2000 pure-noise features
    yz = np.array([0, 1] * 30); rng.shuffle(yz)             # random labels: the true accuracy is 50 %
    demo.append(leakage_compare(Z, yz, StratifiedKFold(5, shuffle=True, random_state=r)))
demo = np.array(demo)
print("60 observations x 2000 noise features, random labels (30 repetitions):")
print(f"   incorrect (selection before CV): {demo[:, 0].mean():.3f}   <- looks like a good classifier")
print(f"   correct   (selection in the Pipeline): {demo[:, 1].mean():.3f}   <- chance level, the truth")


# Result. On the 50-feature training set, with SelectKBest(F-score, k = 10) + logistic regression:
#
# | protocol | 5-fold CV accuracy |
# |---|---|
# | incorrect (selection on all training rows, then CV) | 0.9523 |
# | correct (scaler + selector + classifier in a Pipeline) | 0.9523 |
#
# The gap is 0.000, and averaged over 30 different fold assignments it is only +0.0003 (standard deviation 0.0016): the leak is invisible here.
#
# How information leaks. In the incorrect protocol the F-scores are computed with the labels of *all* training patients, including those that will later be the validation fold. The 10 chosen features are therefore the ones that best separate the validation patients too, so the validation fold is no longer "unseen" and the accuracy is optimistic. In the Pipeline, the scaler and `SelectKBest` are refitted on the 4 training folds at each split, and the validation fold is only used for scoring.
#
# Why the gap is small here. (i) The signal is strong: the same top features (size and concavity) are chosen whichever rows are used, so adding the validation rows hardly changes the selected set; in D.1 the F-score filter never selected a noise feature. (ii) There are 398 observations for 50 features, so the chance correlation of a noise feature with the label is tiny compared with the true signal.
#
# When it becomes large. With few observations, many features and a weak signal (genomics, imaging, text). Demonstration above: 60 observations, 2000 features of pure noise, random labels. The incorrect protocol gives 0.860 accuracy (it looks like a good classifier!), the correct one 0.507, which is chance level and the truth. The more flexible the selection (wrappers, tuning k), the larger the effect.


# ---- D.3 Final comparison on the test set (30 original features) ----

# number of features of every method = the value chosen on validation data in the previous parts
N_FEATS = {"corr": None, "kbest_f": K_F, "kbest_mi": K_MI, "forward": N_FWD, "backward": N_BWD, "rfe": 10, "rfecv": None}
final_specs = [
    ("Baseline (all features)",       make_pipe()),
    ("Filter: correlation (B.2)",     selector_pipe("corr", thr=B2_THRESHOLD, with_clf=True)),
    ("Filter: SelectKBest, F-score",  selector_pipe("kbest_f", n=K_F, with_clf=True)),
    ("Filter: SelectKBest, mutual information", selector_pipe("kbest_mi", n=K_MI, with_clf=True)),
    ("Wrapper: Forward Selection",    selector_pipe("forward", n=N_FWD, with_clf=True)),
    ("Wrapper: Backward Elimination", selector_pipe("backward", n=N_BWD, with_clf=True)),
    ("Wrapper: RFE",                  selector_pipe("rfe", n=10, with_clf=True)),
    ("Wrapper: RFECV",                selector_pipe("rfecv", with_clf=True)),
]

final_rows, preds, support = [], {}, {}
for name, pipe in final_specs:
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)                                     # fitted on the training set only (selection time = whole fit)
    fit_time = time.perf_counter() - t0
    pred = pipe.predict(Xte)                               # the test set is evaluated ONCE per method
    preds[name] = pred
    support[name] = pipe.named_steps["sel"].get_support() if "sel" in pipe.named_steps else np.ones(30, dtype=bool)
    final_rows.append({"Method": name, "Nb features": int(support[name].sum()),
                       "Test accuracy": accuracy_score(yte, pred), "Test F1-score": f1_score(yte, pred),
                       "Selection time (s)": np.nan if name.startswith("Baseline") else fit_time})
final = pd.DataFrame(final_rows).set_index("Method")
display(final.round(4))
print("how the number of features was chosen: correlation -> best CV threshold (%.2f); SelectKBest -> smallest k within 1 sd (F: %d, MI: %d);" % (B2_THRESHOLD, K_F, K_MI))
print("forward -> gain rule of C.1 (%d); backward -> smallest size within 1 sd on its CV curve (%d); RFE -> 10 as in C.3.1; RFECV -> its own optimum." % (N_FWD, N_BWD))

# Is a difference with the baseline significant? (171 test patients: one patient = 0.58 points of accuracy)
n_te = len(yte)
def mcnemar_exact(pred_a, pred_b):
    a_ok, b_ok = pred_a == yte, pred_b == yte
    only_a, only_b = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    n = only_a + only_b
    return only_a, only_b, (1.0 if n == 0 else binomtest(min(only_a, only_b), n, 0.5).pvalue)

rows = []
for name in final.index[1:]:
    a, b, p_ = mcnemar_exact(preds["Baseline (all features)"], preds[name])
    rows.append({"Method": name, "patients right only with baseline": a, "patients right only with method": b, "McNemar exact p-value": p_})
display(pd.DataFrame(rows).set_index("Method").round(3))

acc_b = final.loc["Baseline (all features)", "Test accuracy"]
print(f"one patient = {1 / n_te:.4f} accuracy; standard error of an accuracy of {acc_b:.3f} on {n_te} patients = {np.sqrt(acc_b * (1 - acc_b) / n_te):.4f}")

best_acc = final["Test accuracy"].max()
best_name = final[final["Test accuracy"] >= best_acc - 1e-12].sort_values("Nb features").index[0]
print(f"best test accuracy: {best_acc:.4f}; among the methods reaching it, the fewest features: {best_name} ({final.loc[best_name, 'Nb features']} features)")

# Presence matrix: features (rows) x methods (columns)
methods = list(final.index[1:])
presence = pd.DataFrame({m: support[m] for m in methods}, index=FEATURES)
presence["n_methods"] = presence[methods].sum(axis=1)

fig, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(presence[methods].astype(int), cmap=sns.color_palette(["#f2f2f2", "#4f81bd"]), cbar=False, linewidths=0.5,
            linecolor="white", yticklabels=[f"{f}  ({n})" for f, n in zip(FEATURES, presence["n_methods"])], ax=ax)
ax.set_title("Features selected (blue) by each method; in brackets: number of methods selecting the feature")
ax.set_xticklabels([m.replace("Filter: ", "F: ").replace("Wrapper: ", "W: ").replace("SelectKBest, ", "KBest ") for m in methods], rotation=30, ha="right")
plt.tight_layout(); show()

print("selected by ALL", len(methods), "methods:", presence.index[presence.n_methods == len(methods)].tolist())
print("selected by at least", len(methods) - 1, "methods:", presence.index[presence.n_methods >= len(methods) - 1].tolist())
print("selected by no method:", presence.index[presence.n_methods == 0].tolist())

# link with the correlation structure of Part A: how many members of each redundant group does each method keep?
grp = pd.DataFrame({m: [int(presence.loc[g, m].sum()) for g in REDUNDANT_GROUPS] for m in methods},
                   index=[f"{len(g)} features: " + ", ".join(g[:3]) + (" ..." if len(g) > 3 else "") for g in REDUNDANT_GROUPS])
print("\nmembers kept per redundant group (|corr| > 0.95, Part A):")
display(grp)


# Results on the test set (each pipeline is fitted on the training set and evaluated once; selection times are those of this run and only indicative):
#
# | Method | Nb features | Test accuracy | Test F1 | Selection time |
# |---|---|---|---|---|
# | Baseline (all features) | 30 | 0.9883 | 0.9907 | – |
# | Filter: correlation (B.2, 0.90) | 20 | 0.9766 | 0.9813 | about 0.01 s |
# | Filter: SelectKBest, F-score | 20 | 0.9766 | 0.9813 | about 0.005 s |
# | Filter: SelectKBest, mutual information | 20 | 0.9708 | 0.9770 | about 0.06 s |
# | Wrapper: Forward Selection | 4 | 0.9415 | 0.9528 | about 2 s |
# | Wrapper: Backward Elimination | 3 | 0.9532 | 0.9626 | about 12 s |
# | Wrapper: RFE | 10 | 0.9708 | 0.9767 | about 0.06 s |
# | Wrapper: RFECV | 8 | 0.9591 | 0.9671 | about 0.7 s |
#
# Comments. (1) No selection method beats the baseline: with 171 test patients, 0.9883 means 169 correct. The filters keep 20 features and lose 2–3 patients, RFE keeps 10 and loses 3, RFECV 8 and loses 5, backward (3 features) loses 6 and forward (4 features) loses 8. (2) The very small subsets of forward and backward are the worst on the test set although their CV accuracy was about 0.98: their sizes were chosen by rules that cut the plateau (ε = 0.001, one standard deviation), and a subset picked as the best on the same folds has an optimistic CV score. (3) Filters and RFE are fast (milliseconds); forward and backward take seconds and tens of seconds.
#
# Presence matrix and consistency with Part A. Only `worst texture` is selected by all 7 methods; `worst perimeter` and `worst smoothness` are selected by 6 of 7. This is consistent with the correlation structure of Part A:
# * The univariate filters (F-score, mutual information) keep all 6 members of the radius / perimeter / area block and all 3 error copies, whereas the correlation filter keeps 1 of each, backward 1 and 0, forward 2 and 0, RFE 4 and 1.
# * The features that almost every method agrees on are the ones that are not part of any redundant block (worst texture, worst smoothness): they carry information that nothing else carries. For the size block, the methods agree that *one* representative is needed but not which one (worst radius, worst perimeter, worst area, mean area, ...), because the members are interchangeable.


# ------------------------------------------------------------------------------
# ==============================================================================
# FINAL DISCUSSION
# ==============================================================================
#
# 1. Which method gives the best test accuracy with the fewest features? Is the difference with the baseline significant?
# No reduced model beats the baseline: it has the best test accuracy (0.9883, 30 features). The best compromise is RFE with 10 features (0.9708), which loses 3 patients of 171; the filters with 20 features (0.9766 for correlation and F-score) lose 2. These differences are not significant: one patient is 0.58 points of accuracy, the standard error of an accuracy of 0.988 on 171 patients is about 0.8 points, and McNemar's exact test gives p = 0.5 (correlation, F-score), 0.25 (mutual information, RFE) and 0.06 (RFECV). Only forward selection (4 features, 8 patients lost, p = 0.008) and backward elimination (3 features, 6 patients lost, p = 0.031) are lower than the baseline, and with 7 comparisons these p-values are borderline after correction.
#
# 2. Advantages and limits of filters compared with wrappers.
# Filters are very fast (milliseconds), independent of any model, and scale to thousands of features. But they score each feature alone: they ignore redundancy (6 of the 10 best F-scores are copies of the size feature, and accuracy stays around 0.95–0.96 for k ≤ 15) and interactions (`worst smoothness` has a mediocre F-score, rank 18, yet forward selection picks it second because it complements `worst radius`). Wrappers evaluate subsets with the real model and metric, so they handle redundancy and complementarity and are tuned to the model, but they are slower and the result depends on the model (RFE with a random forest shares only 6 of 10 features with RFE with logistic regression) and on the noisy CV estimate.
#
# 3. Why are wrappers more expensive?
# They must train the model for every candidate subset and every fold. Forward selection with p features, k selected and F folds trains F × (p + (p − 1) + ... + (p − k + 1)) = F × (k·p − k(k − 1)/2) models: 1275 for p = 30, k = 10, F = 5 (measured: seconds, against 5 ms for the F-score filter), and about 99 000 for p = 1000 and k = 20. An exhaustive search would need 2ᵖ − 1 subsets, i.e. more than 10⁹ for p = 30.
#
# 4. Do forward selection and backward elimination guarantee the optimal subset?
# No. They are greedy: forward can never remove a feature it has added, backward can never bring back a feature it has removed, so they cannot find a subset that needs a swap. In our results the 10-feature subsets of the two methods differ (5 features in common) and have different CV accuracies (0.9899 for forward, 0.9849 for backward), so at least the backward subset is not optimal for size 10. For k = 3, where exhaustive search is possible (C.5), both greedy searches did reach the optimum (0.9824), so greedy methods often work well in practice, but without any guarantee. Many subsets also tie because the accuracy changes by steps of 0.0025.
#
# 5. Behaviour in front of the 20 noise features.
# The F-score filter and RFE (logistic regression) never selected a noise feature (0 of 10, for the 4 noise draws). Forward selection selected 0 to 2 and backward elimination 3 to 5 noise features out of 10. Filters and weight-based RFE rely on a statistic with a strong signal-to-noise ratio (F-score near 1 or weight near 0 for noise), so they are robust here. Wrappers optimize a noisy, coarse CV accuracy on a plateau, so a useless feature can be accepted by chance: they are the family most likely to be fooled, and the effect is larger the more features are searched (backward elimination starts from all 50). Part of the backward result comes from tie-breaking by column index (see the caveat in D.1; the robustness cell prints the same experiment with shuffled columns).
#
# 6. Why must feature selection be performed inside the cross-validation loop, and how does a Pipeline prevent leakage?
# Selecting features is part of the model fitting: it uses the labels. If it is done before cross-validation on all rows, the validation folds have already influenced the model (the chosen features are the ones that happen to separate those patients), and the CV score is biased upwards, as shown in D.2 (0.860 against 0.507 on random labels). A `Pipeline` bundles scaler, selector and classifier into one estimator, so `cross_val_score` refits all the steps on the training folds only and applies them to the validation fold; the same holds for the final test set, which is only transformed, never used for fitting.
#
# 7. In which situation would you choose each family?
# * Thousands of features: filters (univariate scores, mutual information, correlation pruning) as a first screening, possibly followed by RFE or an embedded method on the reduced set; wrappers such as forward selection would need hundreds of thousands of models.
# * Small dataset: simple filters or strongly regularized models, with the whole selection inside a (nested) cross-validation; flexible wrappers overfit the CV estimate (here the 3–4 feature subsets had 0.98 CV accuracy but 0.94–0.95 on the test set) and leakage is largest.
# * Need for interpretability: a short, non-redundant list: correlation pruning followed by a filter, or RFE/forward selection with a fixed small number of features, accepting a small loss of accuracy.
# * Limited computing time: filters (milliseconds) or RFE (about 20 fits); avoid forward/backward searches.
# * Moderate number of features (tens) and a fixed model whose accuracy matters most: wrappers (RFECV, forward/backward) are affordable and tune the subset to the model.
#
# Take-away for this dataset. Logistic regression with all 30 features is already excellent (98.8 %) and feature selection does not improve it, but a 10-feature subset (RFE) keeps 97.1 % and a 20-feature non-redundant subset 97.7 %, and the selection is consistent with the correlation structure: one representative per block of redundant size measurements plus the independent texture and smoothness features.


# ------------------------------------------------------------------------------
# All numbers above come from a single run with `SEED = 42` and scikit-learn as printed at the top of the notebook; small differences may appear with another version.*