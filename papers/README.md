# Generated Papers

Each PDF here is the final artifact of one end-to-end Auto-Mini-Claw run.
The complete artifact set per run (`draft.tex`, `metrics.json`,
`python_code.py`, `claim_ledger.json`, `debate_log.json`,
`execution_logs.txt`, etc.) is stored in Supabase Storage under
`artifacts/<run_id>/`.

Each entry below records the exact CLI input that produced the paper, the
pipeline-generated hypothesis it tested, and the final pipeline result.

---

## 001 — k-means clustering on iris with kernel approximation

- **PDF**: [001_kmeans_iris_distance_metrics.pdf](001_kmeans_iris_distance_metrics.pdf)
- **Run ID**: `736dedf0-8002-4db1-864a-2c0910c10909`
- **Started**: 2026-04-26
- **Status**: ✅ success
- **Code retries**: 0 · **LaTeX repair attempts**: 1 · **Confidence**: 6.5

**CLI input** (the topic passed to `mini-claw`):
```
k-means clustering with different distance metrics on iris dataset
```

**Generated hypothesis**:
> Applying Kernel k-means with randomized low-rank kernel approximation
> (using Kernel similarity between sampled points and all data points) to
> the Iris dataset, while evaluating cluster assignments using both the
> Davis-Bouldin Index and Calinski-Harabasz criterion, will reveal that the
> trade-off between Runtime complexity and Clustering quality is modulated
> by Sample Size: specifically, there exists a critical Sample Size threshold
> on the Iris dataset below which randomized Kernel k-means yields Clustering
> quality statistically indistinguishable from standard K-Means (measured by
> Davis-Bouldin Index), yet achieves significantly lower Runtime complexity
> and Memory footprint — and that this threshold generalizes across the three
> Distance Functions already benchmarked for K-Means on Iris.

**Experiment**: 30 runs × 10 Nyström sample sizes (5–150) on iris;
RBF kernel with median-heuristic γ; compared against Euclidean k-means
baseline.

---

## 002 — Logistic-regression decision-threshold sweep on breast cancer (RESCUED)

- **PDF**: [002_breast_cancer_threshold_pareto_RESCUED.pdf](002_breast_cancer_threshold_pareto_RESCUED.pdf)
- **Run ID**: `8ac60002-2e43-4014-beb0-d8587f7f5b46`
- **Started**: 2026-04-27
- **Status**: ⚠️ failed_latex (PDF rescued by manual recompile)
- **Code retries**: 0 · **LaTeX repair attempts**: 5 (exhausted) · **Confidence**: 7.2

**CLI input**:
```
Logistic regression decision threshold sweep on sklearn breast_cancer tabular dataset: precision-recall-FNR Pareto analysis for malignant tumor detection
```

**Generated hypothesis** (after 4 HITL Gate-1 rejections to steer away from
CNN/TPE):
> On the sklearn breast_cancer tabular dataset, a decision threshold sweep
> over Logistic Regression combined with a sparsity penalty will yield a
> superior Pareto frontier in precision-recall-FNR space for malignant tumor
> detection compared to a plain Logistic Regression without penalization, and
> the optimal threshold region identified via Pareto analysis will shift
> significantly (>0.05) when elastic-net penalization is applied versus
> L1-only sparsity penalization — suggesting that the choice of penalty
> structure non-trivially reshapes the precision-recall-FNR trade-off
> surface in a way that scalar metrics alone fail to capture.

**Experiment**: 5 seeds × 80/20 stratified split × 5-fold inner CV ×
{no-penalty, L1, elastic-net} × 99-step threshold sweep on
`sklearn.breast_cancer`.

**Failure cause**: the LaTeX writer emitted
`\includegraphics{figures/placeholder_hypervolume.pdf}` without `,draft`
mode, and the LLM repair agent burned all 5 attempts without spotting the
fix. The PDF here was produced by manually patching the source and running
`pdflatex` locally. The deterministic
`neutralize_missing_graphics()` pre-pass in
[backend/utils/latex_utils.py](../backend/utils/latex_utils.py) was added
afterwards to prevent that failure mode.

---

## 003 — Learning-curve analysis of Gradient Boosting on wine

- **PDF**: [003_wine_gbdt_learning_curves.pdf](003_wine_gbdt_learning_curves.pdf)
- **Run ID**: `766a9c62-b663-41c3-8a25-ef04b8d5a47a`
- **Started**: 2026-04-27
- **Status**: ✅ success
- **Code retries**: 0 · **LaTeX repair attempts**: 0 · **Confidence**: 6.5

**CLI input**:
```
Learning curve analysis of Gradient Boosting classifier on sklearn wine dataset: how training set size affects accuracy, F1, and overfitting gap
```

**Generated hypothesis**:
> On the sklearn wine dataset, the learning curve of a Gradient Boosting
> Decision Tree classifier will exhibit a non-monotonic overfitting gap
> (train accuracy minus validation accuracy) as a function of training set
> size … the critical inflection point occurs at roughly 40–60% of the full
> training set size … F1-score on the validation split will converge to its
> asymptote faster than raw Accuracy … compared against a Neural Networks
> baseline on the same dataset.

**Experiment**: 10 logarithmic training fractions (5–95% of a 142-sample
training pool) × 10 sub-sampling repeats × {GBDT, MLP} on
`sklearn.wine` (178 samples, 13 features, 3 classes). GBDT hyperparameters
fixed after grid search on full training set to isolate data-size effects.

**Honesty note**: 3 of 4 hypothesis predictions were **falsified** by the
data and the paper reports them as such — empirical inflection at ~25.7%
(not 40-60%); GBDT wall-clock scaled only 2.23× (not super-linearly); MLP
showed the same non-monotonic gap pattern (so the effect is dataset-driven,
not GBDT-specific).
