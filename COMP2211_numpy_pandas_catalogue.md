# COMP 2211 NumPy and pandas Knowledge Catalogue

This catalogue records the NumPy and pandas methods used in the implemented
COMP 2211 lab TODOs. It is a record of the current solutions, not a general
reference for every method in either library.

**Source rule:** a method is included only when it appears in code added inside
an implemented TODO block. Imports, function signatures, docstrings, data setup,
prewritten helper bodies, and unfinished TODOs are not treated as the user's
implementation.

**Checked against the task notebooks:** 2026-10-24.

## 1. Scope and source status

| Source | Included | Why |
|---|---|---|
| [Lab 2](lab2/lab2_tasks.ipynb) | All TODO function bodies | The notebook contains completed implementations for Tasks 1–5. |
| [Lab 4](lab4/lab4_tasks.ipynb) | Tasks 1–4 | The notebook is the skeleton; the matching `.py` file contains the completed Task 1–4 bodies. |
| [PA1](pa1/pa1_tasks.ipynb) | TODO 1–8 | These TODO bodies are implemented in `pa1_tasks.py`. |
| PA1 TODO 9–14 | Excluded | The skeleton still contains empty or partial branches for these TODOs. |
| PA1 prewritten helpers | Excluded | Code already present in the notebook, such as `_log_scores`, `predict_proba`, and unused scaffold branches, is not attributed to the TODO implementations. |

The detailed [Lab 1 NumPy guide](lab1/lab1_numpy_guide.md) remains the place for
deeper explanations of indexing, views, and general broadcasting. This catalogue
links to it where those ideas recur in the completed tasks.

## 2. Contents

- [3. Shape and axis model](#3-shape-and-axis-model)
- [4. pandas methods used](#4-pandas-methods-used)
- [5. NumPy counting and grouping](#5-numpy-counting-and-grouping)
- [6. Lab 2: Naive Bayes methods](#6-lab-2-naive-bayes-methods)
- [7. Lab 4: K-Means methods](#7-lab-4-k-means-methods)
- [8. PA1 TODO 1–8: scaling, clustering, GNB, and explanation](#8-pa1-todo-18-scaling-clustering-gnb-and-explanation)
- [9. Reusable recipes](#9-reusable-recipes)
- [10. Current-code notes and common mistakes](#10-current-code-notes-and-common-mistakes)
- [11. Completed-TODO crosswalk](#11-completed-todo-crosswalk)
- [12. Revision checklist](#12-revision-checklist)

## 3. Shape and axis model

These are the shapes that occur repeatedly in the completed task code.

| Object | Shape | Meaning |
|---|---:|---|
| `X` | `(n, d)` | `n` samples and `d` features |
| `y` | `(n,)` | One label per sample |
| `mean` | `(d,)` | One mean per feature |
| `X - mean` | `(n, d)` | The `(d,)` mean is broadcast down every row |
| `X_test[:, None]` | `(n_test, 1)` | Insert a size-1 column axis |
| `mean[None, :]` | `(1, C)` | Insert a size-1 row axis |
| `(n_test, 1) - (1, C)` | `(n_test, C)` | One value per test sample and class |
| `X[:, None, :]` | `(n, 1, d)` | Prepare samples for comparison with centroids |
| `(n, 1, d) - (k, d)` | `(n, k, d)` | Compare every sample with every centroid |
| `prob` | `(K, C)` | One probability per category and class |

### 3.1 What `axis` means

For reductions, `axis` names the axis that disappears:

```python
column_means = X.mean(axis=0)   # (n, d) -> (d,)
row_sums = scores.sum(axis=1)   # (n, C) -> (n,)
feature_sums = scores.sum(axis=0)  # (n, C) -> (C,)
```

The reduction interpretation is more useful than calling `axis=0` “down” and
`axis=1` “across”.

### 3.2 Broadcasting in two steps

1. Right-align the shapes.
2. Stretch every size-1 axis; nonmatching sizes other than 1 are an error.

```python
(n, 1, d)
(k, d)      # treated as (1, k, d)
---------
(n, k, d)
```

This is the exact shape pattern used by Lab 4 `assign_labels` and PA1
`KMeansClustering.fit`. See
[§18 Broadcasting](lab1/lab1_numpy_guide.md#18-broadcasting) for the detailed
rules.

### 3.3 Indexing patterns used

```python
X[:, i]                 # every row, feature i -> (n,)
X[indices]              # selected rows -> (len(indices), d)
mean[y]                 # each sample's class mean -> (n,)
labels == c             # Boolean mask -> (n,)
X[mask]                 # rows selected by mask
prob[test_codes]        # one likelihood row per test code -> (n_test, C)
values[order]           # reorder values into ranking order
```

Boolean masks combine with `&`, `|`, and `~`, not `and`, `or`, and `not`. See
[§12 Boolean array indexing](lab1/lab1_numpy_guide.md#12-boolean-array-indexing).

## 4. pandas methods used

### 4.1 Load, filter, select, and drop missing values

**Used in:** Lab 4 Task 1.

```python
source = pd.read_csv(file_name)
filtered = source[source["Total"] >= 500]
return filtered[features].dropna()
```

Method roles:

| Method | Result in the implementation |
|---|---|
| `pd.read_csv(file_name)` | Reads the CSV into a DataFrame. |
| `source[source["Total"] >= 500]` | Keeps rows where the Boolean condition is `True`. |
| `filtered[features]` | Selects columns in the listed order. |
| `.dropna()` | Removes a row if any selected feature is missing. |

The column selection happens before `dropna`, so missing values in unselected
columns do not remove a row.

### 4.2 Copy before changing categorical columns

**Used in:** Lab 2 Task 1, `pandas_to_numpy_features`.

```python
df_cp = df.copy()
df_cp["Month"] = df_cp["Month"].map(MONTH)
df_cp["VisitorType"] = df_cp["VisitorType"].map(VTYPE)
df_cp["Weekend"] = df_cp["Weekend"].astype(int)
return np.array(df_cp)
```

| Method | Result in the implementation |
|---|---|
| `DataFrame.copy()` | Creates an independent DataFrame for the converted values. |
| `Series.map(mapping)` | Replaces each category using a dictionary. |
| `Series.astype(int)` | Converts Boolean `Weekend` values to `0` and `1`. |
| `np.array(df_cp)` | Converts the processed DataFrame to an `(n, d)` NumPy array. |
| `np.array(...).flatten()` | Converts the one-column revenue result to an `(n,)` label vector. |

The `copy()` matters because the next assignment writes into the copied
DataFrame. The original input DataFrame is left unchanged.

### 4.3 pandas-to-NumPy shape boundary

The implemented functions cross the pandas/NumPy boundary in two ways:

```python
# Lab 2, feature table
X = np.array(processed_dataframe)  # (n, d)

# Lab 2, one-column label table
y = np.array(label_dataframe.astype(int)).flatten()  # (n, 1) -> (n,)
```

Store features as a two-dimensional array and labels as a one-dimensional
array. A `(n, 1)` label column has the wrong shape for indexing such as
`mean[y]`.

### 4.4 pandas method summary

| Method | Purpose in the completed code |
|---|---|
| `pd.read_csv` | Load task data |
| `df.copy` | Avoid modifying the input DataFrame |
| `Series.map` | Convert named categories to integer codes |
| `Series.astype` / `DataFrame.astype` | Convert Boolean or tabular values to integers |
| `df[boolean_series]` | Filter rows |
| `df[feature_list]` | Select columns in a specified order |
| `DataFrame.dropna` | Remove rows with missing selected values |

No `groupby`, `merge`, `pivot`, or time-series pandas methods are part of the
implemented TODO catalogue.

## 5. NumPy counting and grouping

`np.bincount` is the central numerical workhorse in the completed Lab 2 code.
It accepts a one-dimensional array of nonnegative integers and returns counts
indexed by those integers.

### 5.1 Class counts and priors

**Used in:** Lab 2 Task 2.

```python
counts = np.bincount(y.flatten())
priors = counts / y.size
```

For labels `[0, 1, 0, 1, 0]`:

```text
counts -> [3, 2]
priors -> [0.6, 0.4]
```

Shapes:

```text
y          (n,)
counts     (C,)
priors     (C,)
```

This works for the lab because the class labels are contiguous integers
`0, ..., C - 1`.

### 5.2 Weighted counts as grouped sums

**Used in:** Lab 2 Task 4, `num_likelihood`.

```python
class_counts = np.bincount(y)
class_sums = np.bincount(y, weights=x)
group_means = class_sums / class_counts
```

`weights` changes each observation's contribution from `1` to its feature
value. The result is the sum of `x` within each class. Dividing by the class
counts gives grouped means.

The variance follows the same pattern:

```python
deviations = x - group_means[y]
group_squared_deviation = np.bincount(y, weights=deviations**2)
group_variance = group_squared_deviation / class_counts
```

Key shape step:

```text
group_means       (C,)
y                 (n,)
group_means[y]    (n,)
deviations         (n,)
```

`group_means[y]` uses `y` as integer indices, giving every training sample the
mean of its own class.

### 5.3 Combining two integer columns into one bin index

**Used in:** Lab 2 Task 3, `cat_likelihood`.

The implemented code turns a `(category, class)` pair into one integer:

```python
bins = X_train_feature * number_of_classes + y_train
joint_counts = np.bincount(
    bins,
    minlength=number_of_categories * number_of_classes,
).reshape(number_of_categories, number_of_classes)
```

If there are three categories and two classes:

```text
(category, class) -> bin
(0, 0)             -> 0 * 2 + 0 = 0
(0, 1)             -> 0 * 2 + 1 = 1
(1, 0)             -> 1 * 2 + 0 = 2
(2, 1)             -> 2 * 2 + 1 = 5
```

After `reshape(K, C)`, row `k` contains counts for category `k` in every class.

`minlength` preserves bins that have zero observations. Without it, trailing
all-zero categories could be absent from the result.

The category range in the implemented code accounts for both training and test
data:

```python
num_categories = int(
    max(X_train_feature.max(), X_test_feature.max()) + 1
)
```

This prevents an `IndexError` when a test category is larger than every
training category.

### 5.4 `np.unique`

**Used in:** Lab 2 Tasks 3–4.

```python
class_values = np.unique(y)
c_size = np.unique(y).size
```

`np.unique` returns sorted distinct values. Its `.size` gives the number of
classes. In Lab 2 it sizes zero arrays and the category-class table.

PA1 also has `self.classes = np.unique(y)`, but that line is part of the
provided skeleton before TODO 5. It is therefore not included in the
implemented-method inventory.

## 6. Lab 2: Naive Bayes methods

### 6.1 Converting features and labels

**Implemented concepts:** `DataFrame.copy`, `Series.map`, `astype`,
`np.array`, `flatten`.

```python
df_cp = df.copy()
df_cp["Month"] = df_cp["Month"].map(MONTH).astype(int)
df_cp["VisitorType"] = df_cp["VisitorType"].map(VTYPE).astype(int)
df_cp["Weekend"] = df_cp["Weekend"].astype(int)
X = np.array(df_cp)
```

Expected result:

```text
X.shape == (n, 6)
y.shape == (n,)
```

The dictionary mapping is explicit, so the integer assigned to each category is
controlled by the code rather than by alphabetical order.

### 6.2 Estimating class priors

**Implemented method:** `np.bincount(y.flatten()) / y.size`.

```text
count per class -> (C,)
divide by n     -> prior per class, still (C,)
```

The output satisfies:

```python
priors.sum() == 1
```

up to ordinary floating-point rounding.

### 6.3 Categorical likelihood with Laplace smoothing

**Implemented methods:** `np.unique`, `np.bincount`, `reshape`,
`np.zeros`, broadcasting, and integer-array lookup.

The implementation builds a `(K, C)` table:

```text
joint_counts[k, c] = number of training rows with category k and class c
class_counts[c]    = number of training rows in class c
```

Laplace smoothing changes them to:

```text
smoothed_count[k, c] = joint_counts[k, c] + alpha
smoothed_denom[c]    = class_counts[c] + alpha * K
```

Then:

```python
prob = smoothed_count / smoothed_denom
```

Broadcasting divides a `(K, C)` table by a `(C,)` denominator. Finally,
`prob[test_codes]` gathers one full likelihood row for each test sample and
returns `(n_test, C)`.

### 6.4 Gaussian likelihood

**Implemented methods:** grouped `np.bincount`, class-mean indexing,
broadcasting, `np.sqrt`, `np.pi`, and `np.exp`.

The Gaussian formula used is:

```text
                1                    (x - mu)^2
p(x; mu, var) = --------------- * exp(- ----------)
                sqrt(2*pi*var)         2*var
```

The implemented shape flow is:

```text
X_test[:, None]              (n_test, 1)
group_mean[None, :]          (1, C)
X_test[:, None] - mean[None, :]
                             (n_test, C)
```

The variance is floored by adding `1e-9` before taking `sqrt`. This prevents a
zero denominator for a constant feature.

The `(n_test, C)` result means row `i`, column `c` is the likelihood of test
value `i` under class `c`.

### 6.5 Combining per-feature likelihoods

**Implemented methods:** list comprehension, `np.array`, `np.log`,
`array.sum(axis=0)`, `np.log` on priors, and `argmax(axis=1)`.

Each feature function returns `(n_test, C)`. Stacking `d` features produces:

```text
np.array(per_feature_likelihoods)  -> (d, n_test, C)
```

Then:

```python
log_joint = np.log(per_feature_likelihoods).sum(axis=0)
log_posterior = log_joint + np.log(priors)
predictions = log_posterior.argmax(axis=1)
```

Shape flow:

```text
sum over feature axis        (d, n_test, C) -> (n_test, C)
add priors                   (n_test, C) + (C,) -> (n_test, C)
argmax over class axis       (n_test, C) -> (n_test,)
```

Working in log space turns a product of likelihoods into a sum of logarithms
and avoids underflow from multiplying many small probabilities.

## 7. Lab 4: K-Means methods

### 7.1 Preparing the feature table

**Implemented methods:** `pd.read_csv`, Boolean DataFrame filtering, ordered
column selection, and `dropna`.

```python
source = pd.read_csv(file_name)
filtered = source[source["Total"] >= 500]
return filtered[features].dropna()
```

This is a method chain, not several copies:

```text
CSV
-> filter rows
-> select columns
-> remove incomplete selected rows
```

### 7.2 Feature-wise z-score standardization

**Implemented methods:** `mean(axis=0)`, `std(axis=0, ddof=1)`, Boolean
replacement, and broadcasting.

```python
mean = X.mean(axis=0)
std = X.std(axis=0, ddof=1)
std += std == 0
standardized = (X - mean[None, :]) / std
```

Shape flow:

```text
mean                         (d,)
mean[None, :]                (1, d)
X - mean[None, :]            (n, d)
divide by std                (n, d)
```

`ddof=1` selects the sample standard deviation. NumPy's default is `ddof=0`
(population standard deviation), so omitting `ddof=1` changes the expected Lab
4 result.

When a feature is constant, `std == 0`. Adding the Boolean result changes each
zero to `1`, leaving that standardized column as all zeros.

### 7.3 Random distinct-row initialization

**Implemented methods:** `np.random.choice` and integer-array row selection.

```python
indices = np.random.choice(X.shape[0], k, replace=False)
centroids = X[indices]
```

`replace=False` makes the chosen row indices distinct. `X[indices]` gathers
those rows and returns `(k, d)`.

The prewritten Lab 4 main section calls `np.random.seed(42)`. That call belongs
to the skeleton rather than the implemented functions; the functions themselves
do not reset the seed.

### 7.4 All pairwise Euclidean distances

**Implemented methods:** `X[:, None, :]`, broadcasting,
`np.linalg.norm(axis=2)`, and `argmin(axis=1)`.

```python
differences = X[:, None, :] - centroids
distances = np.linalg.norm(differences, axis=2)
labels = distances.argmin(axis=1)
```

Shape flow:

```text
X[:, None, :]       (n, 1, d)
centroids           (k, d)
broadcast result    (n, k, d)
norm over d         (n, k)
argmin over k       (n,)
```

`argmin` returns the first minimum, so ties choose the smallest centroid
index.

### 7.5 Updating centroids with Boolean masks

**Implemented methods:** `np.zeros`, `labels == c`, `np.count_nonzero`,
Boolean-column broadcasting, `sum(axis=0)`, division, and indexed assignment.

The completed code follows this pattern for cluster `c`:

```python
mask = labels == c                    # (n,)
count = np.count_nonzero(mask)        # scalar
points = X * mask[:, None]            # (n, d)
mean = points.sum(axis=0) / count     # (d,)
new_centroids[c] = mean
```

`mask[:, None]` changes `(n,)` to `(n, 1)`, so the mask broadcasts across all
features. Multiplication keeps selected rows and turns unselected values into
zero; summing then dividing gives the mean.

PA1 uses the more direct equivalent:

```python
this_class_X = X[class_mask]
mean = this_class_X.mean(axis=0)
```

Both are present in the completed code. Direct Boolean selection avoids
constructing a full Boolean-sized numeric temporary.

### 7.6 Empty-cluster handling

The Lab 4 implementation checks `np.count_nonzero(mask)` before division:

```python
if count == 0:
    new_centroids[c] = X[c % X.shape[0]]
    continue
```

This avoids a zero denominator and follows Lab 4's deterministic replacement
rule.

### 7.7 Convergence check

**Implemented methods:** `np.linalg.norm(axis=1)`, comparison, and `np.all`.

```python
movement = np.linalg.norm(old_centroids - new_centroids, axis=1)
if np.all(movement < 0.01):
    break
```

Shape flow:

```text
old - new       (k, d)
norm axis=1     (k,)
comparison      (k,) Boolean
all(...)        scalar Boolean
```

The implementation keeps the old centroids when the convergence condition is
met, and recomputes labels against the returned centroids before returning.

### 7.8 Inertia with `take_along_axis`

**Implemented methods:** `np.expand_dims`, `np.square`,
`sum(axis=2)`, `np.take_along_axis`, and `.sum()`.

```python
X_expanded = np.expand_dims(X, axis=1)                  # (n, 1, d)
squared_distances = np.square(X_expanded - centroids).sum(axis=2)
                                                          # (n, k)
assigned = np.take_along_axis(
    squared_distances,
    labels[:, None],
    axis=1,
)                                                         # (n, 1)
inertia = assigned.sum()                                  # scalar
```

`labels[:, None]` is required because `take_along_axis` needs the same number
of dimensions as the source array. For every row, it extracts the squared
distance belonging to that row's assigned cluster.

## 8. PA1 TODO 1–8: scaling, clustering, GNB, and explanation

The PA1 catalogue stops at TODO 8 because TODO 9 onward remains unfinished in
the task notebook/script.

### 8.1 Split-aware standardization

**Implemented in:** PA1 TODO 1.

**Implemented methods:** `mean(axis=0)`, `std(axis=0)`, scalar epsilon, and
broadcasting.

```python
mean = X_train.mean(axis=0)
sd = X_train.std(axis=0) + 1e-9
X_train_scaled = (X_train - mean) / sd
X_val_scaled = (X_val - mean) / sd
X_test_scaled = (X_test - mean) / sd
```

Only training data determines `mean` and `sd`. Validation and test data reuse
those values, preventing information from other splits leaking into training.

PA1 uses `+1e-9` rather than Lab 4's explicit zero replacement. The catalogue
records both as distinct completed patterns.

### 8.2 K-Means labels through `np.broadcast_to`

**Implemented in:** PA1 TODO 2.

**Implemented methods:** `np.broadcast_to`, `np.square`, `sum(axis=2)`,
`np.sqrt`, and `argmin(axis=1)`.

```python
X_k_extend = np.broadcast_to(
    X[:, None, :],
    (n, self.k, n_features),
)
distance_k = np.sqrt(
    np.square(X_k_extend - self.centroids).sum(axis=2)
)
self.labels = distance_k.argmin(axis=1)
```

This produces the same shapes as Lab 4:

```text
X_k_extend          (n, k, d)
centroids           (k, d)
square difference   (n, k, d)
sum axis=2          (n, k)
nearest label       (n,)
```

`broadcast_to` presents the sample axis as repeated views; subtracting the
centroids creates the full `(n, k, d)` intermediate needed for the calculation.

### 8.3 K-Means centroid update in PA1

**Implemented in:** PA1 TODO 3.

**Implemented methods:** `labels == c`, `np.count_nonzero`, Boolean mask
expansion, elementwise multiplication, `sum(axis=0)`, division, and row
assignment.

```python
mask = self.labels == c
count = np.count_nonzero(mask)
if count == 0:
    continue
points = X * mask[:, None]
new_centroids[c] = points.sum(axis=0) / count
```

The arithmetic pattern is identical to Lab 4. The empty-cluster behaviour is
not identical: because `new_centroids` was initialized with zeros, `continue`
leaves a zero centroid in PA1, while the Lab 4 completed function explicitly
substitutes a data row. This difference is recorded in
[§10](#10-current-code-notes-and-common-mistakes).

### 8.4 WCSS in PA1

**Implemented in:** PA1 TODO 4.

**Implemented methods:** `np.broadcast_to`, `np.square`,
`sum(axis=2)`, `np.take_along_axis`, and `.sum()`.

```python
X_k_extend = np.broadcast_to(
    X[:, None, :],
    (X.shape[0], self.k, X.shape[1]),
)
dist_square_k = np.square(
    X_k_extend - self.centroids
).sum(axis=2)
wcss = np.take_along_axis(
    dist_square_k,
    self.labels[:, None],
    axis=1,
).sum()
```

This is the same assigned-distance extraction used by Lab 4 inertia, with
`broadcast_to` supplying the three-dimensional comparison.

### 8.5 GNB grouped means, variances, and log priors

**Implemented in:** PA1 TODO 5.

**Implemented methods:** looping over classes, Boolean masks,
`np.all(..., axis=1)`, `mean(axis=0)`, `std(axis=0)`, `np.square`, `np.log`,
and assignment into prewritten arrays.

The completed class-statistics pattern is:

```python
for i in range(self.classes.size):
    this_class = self.classes[i]
    class_mask = (y == this_class)[:, None]
    class_rows = X[np.all(class_mask, axis=1)]

    self.mean[i, :] = class_rows.mean(axis=0)
    self.var[i, :] = np.square(class_rows.std(axis=0))
    self.log_priors[i] = np.log(class_rows.shape[0] / X.shape[0])
```

Shape flow:

```text
y == this_class              (n,)
(y == this_class)[:, None]   (n, 1)
np.all(..., axis=1)          (n,)
class_rows                   (n_class, d)
mean(axis=0)                 (d,)
```

The `np.all` is redundant for a one-column mask because the mask is already
one-dimensional. The simpler equivalent is:

```python
class_rows = X[y == this_class]
```

The completed implementation uses squared `std(axis=0)` as variance. Its prior
is the class frequency, converted to log space.

Output shapes:

```text
self.mean           (C, d)
self.var            (C, d)
self.log_priors     (C,)
```

The axis matters: `axis=0` reduces over rows/classes and keeps one statistic
for each feature.

### 8.6 GNB prediction

**Implemented in:** PA1 TODO 6.

The TODO body receives `(n, C)` log scores from the prewritten `_log_scores`
helper:

```python
scores = self._log_scores(X)
```

The skeleton already maps the winning score positions back to
`self.classes`. Its `np.argmax` call is therefore not catalogued as a
user-added NumPy method. No new NumPy operation belongs to the completed TODO 6
body; the added work is the call that supplies `scores` to the existing return
expression.

### 8.7 Feature discriminativeness

**Implemented in:** PA1 TODO 7.

**Implemented methods:** `Array.std(axis=0)`, `np.square`, and
`Array.mean(axis=0)`. The skeleton already contains the final division by
`within_class_variance + 1e-9`, so that operation is not attributed to the
TODO.

```python
between_class_variance = np.square(gnb.mean.std(axis=0))
within_class_variance = gnb.var.mean(axis=0)
```

Shape flow:

```text
gnb.mean                   (C, d)
gnb.mean.std(axis=0)       (d,)
np.square(...)             (d,)
gnb.var.mean(axis=0)       (d,)
```

Here `axis=0` is correct because the first axis contains classes. The first
operation measures variation of class means for each feature; the second
measures average within-class variance for each feature.

### 8.8 Ranking and one-prediction explanation

**Implemented in:** PA1 TODO 8.

**Implemented methods:** `ndim`, `x[None, :]`, class prediction,
`predicted_class` indexing, `np.log`, `np.pi`, `np.square`, `flatten`,
`np.argsort`, reverse slicing, and index-array selection.

The completed code normalizes a single sample to one row:

```python
if x.ndim < 2:
    x = x[None, :]       # (d,) -> (1, d)
```

It predicts the class, then selects that class's parameters:

```python
predicted_class = gnb.predict(x)[0]
variance = gnb.var[predicted_class, :] + 1e-9
mean = gnb.mean[predicted_class, :]
```

It computes one evidence value per feature and flattens `(1, d)` to `(d,)`.
Descending rank uses:

```python
order = np.argsort(evidence)[::-1]
ranked_names = feature_names[order]
ranked_evidence = evidence[order]
```

`np.argsort` is ascending; reversing puts the largest evidence first.
`feature_names[order]` and `evidence[order]` apply the same permutation.

The expression currently written in the TODO implementation is:

```python
np.square((x - mean) / (2 * variance))
```

That squares the whole fraction. The evidence formula printed above TODO 8 is
instead `(x - mean)^2 / (2 * variance)`. The current catalogue records the
method used by the implementation and flags this mismatch in
[§10](#10-current-code-notes-and-common-mistakes).

## 9. Reusable recipes

These recipes use only methods that appeared in the completed TODOs.

### 9.1 Column statistics with broadcasting

```python
mean = X.mean(axis=0)       # (d,)
std = X.std(axis=0)         # (d,)
scaled = (X - mean) / std   # (n, d)
```

Use `mean[None, :]` when the extra size-1 axis makes the intended broadcast
clearer.

### 9.2 Class-grouped mean with `bincount`

```python
counts = np.bincount(y)
sums = np.bincount(y, weights=x)
group_means = sums / counts
```

Assumptions: integer nonnegative labels and contiguous class codes.

### 9.3 Class-grouped mean with a Boolean mask

```python
mask = y == class_label
group_mean = X[mask].mean(axis=0)
```

This pattern is in PA1 TODO 5 and is more direct than multiplying the full
matrix by a Boolean column.

### 9.4 Sample-to-centroid distance matrix

```python
differences = X[:, None, :] - centroids
distances = np.linalg.norm(differences, axis=2)
labels = distances.argmin(axis=1)
```

The result is `(n, k)`: one distance for every sample-centroid pair.

### 9.5 Select each row's assigned distance

```python
assigned = np.take_along_axis(
    distances,
    labels[:, None],
    axis=1,
)
```

`distances` is `(n, k)` and `labels[:, None]` is `(n, 1)`. The result is
`(n, 1)`.

### 9.6 Grouped variance from deviations

```python
means = class_sums / class_counts
deviations = x - means[y]
variances = np.bincount(y, weights=deviations**2) / class_counts
```

The indexed lookup `means[y]` is the shape bridge between per-class statistics
and per-sample deviations.

### 9.7 Descending rank

```python
order = np.argsort(values)[::-1]
ranked_values = values[order]
ranked_names = names[order]
```

Reverse only after sorting; reversing the original array would not carry the
matching names.

## 10. Current-code notes and common mistakes

These notes compare the completed code against the task contracts and the
recurring shape model. They do not introduce methods that were absent from the
implementations.

### 10.1 `bincount` is not a collection of labels

`np.bincount` returns counts indexed by integer values. It is not
`np.unique(..., return_counts=True)` and it silently leaves no room for a
missing trailing category unless `minlength` is supplied.

Lab 2 uses both `minlength` and a category range based on both train and test
data, which covers the intended cases.

### 10.2 Boolean multiplication versus Boolean selection

The Lab 4 update uses:

```python
points = X * (labels == c)[:, None]
mean = points.sum(axis=0) / count
```

The PA1 update uses:

```python
points = X[labels == c]
mean = points.mean(axis=0)
```

Both are valid completed patterns. Direct selection is shorter and avoids
making a numeric copy of the full matrix, but it requires a nonzero row count.

### 10.3 `ddof` changes standard deviation

```python
X.std(axis=0)          # population standard deviation, ddof=0
X.std(axis=0, ddof=1)  # sample standard deviation
```

Lab 4 explicitly requires `ddof=1`. PA1's completed scaling uses the NumPy
default. Keep the lab-specific choice tied to the task rather than assuming
that one convention applies everywhere.

### 10.4 Zero variance

Two completed handling styles are present:

```python
# Lab 4: replace zero standard deviation with one
std += std == 0

# PA1: add a tiny positive value
sd = X_train.std(axis=0) + 1e-9
```

Either prevents division by zero, but they produce different numerical results.

### 10.5 Axis selection for class statistics

For a shape `(C, d)` array:

```python
array.mean(axis=0)  # across classes -> (d,)
array.mean(axis=1)  # across features -> (C,)
```

PA1 TODO 7 needs one score per feature, so `axis=0` is the relevant reduction.

### 10.6 `take_along_axis` index shape

With `distances.shape == (n, k)`:

```python
labels              # (n,)
labels[:, None]     # (n, 1)
```

The latter is the form required for `axis=1`. Passing `(n,)` directly is not the
same index shape.

### 10.7 Flatten labels after a one-column conversion

`np.array(df.astype(int))` produces a two-dimensional `(n, 1)` array. The
completed conversion applies `.flatten()` to obtain `(n,)`, which is the shape
used by `np.bincount` and class indexing.

### 10.8 Skeleton methods are not user methods

The skeleton diff identifies these prewritten PA1 operations around the
completed TODO blocks. They are deliberately omitted from the method crosswalk:

| Location | Scaffold-owned operations, not attributed to the user |
|---|---|
| `KMeansClustering.fit` preamble | `np.random.seed`, `np.random.choice`, array `.copy`, `np.zeros`, `np.allclose` |
| `GaussianNaiveBayes.fit` preamble | `np.unique`, parameter allocation with `np.zeros` |
| `GaussianNaiveBayes._log_scores` | The entire scoring helper, including its NumPy arithmetic |
| `GaussianNaiveBayes.predict` existing return | `np.argmax` and class-array lookup |
| `GaussianNaiveBayes.predict_proba` | `np.max`, `np.exp`, and `np.sum` |
| Feature-ranking display | `np.argsort` used to print the already-computed scores |
| `explain_prediction` existing return | Returning the variables supplied by the TODO body |

The method crosswalk includes only lines added inside the TODO blanks, even when
the same operation also happens to appear elsewhere in the scaffold.

### 10.9 Current PA1 implementation notes

1. PA1 TODO 3 says an empty cluster should keep its previous centroid, but the
   completed code initializes `new_centroids` to zero and `continue`s on an
   empty cluster. That leaves a zero vector, not the previous centroid.
2. PA1 TODO 8 squares `(x - mean) / (2 * variance)` instead of computing
   `(x - mean)^2 / (2 * variance)`.
3. PA1 TODO 5's `np.all` mask is equivalent to the simpler one-dimensional
   `y == this_class`; it is not wrong for the current shape.
4. PA1 `standardize_features` uses population standard deviation because it
   omits `ddof`; Lab 4 explicitly uses sample standard deviation.

## 11. Completed-TODO crosswalk

| Source | Completed TODO | NumPy / pandas methods actually used |
|---|---|---|
| Lab 2 | Convert features and revenue | `DataFrame.copy`, `Series.map`, `Series.astype`, `np.array`, `flatten` |
| Lab 2 | Estimate priors | `flatten`, `np.bincount`, `.size`, division |
| Lab 2 | Categorical likelihood | `np.unique`, `np.zeros`, `.max`, `np.bincount`, `reshape`, integer indexing |
| Lab 2 | Numerical likelihood | `flatten`, `astype`, `np.bincount` with weights, `np.sqrt`, `np.pi`, `np.exp`, two-axis broadcasting |
| Lab 2 | Predict | `np.array`, `np.log`, `sum(axis=0)`, `argmax(axis=1)` |
| Lab 4 | Task 1: load/filter | `pd.read_csv`, Boolean row filter, ordered column selection, `dropna` |
| Lab 4 | Task 2: standardize | `.shape`, `.mean(axis=0)`, `.std(axis=0, ddof=1)`, Boolean replacement, broadcasting |
| Lab 4 | Task 3.1: initialize | `np.random.choice`, integer row indexing |
| Lab 4 | Task 3.2: assign | `None` axis insertion, broadcasting, `np.linalg.norm(axis=2)`, `argmin(axis=1)` |
| Lab 4 | Task 3.3: update | `np.zeros`, Boolean mask, `np.count_nonzero`, mask broadcasting, `sum(axis=0)`, centroid assignment |
| Lab 4 | Task 3.4: loop | `np.linalg.norm(axis=1)`, comparison, `np.all`, repeated assignment calls |
| Lab 4 | Task 4: inertia | `np.expand_dims`, `np.square`, `sum(axis=2)`, `np.take_along_axis`, scalar sum |
| PA1 | TODO 1 | `.mean(axis=0)`, `.std(axis=0)`, epsilon, broadcasting |
| PA1 | TODO 2 | `np.broadcast_to`, `np.square`, `sum(axis=2)`, `np.sqrt`, `argmin(axis=1)`, `np.zeros` |
| PA1 | TODO 3 | Boolean mask, `np.count_nonzero`, `[:, None]`, elementwise multiplication, `sum(axis=0)`, indexed assignment |
| PA1 | TODO 4 | `np.broadcast_to`, `np.square`, `sum(axis=2)`, `np.take_along_axis`, `.sum()` |
| PA1 | TODO 5 | Class loop, Boolean mask, `np.all(axis=1)`, `mean(axis=0)`, `std(axis=0)`, `np.square`, `np.log` |
| PA1 | TODO 6 | Adds the call to existing `_log_scores`; the skeleton's `argmax` lookup is excluded |
| PA1 | TODO 7 | `std(axis=0)`, `np.square`, `mean(axis=0)`; the skeleton's final division is excluded |
| PA1 | TODO 8 | `ndim`, `x[None, :]`, class indexing, `np.log`, `np.pi`, `np.square`, `flatten`, `np.argsort`, reverse slicing, index-array lookup |

### Excluded from the catalogue

| Source | Excluded part | Reason |
|---|---|---|
| Lab 4 | Task 5 optional K choice | Not implemented |
| PA1 | TODO 9 | Distance calculations still contain an unfinished branch |
| PA1 | TODO 10 | Majority-vote branch still contains `pass` |
| PA1 | TODO 11 | Validation-tuning branch still contains `pass` |
| PA1 | TODO 12 | Threshold-tuning branch still contains `pass` |
| PA1 | TODO 13 | Fallback mask and prediction copy are not implemented |
| PA1 | TODO 14 | Precision, recall, F1, and macro-F1 are not implemented |

## 12. Revision checklist

Before submitting or extending a task implementation, check:

- Every feature matrix is `(n_samples, n_features)`.
- Every label vector is `(n_samples,)`, not `(n_samples, 1)`.
- A reduction's `axis` is the axis you want to remove.
- A broadcast difference right-aligns shapes and only stretches size-1 axes.
- Boolean masks use `&`, `|`, and `~`.
- A selected mask has shape `(n,)`; `[:, None]` is used only when a column axis
  is needed.
- `np.bincount` receives nonnegative integer values, and `minlength` covers
  required zero-count bins.
- Grouped variance is computed from deviations from the matching group mean.
- `ddof` is chosen deliberately rather than left implicit.
- Zero standard deviation, zero variance, and empty clusters are handled before
  division or logarithm.
- `argmax`/`argmin` operate over the intended candidate axis.
- `take_along_axis` receives an index array with the same number of dimensions
  as the data array.
- Inertia selects each row's assigned squared distance before summing.
- `np.argsort(...)[::-1]` reverses the permutation, not just the values.
- A categorical lookup table covers category values seen in both training and
  test data.
- Methods copied from scaffold code are kept distinct from methods implemented
  inside a TODO.
