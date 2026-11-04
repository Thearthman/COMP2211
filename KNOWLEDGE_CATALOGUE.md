# COMP 2211 NumPy and pandas Method Database

Compact lookup for methods used in completed TODO code. See the
[detailed catalogue](COMP2211_numpy_pandas_catalogue.md) when explanation is
needed.

**Scope:** Lab 2 all tasks, Lab 4 Tasks 1–4, PA1 TODOs 1–8.

**Legend:** `L2` = Lab 2, `L4` = Lab 4, `T` = task, `PA1` = TODO number.
Methods present only in scaffold code are excluded.

## pandas

| API | Source | Used pattern |
|---|---|---|
| `pd.read_csv` | L4 T1 | `source = pd.read_csv(path)` |
| `DataFrame.copy` | L2 T1 | `df_cp = df.copy()` |
| `Series.map` | L2 T1 | `df["Month"].map(MONTH)` |
| `Series.astype` | L2 T1–2 | `df["Weekend"].astype(int)` |
| `DataFrame.astype` | L2 T1 | `df.astype(int)` |
| `df[boolean_series]` | L4 T1 | `source[source["Total"] >= 500]` |
| `df[column_list]` | L4 T1 | `filtered[features]` |
| `DataFrame.dropna` | L4 T1 | `filtered[features].dropna()` |

## NumPy arrays and counting

| API | Source | Used pattern |
|---|---|---|
| `np.array` | L2 T1, T5 | DataFrame → `(n, d)` array; list of arrays → `(d, n, C)` |
| `array.flatten()` | L2 T1–2, PA1 8 | `(n, 1) → (n,)`; `(1, d) → (d,)` |
| `array.reshape()` | L2 T3 | flat bincount result → `(K, C)` |
| `np.unique` | L2 T3–4 | `.size` for class/category count |
| `np.zeros` | L2 T3–4, L4 T3.3, PA1 2 | allocate result arrays |
| `np.bincount` | L2 T2–4 | counts, `weights=`, `minlength=` |
| `bincount` joint bin | L2 T3 | `category * C + class` |

## Shapes and indexing

| API / syntax | Source | Used pattern |
|---|---|---|
| `array.shape` | All | `(n, d)`, `(n,)`, `(K, C)` |
| `array.size` | L2 T2–3 | denominator `n` or empty-test check |
| `array.ndim` | PA1 8 | normalize `(d,)` input to `(1, d)` |
| `array[:, i]` | L2 T5 | select feature column → `(n,)` |
| `array[:, None]` | L2 T4, L4 T3.3, PA1 2–3 | insert column axis |
| `array[None, :]` | L2 T4, PA1 8 | insert row axis |
| `array[indices]` | L4 T3.1 | select random centroid rows |
| `array[boolean_mask]` | L4 T1, PA1 5 | select rows |
| `mask[:, None]` | L4 T3.3, PA1 2–3 | broadcast mask over features |
| `lookup[labels]` | L2 T4 | class value per sample |
| `prob[category_codes]` | L2 T3 | likelihood rows `(n_test, C)` |
| `array[order]` | PA1 8 | reorder names/evidence |
| `array[::-1]` | PA1 8 | reverse ranking |
| `array[index] = value` | L4 T3.3, PA1 3 | assign one centroid |

## Aggregation and broadcasting

| API | Source | Used pattern |
|---|---|---|
| `array.mean(axis=0)` | L4 T2, PA1 1, 5, 7 | column/class-feature means |
| `array.std(axis=0)` | L4 T2, PA1 1, 5 | column/class-feature standard deviation |
| `array.std(axis=0, ddof=1)` | L4 T2 | sample standard deviation |
| `array.sum(axis=0)` | L2 T5, L4 T3.3 | sum feature values; cluster totals |
| `array.sum(axis=2)` | L4 T4, PA1 2, 4 | reduce feature axis of `(n, k, d)` |
| `array.sum()` | L4 T4, PA1 4 | scalar total |
| `axis=0` broadcast | L4 T2, PA1 1 | `(n, d) - (d,)` |
| `(n, 1) - (1, C)` | L2 T4 | `(n, C)` likelihood table |
| `(n, 1, d) - (k, d)` | L4 T3.2, T4 | `(n, k, d)` comparisons |

## Math and scoring

| API | Source | Used pattern |
|---|---|---|
| `np.square` | L4 T4, PA1 2, 4–8 | squared deviations/distances |
| `np.sqrt` | L2 T4, PA1 2 | standard deviation denominator; Euclidean norm |
| `np.exp` | L2 T4 | Gaussian likelihood |
| `np.log` | L2 T5, PA1 5, 8 | log priors/likelihood/evidence |
| `np.pi` | L2 T4, PA1 8 | Gaussian log density |
| `np.argmax` | L2 T5 | class winner over `axis=1` |
| `np.argmin` | L4 T3.2 | nearest centroid over `axis=1` |
| `np.argsort` | PA1 8 | feature ranking |
| `np.all` | L4 T3.4, PA1 5 | convergence; Boolean row reduction |
| `np.count_nonzero` | L4 T3.3, PA1 3 | cluster population |

## Vectorized algorithms

| API | Source | Used pattern |
|---|---|---|
| `np.random.choice` | L4 T3.1 | distinct centroid indices, `replace=False` |
| `np.linalg.norm` | L4 T3.2, T3.4 | sample-centroid distance; centroid movement |
| `np.expand_dims` | L4 T4 | `(n, d) → (n, 1, d)` |
| `np.take_along_axis` | L4 T4, PA1 4 | select each row's assigned distance |
| `np.broadcast_to` | PA1 2, 4 | expand samples to `(n, k, d)` |

## Learned patterns

| ID | Pattern | Source |
|---|---|---|
| P1 | pandas categorical mapping → NumPy matrix | L2 T1 |
| P2 | class counts and priors with `bincount` | L2 T2 |
| P3 | joint category/class counts with one composite bin index | L2 T3 |
| P4 | grouped sums, means, and variance with weighted `bincount` | L2 T4 |
| P5 | multiply likelihoods as a sum of logs; `argmax` for classes | L2 T5 |
| P6 | z-score with feature-axis statistics and broadcasting | L4 T2, PA1 1 |
| P7 | all sample-centroid distances with `(n, k, d)` broadcasting | L4 T3.2, PA1 2 |
| P8 | centroid means from Boolean row masks | L4 T3.3, PA1 3 |
| P9 | WCSS by selecting each row's assigned squared distance | L4 T4, PA1 4 |
| P10 | GNB class means, variances, and priors from class masks | PA1 5 |
| P11 | feature discriminativeness across the class axis | PA1 7 |
| P12 | descending feature ranking by permutation | PA1 8 |

## Excluded

- PA1 TODO 9–14: unfinished.
- PA1 scaffold: `_log_scores`, `predict_proba`, existing `argmax` return,
  random-init preliminaries, distance formula, threshold helpers, and evaluation
  scaffold.
- Lab 4 Task 5 optional K choice.
