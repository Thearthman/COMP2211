# COMP 2211 NumPy and pandas Method Database

Compact lookup for methods used in completed TODO code. See the
[detailed catalogue](COMP2211_numpy_pandas_catalogue.md) when explanation is
needed.

**Scope:** Lab 2 all tasks, Lab 4 Tasks 1–4, PA1 TODOs 1–8.

**Legend:** `L2` = Lab 2, `L4` = Lab 4, `T` = task, `PA1` = TODO number.
Methods present only in scaffold code are excluded.

## Current Focus

- Topic: Lab 8 CNN pet foreground segmentation
- Goal: understand the encoder-decoder flow, spatial resolution versus feature channels, and how the network produces a binary segmentation mask
- Last updated: 2026-11-17

## Concepts

### Upsampling in segmentation decoders

- Status: introduced
- Meaning: increase a feature map's height and width so a CNN can return from pooled feature maps to a full-resolution prediction.
- Seen in: Lab 8 `UpSampling2D` in the decoder before the final `128 x 128` mask.
- Related methods: `MaxPooling2D`, `Conv2D`, nearest-neighbor interpolation, bilinear interpolation, transposed convolution.
- Minimal example: default `UpSampling2D(size=(2, 2))` maps `[[1, 2], [3, 4]]` to a `4 x 4` grid, repeating each value in a `2 x 2` block.
- Misconception to watch: `size=(2, 2)` multiplies height and width by two rather than adding two; channels stay unchanged. Keras's default nearest-neighbor upsampling also performs no interpolation or learned calculation; it copies values, and later convolutions do the learned refinement.

### Binary target versus probability mask

- Status: introduced
- Meaning: the training target contains hard values in `{0, 1}`, while the final sigmoid layer produces a foreground probability in `[0, 1]`; thresholding converts that probability map into a binary prediction.
- Seen in: Lab 8 mask preprocessing, `BinaryCrossentropy`, and foreground IoU thresholding.
- Related methods: sigmoid activation, binary cross-entropy, thresholding, IoU.
- Minimal example: sigmoid probabilities `[0.1, 0.8]` become `[0, 1]` with a `0.5` threshold.
- Misconception to watch: a sigmoid does not normally output exact `0` or `1`; the raw network output is a continuous probability map.

### Feature channels versus spatial dimensions

- Status: introduced
- Meaning: a tensor `(batch, H, W, C)` has `H x W` spatial positions and `C` feature channels; a `Conv2D` layer's `filters` argument sets the next tensor's channel count.
- Seen in: Lab 8 encoder-decoder feature maps and the final one-filter segmentation convolution.
- Related methods: `Conv2D`, `UpSampling2D`, `MaxPooling2D`, the final `1 x 1` convolution with sigmoid.
- Minimal example: `(8, 8, 8, 16) --Conv2D(32, 3)--> (8, 8, 8, 32)`; each new channel combines all 16 input channels over a local window.
- Misconception to watch: filter outputs are connected feature channels, not independent candidate images. The final layer controls mask classes through its number of filters, not through its spatial size.

### Convolution padding and shape control

- Status: introduced
- Meaning: `padding="same"` adds border values, usually zeros, so a stride-one convolution preserves height and width; `padding="valid"` adds nothing and keeps only positions where the whole kernel fits.
- Seen in: shape tracking for Lab 8 encoder and decoder `Conv2D` layers.
- Related methods: `Conv2D` `kernel_size`, `stride`, and `filters`; `MaxPooling2D`; `UpSampling2D`.
- Minimal example: `(128, 128, 3) --Conv2D(32, 3, padding="same")--> (128, 128, 32)`, while `padding="valid"` gives `(126, 126, 32)`.
- Misconception to watch: padding controls spatial height and width, while `filters` controls output channels. Keras `Conv2D` defaults to `padding="valid"`, not `"same"`.

### Speaker-disjoint validation

- Status: practicing
- Meaning: keep every recording from a speaker on one side of a train/validation split.
- Seen in: Lab 6 `GroupShuffleSplit(..., groups=speakers)` experiments.
- Related methods: required `train_test_split`, `GroupShuffleSplit`, macro F1.
- Minimal example: `next(GroupShuffleSplit(test_size=1/9).split(X, y, groups=speakers))`
- Misconception to watch: a random split can look strong when the same speakers occur in both partitions.

### Macro F1

- Status: comfortable
- Meaning: compute F1 per class, then take the unweighted mean so every command contributes equally.
- Seen in: Lab 6 `MacroF1Score` and candidate comparisons.
- Related methods: confusion matrix, precision, recall.
- Minimal example: `f1_score(y_true, y_pred, average="macro")`
- Misconception to watch: a strong mean can conceal a weak worst class or worst speaker fold.

### Waveform augmentation

- Status: introduced
- Meaning: train on deterministic shifted, gain-scaled, and noisy copies before extracting the unchanged MFCC features.
- Seen in: Lab 6 candidate scripts and augmented feature cache.
- Related methods: MFCC extraction, regularization, speaker-disjoint validation.
- Minimal example: extract MFCCs from both the original waveform and a 10 dB SNR noisy copy.
- Misconception to watch: augmentation must be selected by speaker-held-out validation, not the leaky random split.

### Probability-averaged MLP ensemble

- Status: introduced
- Meaning: average the softmax outputs of independently initialized or differently regularized MLP branches.
- Seen in: Lab 6 three-branch candidate.
- Related methods: dropout, weight averaging, macro F1.
- Minimal example: `mean([branch(x, training=False) for branch in branches])`
- Misconception to watch: ensemble gains are only credible when every member is evaluated on the same speaker-disjoint folds.

## Next Reviews

- [ ] Predict the shape and values produced by default `UpSampling2D((2, 2))` on a small `2 x 2` feature map.
- [ ] Explain why a binary segmentation target still uses sigmoid probabilities and thresholding instead of exact `0`/`1` activations.
- [ ] Track a decoder feature map from `32 x 32 x 64` to `64 x 64 x 1` and identify which layer changes resolution and which changes channels.
- [ ] Compare the output shapes of `Conv2D(16, 3)` with `padding="same"` and `padding="valid"` for a `32 x 32 x 64` input.
- [ ] Compare a single MLP and a three-member ensemble by worst-fold macro F1 on speaker-disjoint folds.
- [ ] Report per-class F1 for `down`, `go`, `no`, and `right`, not only the mean.

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
