# Study Catalogue

## Current Focus
- Topic: PA1 end-to-end traffic-sign classification pipeline
- Goal: Distinguish preprocessing, exploration, classification, routing, and evaluation
- Last updated: 2026-10-24

## Concepts

### PA1 component roles
- Status: introduced
- Meaning: GNB is the primary supervised classifier, KNN is an alternative local classifier, and the hybrid routes ambiguous GNB cases to KNN. K-Means only explores unlabeled structure; feature analysis explains GNB; evaluation scores predictions.
- Seen in: PA1 Parts 0–7.
- Related methods: train/validation/test splits; standardization; GNB; KNN; confidence margin; confusion matrix.
- Minimal example: fit GNB and KNN on train data, select K and the margin threshold on validation data, then compare all three classifiers once on test data.
- Misconception to watch: Treating K-Means cluster IDs as traffic-sign class predictions or using the test set to tune K or the threshold.

### Continuous inputs versus model assumptions
- Status: introduced
- Meaning: Both GNB and KNN accept the same continuous feature space. GNB models each class with continuous Gaussian feature distributions; KNN instead relies on a numeric distance metric and nearby stored examples.
- Seen in: PA1 Parts 3 and 5.
- Related methods: Gaussian likelihood; Euclidean distance; parametric versus non-parametric models.
- Minimal example: GNB learns means and variances; KNN stores training rows and compares distances.
- Misconception to watch: Saying GNB is preferred because it accepts continuous features; KNN also does. The relevant trade-off is probabilistic and interpretable global modelling versus local non-parametric prediction.

### Sample-feature matrices
- Status: comfortable
- Meaning: A feature array has shape `(n_samples, n_features)`.
- Seen in: PA1 dataset split; all three real-data splits have 10 feature columns.
- Related methods: `array.shape`; row and column indexing.
- Minimal example: `(2239, 10)` means 2,239 samples with 10 features each.
- Misconception to watch: Treating the sample count as the feature count.

### Pairwise sample dot products
- Status: practicing
- Meaning: `X_test @ X_train.T` compares every test row with every training row.
- Seen in: PA1 TODO-9 `compute_distances`.
- Related methods: `np.dot`; `.T`; matrix multiplication.
- Minimal example: `(m, d) @ (d, n) -> (m, n)`.
- Misconception to watch: Forgetting that fitted training data is `self.X_train`.

### Per-row ranking and label lookup
- Status: practicing
- Meaning: A distance-matrix row belongs to one test sample; each column is a training sample. `argsort(..., axis=1)` returns training indices nearest-first, and `y_train[nearest_idx]` maps those indices to labels with shape `(num_test, k)`.
- Seen in: PA1 TODO-9 and the provided neighbour selection for TODO-10.
- Related methods: `np.argsort`; fancy indexing; `y_train[indices]`.
- Minimal example: `nearest_idx[0, 1] == 2` means training sample 2 is the second-nearest neighbour of test sample 0.
- Misconception to watch: Reading a stored index as a distance or rank value.

### Broadcasting singleton axes
- Status: practicing
- Meaning: Row reductions need explicit axes before combining them with a 2D result.
- Seen in: PA1 TODO-9 squared-distance formula.
- Related methods: `[:, None]`; `[None, :]`; broadcasting.
- Minimal example: `test_sq[:, None] + train_sq[None, :] - 2 * cross`.
- Misconception to watch: Equal-length 1D arrays can align elementwise without an error but give wrong pairwise results.

### Unweighted majority voting
- Status: practicing
- Meaning: Each of the `k` selected neighbours contributes one vote; distance determines membership but not vote weight.
- Seen in: PA1 TODO-10 `predict`.
- Related methods: `np.bincount`; `np.argmax`; tie-breaking by smallest label.
- Minimal example: Neighbour labels `[2, 0, 2]` produce counts `[1, 0, 2]` and prediction `2`.
- Misconception to watch: Weighting votes by distance when the specification asks for an unweighted majority.

### Degenerate-case handling
- Status: practicing
- Meaning: Required edge behavior follows the written formula rather than what the public tests happen to exercise: empty K-Means clusters retain their previous centroid, zero variances use `1e-9`, zero precision/recall denominators produce `0`, and threshold equality stays with GNB.
- Seen in: PA1 TODO-3, TODO-8, TODO-12, and TODO-14 edge-case testing.
- Related methods: epsilon stabilization; denominator guards; strict versus inclusive comparisons; retained cluster state.
- Minimal example: If precision and recall are both zero, return F1 `0` instead of evaluating `0 / 0`.
- Misconception to watch: Passing a mean-centred public test does not validate the Gaussian evidence formula because its squared-distance term is zero.

## Next Reviews
- [ ] For `dists` shaped `(7, 20)`, `nearest_idx` shaped `(7, 5)`, and `y_train` shaped `(20,)`, state the shape and meaning of `y_train[nearest_idx]`.
- [ ] For neighbour labels `[3, 1, 3, 1]`, determine the prediction and explain which `np.argmax` behaviour resolves the tie.
- [ ] For an empty K-Means cluster, explain why continuing alone would leave the new centroid at the origin and what assignment prevents that.
- [ ] Compute Macro-F1 for an empty prediction array and for an all-wrong four-class prediction array; both should be finite `0`.
