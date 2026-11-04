# -*- coding: utf-8 -*-
"""pa1_tasks.ipynb
## Data description

The file `traffic_signs_features.csv` contains 3,200 samples and 10 continuous features extracted from traffic-sign images.

The label column uses:

- `0`: Speed Limit
- `1`: Stop
- `2`: Yield
- `3`: Ahead Only

The feature names are:

`Red_Color_Ratio`, `Blue_Color_Ratio`, `White_Color_Ratio`, `Circularity`, `Triangularity`, `Octagonality`, `Symmetry_X`, `Symmetry_Y`, `Center_Density`, `Edge_Contrast`

## Public Test Setup

Download `public_test_cases.zip` from the assignment webpage. In Colab, upload it through the left-hand Files panel. In local Jupyter, place it in the notebook's working directory. Run the following setup cell once. Public tests use your current notebook definitions and do not require exporting your code.

**Public Tests** cells run the same public cases as ZINC for a preliminary correctness check. Your final score is determined by separate hidden tests.
"""

if __name__ == "__main__":
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from zipfile import ZipFile
    import importlib.util

    public_tests_zip = Path("public_test_cases.zip")
    public_test_runner = None
    if not public_tests_zip.is_file():
        print("Public tests are optional. You can continue working without the ZIP.")
    else:
        public_tests_folder = TemporaryDirectory(prefix="pa1_public_tests_")
        public_tests_path = Path(public_tests_folder.name)
        with ZipFile(public_tests_zip) as archive:
            for name in archive.namelist():
                if Path(name).name != name or not name.endswith(".py"):
                    raise ValueError("Please use the public_test_cases.zip provided on the assignment webpage.")
                (public_tests_path / name).write_bytes(archive.read(name))
        public_runner_spec = importlib.util.spec_from_file_location(
            "pa1_public_test_runner", public_tests_path / "public_test_runner.py"
        )
        public_test_runner = importlib.util.module_from_spec(public_runner_spec)
        public_runner_spec.loader.exec_module(public_test_runner)
        print("Public tests are ready.")

"""## Part 0: Setup & Data Exploration"""

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Load the provided feature dataset
    df_data = pd.read_csv("traffic_signs_features.csv")

    print(df_data.head())
    print(f"\nDataset shape: {df_data.shape}")
    print("\nClass counts:")
    print(df_data["label"].value_counts().sort_index())

    # Extract features X and labels y
    X = df_data.drop(columns=["label"]).values
    y = df_data["label"].values.astype(int)

    # 70% train, 15% validation, 15% test.
    # Validation is used only for model/hyperparameter decisions.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42, stratify=y_temp
    )

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")


def standardize_features(X_train, X_val, X_test):
    """Standardize features using training-set statistics only."""
    ### TODO-1 ###
    # 1. Compute feature-wise mean from X_train.
    # 2. Compute feature-wise standard deviation from X_train.
    # 3. Add epsilon = 1e-9 to std to avoid division by zero.
    # 4. Transform train, validation, and test using the SAME train mean/std.
    epsilon = 1e-9
    mean = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + epsilon
    X_train_scaled = (X_train-mean)/sd
    X_val_scaled = (X_val-mean)/sd
    X_test_scaled = (X_test-mean)/sd
    ###########################
    return X_train_scaled, X_val_scaled, X_test_scaled


if __name__ == "__main__":
    X_train_scaled, X_val_scaled, X_test_scaled = standardize_features(
        X_train, X_val, X_test
    )
    print("Scaled train mean:", np.mean(X_train_scaled, axis=0))
    print("Scaled train std:", np.std(X_train_scaled, axis=0))

"""### Public Tests: Part 1

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(1, globals())


class KMeansClustering:
    def __init__(self, k=4, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices].copy()
        self.labels = np.zeros(X.shape[0], dtype=int)

        for _ in range(self.max_iters):
            ### TODO-2: Assign each point to its nearest centroid ###
            # Use the Euclidean distance to measure the distance between samples and centroids.
            # Update self.labels with the nearest-centroid index for each sample.
            # Initialize new_centroids as a zero array of shape (self.k, X.shape[1])
            # for TODO-3 to fill in during this iteration.
            n = X.shape[0]
            feats = X.shape[1]
            X_k_extend = np.broadcast_to(X[:,None,:],(n,self.k,feats))      # Shape (n,k,feats)
            distance_k = np.sqrt(np.square(X_k_extend - self.centroids).sum(axis=2))    # Shape (n,k)
            self.labels = distance_k.argmin(axis=1)                         # Shape (n)
            new_centroids = np.zeros((self.k,feats))
            
            ### TODO-3: Update each centroid from its assigned points ###
            # If a cluster receives no points, keep its previous centroid.
            for c in range(self.k):
                #### TODO 3 Starts ##################
                index_c = self.labels == c
                count_c = np.count_nonzero(index_c)
                if count_c == 0:
                    new_centroids[c] = self.centroids[c]
                    continue
                points_c = X * index_c[:,None]
                avg_c = points_c.sum(axis=0)/count_c
                new_centroids[c] = avg_c

                #### TODO 3 Ends   ##################

            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids

    def compute_wcss(self, X):
        """Return the within-cluster sum of squared distances."""
        wcss = 0.0
        ### TODO-4: Compute WCSS ###
        # Your implementation should correctly handle cases where a cluster
        # has no assigned data points. An empty cluster contributes 0 to WCSS.
        
        X_k_extend = np.broadcast_to(X[:,None,:],(X.shape[0],self.k,X.shape[1]))
        dist_square_k = np.square(X_k_extend - self.centroids).sum(axis=2)          # Shape (n,k)
        wcss = np.take_along_axis(dist_square_k,self.labels[:,None],axis=1).sum()

        return wcss

"""##### The Elbow Method"""

if __name__ == "__main__":
    wcss_list = []
    k_range = range(1, 11)

    for k_val in k_range:
        kmeans = KMeansClustering(k=k_val)
        kmeans.fit(X_train_scaled)
        wcss_list.append(kmeans.compute_wcss(X_train_scaled))

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, wcss_list, marker="x")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS (Inertia)")
    plt.show()

"""### Public Tests: Part 2

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(2, globals())


class GaussianNaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.log_priors = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.mean = np.zeros((n_classes, n_features))
        self.var = np.zeros((n_classes, n_features))
        self.log_priors = np.zeros(n_classes)


        ### TODO-5: Learn mean, variance, and class prior for each class ###
        # Store the learned values in self.mean, self.var, self.log_priors.
        for i in range(self.classes.size):
            this_class = self.classes[i]                                
            this_class_X = X[np.all((y==this_class)[:,None],axis=1)]    # shape(#this_class_X,feats)
            self.mean[i,:] = this_class_X.mean(axis=0)                   # shape(1,feats)
            self.var[i,:] = np.square(this_class_X.std(axis=0))
            self.log_priors[i]=np.log(this_class_X.shape[0]/X.shape[0])

        #####################################

    def _log_scores(self, X):
        """Return an array of shape (n_samples, n_classes)."""
        scores = []
        for idx, _ in enumerate(self.classes):
            var_eps = self.var[idx] + 1e-9
            log_prob = (
                -0.5 * np.sum(np.log(2 * np.pi * var_eps))
                -0.5 * np.sum(((X - self.mean[idx]) ** 2) / var_eps, axis=1)
            )
            scores.append(self.log_priors[idx] + log_prob)
        return np.column_stack(scores)

    def predict(self, X):
        ### TODO-6: Predict the class with the largest log posterior score (You need to compute the scores) ###
        scores = self._log_scores(X)
        #####################################
        return self.classes[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        """Convert log scores into normalized class probabilities."""
        scores = self._log_scores(X)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        probs = np.exp(scores)
        return probs / np.sum(probs, axis=1, keepdims=True)

if __name__ == "__main__":
    gnb = GaussianNaiveBayes()
    gnb.fit(X_train_scaled, y_train)

    val_preds_gnb = gnb.predict(X_val_scaled)
    test_preds_gnb = gnb.predict(X_test_scaled)

    print(f"GNB Validation Accuracy: {np.mean(val_preds_gnb == y_val):.4f}")
    print(f"GNB Test Accuracy: {np.mean(test_preds_gnb == y_test):.4f}")

"""### Public Tests: Part 3

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(3, globals())

"""## Part 4: Learned Feature Analysis

A key advantage of simpler models like Gaussian Naive Bayes is their interpretability. Instead of treating the model as a "black box", we can inspect its learned parameters to understand what it has learned from the data. This section is divided into two parts. First, you will perform a global analysis to determine which features are the most "discriminative" across all classes by comparing how much the class means vary relative to the variance within the classes. Second, you will conduct a local analysis of a single prediction by ranking the features according to their Gaussian log-density contributions for the predicted class.

### 4.1 Global feature discriminativeness

Use the learned class means and variances to compute a simple Fisher-style separation score for feature $j$:

$$F_j = \frac{Var_c(\mu_{c,j})}{Mean_c(\sigma^2_{c,j}) + \epsilon}$$

where:

- $j$ is a feature index, and $c$ is a class label.
- $\mu_{c,j}$ and $\sigma^2_{c,j}$ are the mean and variance of feature $j$ within class $c$, learned from the standardized training data in Part 3.
- $F_j$ is the discriminativeness score for feature $j$.
- $Var_c(\mu_{c,j})$ is the variance of the learned class means for feature $j$, calculated across classes.
- $Mean_c(\sigma^2_{c,j})$ is the average of the learned within-class variances for feature $j$, calculated across classes.
- $\epsilon=10^{-9}$ is a small constant added to prevent division by zero.

The subscript $c$ in $Var_c$ and $Mean_c$ means that each operation is performed across classes while keeping feature $j$ fixed. These are not the variance and mean of all training samples pooled together.

A larger $F_j$ means that class means are more separated relative to within-class variation. In **TODO-7**, compute one score for each feature; the code that ranks and prints these scores is already provided.

### 4.2 Local evidence for an individual prediction

For one sample and one class, the Gaussian log-density contribution of feature $j$ is:

$$e_j(c)=-\frac12\log(2\pi\sigma^2_{c,j})-\frac{(x_j-\mu_{c,j})^2}{2\sigma^2_{c,j}}$$

where:

- $x_j$ is the standardized value of feature $j$ for the sample being explained.
- $c$ is the class whose feature distribution is being evaluated; for the explanation, use the model's predicted class.
- $\mu_{c,j}$ and $\sigma^2_{c,j}$ are the learned mean and variance defined above.
- $e_j(c)$ is the contribution of feature $j$ to the Gaussian log-likelihood for class $c$.
- $\log$ is the natural logarithm, and $\pi$ is the mathematical constant pi.

**In the implementation, replace $\sigma^2_{c,j}$ with $\sigma^2_{c,j}+10^{-9}$ wherever it appears in this formula, to avoid division by zero and $\log(0)$.**

In **TODO-8**, predict the class for the sample, compute the feature contributions for that class, and rank all features from largest to smallest contribution. Larger values indicate stronger evidence under that class's learned feature distributions, even if the values are negative. Return the prediction and a list of `(feature_name, evidence)` pairs.

This ranking describes the individual feature log-density contributions for the predicted class. It does not compare those contributions with other classes or establish that a feature caused the prediction.

##### **TODO-7: Rank global feature discriminativeness from learned GNB statistics**
"""

def feature_discriminativeness(gnb):
    """Return one separation score per feature."""
    ### TODO-7 ###
    # Compute between-class variance of learned means and divide by
    # the mean within-class variance.
    between_class_variance = np.square(gnb.mean.std(axis=0))
    within_class_variance = gnb.var.mean(axis=0)

    #####################################
    return between_class_variance / (within_class_variance + 1e-9)


if __name__ == "__main__":
    feature_names = df_data.drop(columns=["label"]).columns.to_numpy()
    scores = feature_discriminativeness(gnb)
    ranking = np.argsort(scores)[::-1]
    print("Feature ranking (most discriminative first):")
    for idx in ranking:
        print(f"{feature_names[idx]:20s} {scores[idx]:.4f}")

"""##### **TODO-8: Explain one prediction using learned feature evidence**"""

def explain_prediction(gnb, x, feature_names):
    """Return the predicted class and feature evidence for that class."""
    # Hint: use variance + 1e-9 in e_j(c) to avoid division by zero and log(0).
    ### TODO-8 ###
    # 1. Predict the class for x.
    # 2. For the predicted class, compute e_j(c) for every feature.
    # 3. Rank features from strongest to weakest evidence.
    if x.ndim < 2:
        x = x[None,:]
    predicted_class = gnb.predict(x)[0]
    var_nonezero = gnb.var + 1e-9
    evidence = (-0.5*np.log(2*np.pi*var_nonezero[predicted_class,:])- np.square(x-gnb.mean[predicted_class,:]) / (2*var_nonezero[predicted_class,:])).flatten()
    order = np.argsort(evidence)[::-1]
    ranked_evidence = list(zip(feature_names[order],evidence[order]))

    #####################################

    # 4. Return the prediction and a list of (feature_name, evidence).
    return predicted_class, ranked_evidence

"""### Public Tests: Part 4

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(4, globals())

"""## Part 5: KNN Fallback Learner

Different machine learning models have different strengths. While GNB creates a parametric model of the data, the K-Nearest Neighbors (KNN) algorithm takes a non-parametric, instance-based approach. It classifies new data points based on the labels of the most similar examples in the training set (its "neighbors"). Unlike GNB, KNN does not learn parameters for a fixed probability distribution. Instead, it stores the training samples and uses them directly when making predictions.

In this section, you will implement the KNN classifier and tune its hyperparameter K. Here, K is the number of nearest training samples used in the majority vote, not the number of clusters used in Part 2. A smaller K makes predictions depend on fewer nearby samples and can be more sensitive to noise; a larger K combines more neighbors but can smooth over local differences.

Use the Euclidean distance in the **scaled** feature space. The distance matrix should be computed with vectorized NumPy operations rather than nested Python loops.

The tasks are:

- **TODO-9:** compute the Euclidean distance matrix between query samples and training samples. Each row corresponds to a query sample, and each column corresponds to a training sample.
- **TODO-10:** predict each query sample's class by majority vote among its K nearest training neighbors. The code that selects the neighbors is already provided.
- **TODO-11:** fit and evaluate KNN for each candidate K using the training and validation sets. The code that calculates validation accuracy and tracks the best K is already provided.

Choose the K with the highest validation accuracy **among the provided candidates**. Use only the training samples as the neighbor database, the validation set to select K, and the test set for final evaluation. Do not choose K based on test accuracy.

This section builds and tunes the KNN classifier. In Part 6, you will decide when it should act as a fallback for uncertain GNB predictions.

##### **TODO-9,10: Compute distances and predict neighbour majority**
"""

class KNNClassifier:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.astype(int)

    def compute_distances(self, X_test):
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        ### TODO-9: Compute vectorized Euclidean distances ###
        # Define the intermediate arrays used by the provided distance formula:
        # test_sq: sum of squared feature values for each query sample,
        #          shape (num_test, 1).
        # train_sq: sum of squared feature values for each training sample,
        #           shape (1, num_train).
        # cross: dot product of each query sample with each training sample,
        #        shape (num_test, num_train).
        # The formula below uses squared distance = test_sq + train_sq - 2 * cross.
        test_sq = np.sum(np.square(X_test),axis=1)[:,None]
        train_sq = np.sum(np.square(self.X_train),axis=1)[None,:]
        cross = np.dot(X_test,self.X_train.T)
        #######################################
        dists = np.sqrt(np.maximum(test_sq + train_sq - 2 * cross, 0.0))
        return dists

    def predict(self, X_test):
        dists = self.compute_distances(X_test)

        nearest_idx = np.argsort(dists, axis=1)[:, :self.k]
        nearest_labels = self.y_train[nearest_idx]
        predictions = []
        for row in nearest_labels:
            ### TODO-10: Predict by majority vote among the k nearest neighbours ###
            # If there is a tie in the majority vote, the class with the smallest label index should be chosen.
            predictions.append(np.argmax(np.bincount(row)))

            ###############################################

        return np.array(predictions, dtype=int)

"""##### **TODO-11: Tune K on the validation set**"""

def find_best_k(X_train, y_train, X_val, y_val, k_choices):
    """Choose the K with the highest validation accuracy."""
    best_k = k_choices[0]
    best_acc = -1.0
    val_accuracies = []

    for k in k_choices:
        ### TODO-11 ####################
        #   1. Fit KNN on training data.
        #   2. Predict the validation set.
        knn = KNNClassifier(k=k)
        knn.fit(X_train,y_train)
        val_pred = knn.predict(X_val)
        #################################
        #   3. Compute accuracy.
        acc = np.mean(val_pred == y_val)
        val_accuracies.append(acc)
        #   4. Track the best K.
        if acc > best_acc:
            best_acc = acc
            best_k = k

    return best_k, val_accuracies

if __name__ == "__main__":
    k_choices = [1, 3, 5, 7, 9, 11, 15]
    best_k, val_accs = find_best_k(
        X_train_scaled, y_train, X_val_scaled, y_val, k_choices
    )

    print("Best K:", best_k)
    print("Validation accuracies:", val_accs)

    best_knn = KNNClassifier(k=best_k)
    best_knn.fit(X_train_scaled, y_train)
    test_preds_knn = best_knn.predict(X_test_scaled)
    print(f"KNN Test Accuracy (K={best_k}): {np.mean(test_preds_knn == y_test):.4f}")

"""### Public Tests: Part 5

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(5, globals())

"""## Part 6: Confidence-Based GNB → KNN Fallback

No single model is perfect for every situation. In this part, you will explore a hybrid system that uses GNB as the primary classifier and KNN as a fallback for cases where GNB is uncertain. The goal is to test whether switching to KNN for these cases improves prediction accuracy; combining the models does not guarantee better performance.

Using the GNB output scores, you will measure how clearly its preferred class stands out from the alternatives. Then, using the validation set, you will select a threshold that determines when to keep GNB's prediction and when to use KNN instead.

We will use the **margin** between the highest and second-highest GNB log-posterior scores:

$$margin = score_{(1)} - score_{(2)}$$

where:

- $score_{(1)}$ is the highest GNB score across all classes for one sample.
- $score_{(2)}$ is the second-highest GNB score for the same sample.
- The subscripts $(1)$ and $(2)$ denote ranks, not class labels.
- $margin$ is the difference between these two scores. A small margin means the top two candidate classes have similar scores, while a large margin means the highest-scoring class stands out more clearly.

The margin is a relative confidence measure based on log scores, not a probability.

For a candidate threshold $t$:

- use GNB when `margin >= t`, including when the margin equals the threshold;
- use KNN when `margin < t`.

For each candidate threshold, generate the hybrid predictions and calculate accuracy over the **entire validation set**, not just the samples that use KNN. Select the threshold with the highest validation accuracy **among the provided candidates**. Keep the trained GNB and KNN models fixed while comparing thresholds.

Once selected, keep the threshold fixed for final evaluation on the test set. **Do not choose the threshold from the test set.** Compare the hybrid model with the individual classifiers to see whether the fallback helps.

The tasks are:

- **TODO-12:** for each candidate threshold, combine the GNB and KNN predictions according to the margin rule, calculate validation accuracy, and track the best threshold. The validation predictions and margins are already provided. Start each candidate with a fresh copy of the GNB predictions so that changes from one candidate do not carry over to the next.
- **TODO-13:** construct the boolean `fallback_mask` identifying samples with `margin < threshold`, and initialize `final_preds` as a copy of the GNB predictions. The code that calls KNN only for the selected samples, replaces their predictions, and returns `final_preds` and `fallback_mask` is already provided.

##### **TODO-12: Choose the fallback threshold from validation data**
"""

def select_fallback_threshold(gnb, knn, X_val, y_val, candidate_thresholds):
    """Select a margin threshold using validation accuracy only."""
    #   1. Get GNB log scores on the validation set.
    scores = gnb._log_scores(X_val)
    #   2. Compute top-vs-second margin.
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    gnb_preds = gnb.classes[np.argmax(scores, axis=1)]
    knn_preds = knn.predict(X_val)

    best_threshold = candidate_thresholds[0]
    best_acc = -1.0
    # For each threshold:
    for threshold in candidate_thresholds:
        ### TODO-12 ####################
        #   3. Use KNN predictions only where margin < threshold.
        #   4. Compute validation accuracy.
        # Select the threshold with highest validation accuracy.

        preds = gnb_preds * (margins >= threshold) + knn_preds * (margins < threshold)
        acc = np.mean(preds == y_val)
        if (acc>best_acc):
            best_acc = acc
            best_threshold = threshold


        #################################
    return best_threshold

"""##### **TODO-13: Implement the final hybrid predictor**"""

def hybrid_predict(gnb, knn, X, threshold):
    """Use GNB for confident cases and KNN for ambiguous cases."""

    # 1. Compute GNB log scores and initial predictions.
    # 2. Compute the top-vs-second margin.
    scores = gnb._log_scores(X)
    gnb_preds = gnb.classes[np.argmax(scores, axis=1)]
    sorted_scores = np.sort(scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    ### TODO-13 ######################################
    # 3.1. Create a Boolean mask (fallback_mask) to identify samples where margin < threshold.
    # 3.2. Initialize final_preds as a copy of the GNB predictions. The code to call KNN
    #      on the subset of data X[fallback_mask] and update final_preds is provided
    #      after this block.
    fallback_mask = margins < threshold
    final_preds = gnb_preds.copy()

    ###################################################
    # 4. Return final predictions and the fallback mask.
    if np.any(fallback_mask):
        knn_preds = knn.predict(X[fallback_mask])
        final_preds[fallback_mask] = knn_preds
    return final_preds, fallback_mask

if __name__ == "__main__":
    candidate_thresholds = [2, 4, 6, 8, 10, 12, 14, 16, 18]

    gnb_for_fallback = GaussianNaiveBayes()
    gnb_for_fallback.fit(X_train_scaled, y_train)

    knn_for_fallback = KNNClassifier(k=best_k)
    knn_for_fallback.fit(X_train_scaled, y_train)

    best_threshold = select_fallback_threshold(
        gnb_for_fallback, knn_for_fallback, X_val_scaled, y_val, candidate_thresholds
    )

    print(f"Best fallback threshold: {best_threshold}")

    hybrid_preds, fallback_mask = hybrid_predict(
        gnb_for_fallback, knn_for_fallback, X_test_scaled, best_threshold
    )

    print(f"Hybrid Test Accuracy: {np.mean(hybrid_preds == y_test):.4f}")
    print(f"Fallback rate on test set: {np.mean(fallback_mask):.2%}")

"""### Public Tests: Part 6

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(6, globals())

"""## Part 7: Multi-Class Evaluation

While overall accuracy gives a general sense of performance, it doesn't reveal the whole story, especially in multi-class classification. It's important to understand what kind of errors a model is making. In this final section, you will complete `evaluate_multiclass()` to calculate per-class evaluation metrics and Macro-F1. The provided code builds a confusion matrix, which you can inspect to see which classes are being confused with others. These metrics help you compare performance on each traffic-sign category and summarize performance across all four classes.

For each model, calculate:

- **4×4 confusion matrix:** each row represents a true class, and each column represents a predicted class. An entry counts the samples with that true and predicted class. Diagonal entries count correct predictions; off-diagonal entries show errors.
- **Per-class precision:** among the samples predicted as a class, the fraction that actually belongs to that class.
- **Per-class recall:** among the samples that actually belong to a class, the fraction correctly predicted as that class.
- **Per-class F1:** the harmonic mean of that class's precision and recall, combining both measures into one score.
- **Macro-F1:** the unweighted mean of the four per-class F1 scores. Each class contributes equally, regardless of its number of samples.

In **TODO-14**, use the provided confusion matrix to calculate precision, recall, and F1 for each class, then calculate Macro-F1. If a precision or recall denominator is zero, use 0 for that metric. If precision plus recall is zero, use 0 for F1.

The confusion-matrix construction and the code that prints the results are already provided; no additional plotting is required. Calculate the per-class metrics internally, and keep the function's return values as **the confusion matrix and Macro-F1**.

Use the same test set to evaluate:

1. Gaussian Naive Bayes;
2. KNN with the K selected using the validation set;
3. the hybrid GNB→KNN pipeline with the threshold selected using the validation set.

Compare the confusion matrices and Macro-F1 scores to see which classes each model handles well and whether the hybrid pipeline improves performance. Do not use these test-set results to choose K or the fallback threshold.

##### **TODO-14: Confusion matrix and Macro-F1**
"""

def evaluate_multiclass(y_true, y_pred, num_classes=4):
    """Return confusion matrix and Macro-F1."""

    # 1. Build the 4x4 confusion matrix.
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1

    per_class_f1 = []

    for c in range(num_classes):
        ### TODO-14 ###################################
        # 2. Compute precision/recall/F1 for each class.
        predicted_c = cm[:,c].sum()
        true_c = cm[c,:].sum()
        precision = cm[c,c] / predicted_c if predicted_c else 0.0
        recall = cm[c,c] / true_c if true_c else 0.0
        denominator = precision + recall
        f1 = 2*(precision*recall)/denominator if denominator else 0.0
        ###############################################
        per_class_f1.append(f1)

    # 3. Return the matrix and Macro-F1.
    macro_f1 = np.array(per_class_f1).mean()

    ###############################################
    return cm, macro_f1

if __name__ == "__main__":
    for name, preds in [
        ("GNB", test_preds_gnb),
        ("KNN", test_preds_knn),
        ("Hybrid", hybrid_preds),
    ]:
        cm, macro_f1 = evaluate_multiclass(y_test, preds, num_classes=4)
        print(f"\n{name} Confusion Matrix:")
        print(cm)
        print(f"{name} Macro F1-Score: {macro_f1:.4f}")

"""### Public Tests: Part 7

Run this cell to check this Part using the ZINC public test cases. After changing your implementation, rerun its code cell before running the tests.

"""

if __name__ == "__main__":
    if globals().get("public_test_runner") is None:
        print("To run public tests, place public_test_cases.zip in the current working directory and run Public Test Setup.")
    else:
        public_test_runner.run_public_tests(7, globals())
