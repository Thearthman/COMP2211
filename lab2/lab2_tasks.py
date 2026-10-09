import pandas as pd
import numpy as np
from pandas import DataFrame


def pandas_to_numpy_features(df: DataFrame) -> np.ndarray:
    """Convert the feature DataFrame into a NumPy array.

    Categorical columns (Month, VisitorType, Weekend) should be mapped to
    integer codes according to the mapping rules in Task 1, while the
    numerical columns (ProductRelated_Duration, BounceRates, PageValues)
    are passed through unchanged.

    Args:
        df (DataFrame): DataFrame containing the six feature columns
            "Month", "VisitorType", "Weekend", "ProductRelated_Duration",
            "BounceRates", "PageValues". "Month" and "VisitorType" hold
            strings, "Weekend" holds booleans, and the remaining columns
            hold numeric values.

    Returns:
        np.ndarray: Array of shape (n, 6) containing the six features in
            the same column order, with Month/VisitorType/Weekend
            converted to integer codes.

    TODO:
    - Map Month, VisitorType, and Weekend to integer codes.
    - Return the six feature columns as a NumPy array with shape (n, 6).
    """
    MONTH = {"Jan":0, "Feb":1, "Mar":2, "Apr":3, "May":4, "June":5, "Jul":6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
    VTYPE = {"Returning_Visitor": 0, "New_Visitor":1, "Other": 2}
    
    
    df_cp = df.copy()
    df_cp["Month"] = df_cp["Month"].map(MONTH).astype(int)
    df_cp["VisitorType"] = df_cp["VisitorType"].map(VTYPE).astype(int)
    df_cp["Weekend"] = df_cp["Weekend"].astype(int)
    return np.array(df_cp)

def pandas_to_numpy_revenue(df: DataFrame) -> np.ndarray:
    """Convert the "Revenue" column into a NumPy array.

    Args:
        df (DataFrame): DataFrame containing a boolean "Revenue" column,
            where True means the session ended in a purchase and False
            means it did not.

    Returns:
        np.ndarray: Array of shape (n,) with Revenue mapped to integer
            labels in {0, 1} (False -> 0, True -> 1).

    TODO:
    - Map Revenue values to integer codes.
    - Return a NumPy array with shape (n,).
    """

    return np.array(df.astype(int)).flatten()
    

if __name__ == "__main__":
    # run the above 2 functions and produce a small expected output

    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    X_train_np = pandas_to_numpy_features(X_train)
    y_train_np = pandas_to_numpy_revenue(y_train)
    X_test_np = pandas_to_numpy_features(X_test)

    print("X_train_np:\n", X_train_np[:3])
    print("y_train_np:\n", y_train_np[:3])
    print("X_test_np:\n", X_test_np[:3])    


def estimate_priors(y_train: np.ndarray) -> np.ndarray:
    """Estimate the prior probability of each class from the training labels.

    Args:
        y_train (np.ndarray): Array of shape (n_train,) containing the
            integer class labels {0, 1} for the training set.

    Returns:
        np.ndarray: Array of shape (C,) containing the prior probability
            P(Y = j) for each class j, computed as the fraction of
            training examples belonging to that class.

    TODO:
    - Count how often each class appears in y_train.
    - Return the class probabilities as a NumPy array.
    """
    
    return np.bincount(y_train.flatten())/y_train.size

if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_train_np = pandas_to_numpy_features(X_train)
    y_train_np = pandas_to_numpy_revenue(y_train)

    priors = estimate_priors(y_train_np)
    print(priors)


def cat_likelihood(X_train_feature: np.ndarray, y_train_feature, X_test_feature, alpha) -> np.ndarray:
    """Estimate P(X_f = x | Y = j) for one categorical feature using Laplace smoothing.

    Args:
        X_train_feature (np.ndarray): Array of shape (n_train,) containing
            the integer-coded category values for this feature in the
            training set.
        y_train_feature (np.ndarray): Array of shape (n_train,) containing
            the integer class labels for the training set.
        X_test_feature (np.ndarray): Array of shape (n_test,) containing
            the integer-coded category values for this feature in the
            test set.
        alpha (float): Laplace smoothing parameter added to every
            category/class count.

    Returns:
        np.ndarray: Array of shape (n_test, C) where entry [i, j] is the
            Laplace-smoothed likelihood P(X_f = X_test_feature[i] | Y = j).

    TODO:
    - Compute the likelihood for each class and each test example.
    - Use Laplace smoothing with the provided alpha value.
    - Return an array with shape (n_test, C).
    """
    c_size = np.unique(y_train_feature.flatten()).size
    if X_test_feature.size ==0:
        return np.zeros((0,c_size))
    num_categories = int(max(X_train_feature.max(), X_test_feature.max())+1)
    bin_coded_feature_class = (X_train_feature * c_size + y_train_feature.flatten()).astype(int)
    count_numerator = np.bincount(bin_coded_feature_class,minlength=num_categories*c_size).reshape(num_categories,c_size) + alpha
    count_denom = np.bincount(y_train_feature.flatten()) + alpha*num_categories
    prob = count_numerator/count_denom
    return prob[X_test_feature.astype(int)]


if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    X_train_np = pandas_to_numpy_features(X_train)
    y_train_np = pandas_to_numpy_revenue(y_train)
    X_test_np = pandas_to_numpy_features(X_test)

    alpha = 1.0
    month_likelihoods = cat_likelihood(X_train_np[:, 0], y_train_np, X_test_np[:, 0], alpha)
    visitor_likelihoods = cat_likelihood(X_train_np[:, 1], y_train_np, X_test_np[:, 1], alpha)
    weekend_likelihoods = cat_likelihood(X_train_np[:, 2], y_train_np, X_test_np[:, 2], alpha)

    print("Month Likelihoods:\n", month_likelihoods[:3])
    print("Visitor Likelihoods:\n", visitor_likelihoods[:3])
    print("Weekend Likelihoods:\n", weekend_likelihoods[:3])


def num_likelihood(X_train_feature: np.ndarray, y_train_feature: np.ndarray, X_test_feature: np.ndarray) -> np.ndarray:
    """Estimate P(X_f = x | Y = j) for one numerical feature using a Gaussian distribution.

    For each class j, the mean and variance of the feature should be
    computed from the training examples belonging to that class, and the
    Gaussian PDF with those parameters evaluated at each test value.

    Args:
        X_train_feature (np.ndarray): Array of shape (n_train,) containing
            the values of this numerical feature in the training set.
        y_train_feature (np.ndarray): Array of shape (n_train,) containing
            the integer class labels for the training set.
        X_test_feature (np.ndarray): Array of shape (n_test,) containing
            the values of this numerical feature in the test set.

    Returns:
        np.ndarray: Array of shape (n_test, C) where entry [i, j] is the
            Gaussian likelihood P(X_f = X_test_feature[i] | Y = j).

    TODO:
    - Compute the mean and variance for each class.
    - Evaluate the Gaussian PDF for each test value.
    - Return an array with shape (n_test, C).
    """
    # Completely based on the fact that y has only two class 0 and 1
    y_train_feature = y_train_feature.flatten().astype(int)
    if X_test_feature.size ==0:
        c_size = np.unique(y_train_feature.flatten()).size
        return np.zeros((0,c_size))
    mean_denom = np.bincount(y_train_feature)
    mean = np.bincount(y_train_feature,weights=X_train_feature) / mean_denom
    dev = X_train_feature - mean[y_train_feature]
    var = (np.bincount(y_train_feature,weights=dev**2) / mean_denom) + 10**(-9)
    prob_denom = np.sqrt(var*2*np.pi)
    prob = np.exp(-(X_test_feature[:,None]-mean[None,:])**2/(var[None,:] * 2)) /prob_denom[None,:]
    return prob


if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    X_train_np = pandas_to_numpy_features(X_train)
    y_train_np = pandas_to_numpy_revenue(y_train)
    X_test_np = pandas_to_numpy_features(X_test)

    product_related_duration_likelihoods = num_likelihood(X_train_np[:, 3], y_train_np, X_test_np[:, 3])
    bounce_rates_likelihoods = num_likelihood(X_train_np[:, 4], y_train_np, X_test_np[:, 4])
    page_values_likelihoods = num_likelihood(X_train_np[:, 5], y_train_np, X_test_np[:, 5])

    print("ProductRelated_Duration Likelihoods:\n", product_related_duration_likelihoods[:2])
    print("BounceRates Likelihoods:\n", bounce_rates_likelihoods[:2])
    print("PageValues Likelihoods:\n", page_values_likelihoods[:2])


def predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, feat_type_list: list) -> np.ndarray:
    """Predict the class label for each test example using Naive Bayes.

    Combines the log prior with the log likelihood of every feature
    (categorical features via cat_likelihood, numerical features via
    num_likelihood) and picks the class with the highest log posterior.

    Args:
        X_train (np.ndarray): Array of shape (n_train, d) containing the
            training feature matrix.
        y_train (np.ndarray): Array of shape (n_train,) containing the
            integer class labels for the training set.
        X_test (np.ndarray): Array of shape (n_test, d) containing the
            test feature matrix.
        feat_type_list (list): List of length d where each entry is either
            "categorical" or "numerical", indicating how the corresponding
            column of X_train/X_test should be modeled.

    Returns:
        np.ndarray: Array of shape (n_test,) containing the predicted
            class label in {0, 1} for each test example.

    TODO:
    - Estimate the priors from the training labels.
    - Combine categorical and numerical likelihoods.
    - Use log probabilities to choose the most likely class.
    - Return predictions with shape (n_test,).
    """
    return (np.log(np.array([cat_likelihood(X_train[:,i],y_train,X_test[:,i],alpha=1) if (feat_type_list[i]=="categorical") else num_likelihood(X_train[:,i],y_train,X_test[:,i]) for i in range(len(feat_type_list))])).sum(axis=0) + np.log(estimate_priors(y_train))).argmax(axis=1)


if __name__ == "__main__":
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    X_test = pd.read_csv("X_test.csv")

    X_train_np = pandas_to_numpy_features(X_train)
    y_train_np = pandas_to_numpy_revenue(y_train)
    X_test_np = pandas_to_numpy_features(X_test[:10]) # Use only the first 10 samples for testing

    feat_type_list = ["categorical", "categorical", "categorical", "numerical", "numerical", "numerical"]

    predictions = predict(X_train_np, y_train_np, X_test_np, feat_type_list)
    print("Predictions:\n", predictions)
