"""
ml_engine.py
Pure-NumPy Machine Learning Engine for Network Intrusion Detection.
Implements:
  - DecisionTreeClassifier (Gini Impurity, recursive splitting)
  - RandomForestClassifier (Ensemble Bagging, Random Subspace, Gini Feature Importance)
  - KNeighborsClassifier (Vectorized Euclidean Distance)
  - LogisticRegression (L2-Regularized Gradient Descent)
  - StandardScaler & LabelEncoder
  - Evaluation Metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
"""

import pickle
import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

class LabelEncoder:
    def __init__(self):
        self.classes_ = []
        self.mapping = {}

    def fit(self, y):
        unique = sorted(list(set(y)))
        self.classes_ = unique
        self.mapping = {val: idx for idx, val in enumerate(unique)}
        return self

    def transform(self, y):
        return np.array([self.mapping.get(val, 0) for val in y], dtype=np.int64)

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_indices):
        return [self.classes_[i] if 0 <= i < len(self.classes_) else str(i) for i in y_indices]

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, probas=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.probas = probas

    def is_leaf(self):
        return self.value is not None

def gini_impurity(y, n_classes):
    m = len(y)
    if m == 0:
        return 0.0
    counts = np.bincount(y, minlength=n_classes)
    probs = counts / m
    return 1.0 - np.sum(probs ** 2)

class DecisionTreeClassifier:
    def __init__(self, max_depth=12, min_samples_split=5, max_features=None, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.root = None
        self.n_classes_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(np.max(y)) + 1
        self.n_features_ = X.shape[1]
        self.feature_importances_ = np.zeros(self.n_features_)
        self.root = self._build_tree(X, y, depth=0)
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        counts = np.bincount(y, minlength=self.n_classes_)
        leaf_value = int(np.argmax(counts))
        leaf_probas = counts / max(1, n_samples)

        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return Node(value=leaf_value, probas=leaf_probas)

        # Feature subsampling
        if self.max_features is None:
            feat_indices = np.arange(n_features)
        elif isinstance(self.max_features, int):
            feat_indices = np.random.choice(n_features, size=min(n_features, self.max_features), replace=False)
        else:
            n_sub = max(1, int(np.sqrt(n_features)))
            feat_indices = np.random.choice(n_features, size=n_sub, replace=False)

        best_feat, best_thresh = None, None
        best_gain = -1.0
        current_gini = gini_impurity(y, self.n_classes_)

        for feat in feat_indices:
            values = np.unique(X[:, feat])
            if len(values) > 10:
                thresholds = np.percentile(values, np.linspace(10, 90, 8))
            else:
                thresholds = values

            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                gini_left = gini_impurity(y[left_mask], self.n_classes_)
                gini_right = gini_impurity(y[right_mask], self.n_classes_)
                w_left = np.sum(left_mask) / n_samples
                w_right = np.sum(right_mask) / n_samples
                gain = current_gini - (w_left * gini_left + w_right * gini_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_gain <= 1e-5 or best_feat is None:
            return Node(value=leaf_value, probas=leaf_probas)

        self.feature_importances_[best_feat] += best_gain * n_samples

        left_mask = X[:, best_feat] <= best_thresh
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return Node(feature=best_feat, threshold=best_thresh, left=left_child, right=right_child)

    def _predict_row(self, node, x):
        if node.is_leaf():
            return node.value, node.probas
        if x[node.feature] <= node.threshold:
            return self._predict_row(node.left, x)
        return self._predict_row(node.right, x)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = [self._predict_row(self.root, x)[0] for x in X]
        return np.array(preds, dtype=np.int64)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probas = [self._predict_row(self.root, x)[1] for x in X]
        return np.array(probas, dtype=np.float64)

class RandomForestClassifier:
    def __init__(self, n_estimators=40, max_depth=12, min_samples_split=4, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.feature_importances_ = None
        self.n_classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n_samples, n_features = X.shape
        self.n_classes_ = int(np.max(y)) + 1
        np.random.seed(self.random_state)

        self.trees = []
        self.feature_importances_ = np.zeros(n_features)
        max_feat = max(1, int(np.sqrt(n_features)))

        for i in range(self.n_estimators):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_b, y_b = X[boot_idx], y[boot_idx]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=max_feat,
                random_state=self.random_state + i
            )
            tree.fit(X_b, y_b)
            self.trees.append(tree)
            self.feature_importances_ += tree.feature_importances_

        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_probas = np.zeros((len(X), self.n_classes_))
        for tree in self.trees:
            all_probas += tree.predict_proba(X)
        all_probas /= len(self.trees)
        return all_probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)

class KNeighborsClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.n_classes_ = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.int64)
        self.n_classes_ = int(np.max(self.y_train)) + 1
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = []
        # Chunked distance calculation to conserve memory
        batch_size = 200
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i + batch_size]
            # Euclidean distances: ||A - B||^2 = A^2 + B^2 - 2AB
            dists = np.linalg.norm(X_batch[:, None, :] - self.X_train[None, :, :], axis=2)
            top_k_idx = np.argpartition(dists, self.n_neighbors, axis=1)[:, :self.n_neighbors]
            for row in top_k_idx:
                k_labels = self.y_train[row]
                counts = np.bincount(k_labels, minlength=self.n_classes_)
                preds.append(np.argmax(counts))
        return np.array(preds, dtype=np.int64)

class LogisticRegression:
    def __init__(self, lr=0.05, n_iters=300, reg=0.01):
        self.lr = lr
        self.n_iters = n_iters
        self.reg = reg
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear = np.dot(X, self.weights) + self.bias
            # Clipped sigmoid to prevent overflow
            linear = np.clip(linear, -25.0, 25.0)
            preds = 1.0 / (1.0 + np.exp(-linear))

            dw = (1.0 / n_samples) * np.dot(X.T, (preds - y)) + (self.reg / n_samples) * self.weights
            db = (1.0 / n_samples) * np.sum(preds - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        linear = np.dot(X, self.weights) + self.bias
        linear = np.clip(linear, -25.0, 25.0)
        p1 = 1.0 / (1.0 + np.exp(-linear))
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X):
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(np.int64)

# Metrics
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = np.mean(y_true == y_pred)
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(classes), len(classes)), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if len(classes) == 2:
        tp = cm[1, 1]
        fp = cm[0, 1]
        fn = cm[1, 0]
        tn = cm[0, 0]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        precisions = []
        recalls = []
        for c in range(len(classes)):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": round(float(acc) * 100, 2),
        "precision": round(float(precision) * 100, 2),
        "recall": round(float(recall) * 100, 2),
        "f1_score": round(float(f1) * 100, 2),
        "confusion_matrix": cm.tolist()
    }
