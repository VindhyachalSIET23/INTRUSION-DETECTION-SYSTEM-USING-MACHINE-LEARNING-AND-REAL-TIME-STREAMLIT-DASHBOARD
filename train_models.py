"""
train_models.py
Trains and evaluates multiple Machine Learning classifiers for Network Intrusion Detection:
  1. Random Forest (Binary & Multi-class)
  2. Decision Tree
  3. K-Nearest Neighbors (KNN)
  4. Logistic Regression

Uses ml_engine (Pure-NumPy) for 100% platform portability and zero blocked binary dependencies.
"""

import os
import csv
import json
import pickle
import numpy as np

from ml_engine import (
    StandardScaler, LabelEncoder, RandomForestClassifier,
    DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression,
    compute_metrics
)

def load_csv_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    return header, rows

def train_and_evaluate(data_dir="data", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    train_path = os.path.join(data_dir, "train_nsl_kdd.csv")
    test_path = os.path.join(data_dir, "test_nsl_kdd.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset not found. Generating benchmark datasets...")
        from dataset_generator import save_benchmark_datasets
        save_benchmark_datasets(data_dir)

    print("Loading datasets from disk...")
    header, train_rows = load_csv_data(train_path)
    _, test_rows = load_csv_data(test_path)

    col_idx = {name: i for i, name in enumerate(header)}
    cat_cols = ['protocol_type', 'service', 'flag']
    drop_cols = ['attack_type', 'attack_category', 'is_intrusion']
    feature_cols = [c for c in header if c not in drop_cols]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    # Encoders
    encoders = {}
    for col in cat_cols:
        idx = col_idx[col]
        all_vals = list(set([r[idx] for r in train_rows] + [r[idx] for r in test_rows]))
        le = LabelEncoder().fit(all_vals)
        encoders[col] = le

    # Multi-class attack category encoder
    multi_idx = col_idx['attack_category']
    multi_le = LabelEncoder().fit([r[multi_idx] for r in train_rows])
    encoders['target_multi'] = multi_le

    def transform_rows(rows):
        X_mat = []
        for r in rows:
            row_features = []
            for col in feature_cols:
                idx = col_idx[col]
                val = r[idx]
                if col in cat_cols:
                    le = encoders[col]
                    row_features.append(le.mapping.get(val, 0))
                else:
                    row_features.append(float(val))
            X_mat.append(row_features)
        return np.array(X_mat, dtype=np.float64)

    print("Preprocessing & Scaling features...")
    train_X = transform_rows(train_rows)
    test_X = transform_rows(test_rows)

    bin_idx = col_idx['is_intrusion']
    train_y_bin = np.array([int(r[bin_idx]) for r in train_rows], dtype=np.int64)
    test_y_bin = np.array([int(r[bin_idx]) for r in test_rows], dtype=np.int64)

    train_y_multi = np.array([multi_le.mapping[r[multi_idx]] for r in train_rows], dtype=np.int64)
    test_y_multi = np.array([multi_le.mapping.get(r[multi_idx], 0) for r in test_rows], dtype=np.int64)

    # Standardize numerical features
    num_indices = [feature_cols.index(c) for c in num_cols]
    scaler = StandardScaler()
    train_X[:, num_indices] = scaler.fit_transform(train_X[:, num_indices])
    test_X[:, num_indices] = scaler.transform(test_X[:, num_indices])

    # Save encoders & scaler
    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(models_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)

    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump({
            "features": feature_cols,
            "categorical": cat_cols,
            "numerical": num_cols
        }, f, indent=2)

    # Train Models
    print("\n--- Training Supervised Classifiers ---")
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=30, max_depth=12, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(lr=0.08, n_iters=350)
    }

    metrics_results = {}
    trained_estimators = {}

    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(train_X, train_y_bin)
        preds = clf.predict(test_X)
        metrics = compute_metrics(test_y_bin, preds)
        metrics_results[name] = metrics
        trained_estimators[name] = clf
        print(f" -> {name}: Accuracy={metrics['accuracy']}%, F1={metrics['f1_score']}%")

    # Feature Importances
    rf = trained_estimators["Random Forest"]
    feat_imp = sorted(
        [{"feature": f, "importance": round(float(imp), 4)} for f, imp in zip(feature_cols, rf.feature_importances_)],
        key=lambda x: x["importance"],
        reverse=True
    )
    metrics_results["feature_importances"] = feat_imp

    # Train Multi-Class Random Forest
    print("\n--- Training Multi-Class Attack Classifier (Random Forest) ---")
    rf_multi = RandomForestClassifier(n_estimators=25, max_depth=12, random_state=42)
    rf_multi.fit(train_X, train_y_multi)
    multi_preds = rf_multi.predict(test_X)
    multi_metrics = compute_metrics(test_y_multi, multi_preds)
    multi_classes = multi_le.classes_

    metrics_results["multi_class"] = {
        "accuracy": multi_metrics["accuracy"],
        "classes": multi_classes,
        "confusion_matrix": multi_metrics["confusion_matrix"]
    }
    print(f"Multi-class Classifier Accuracy: {multi_metrics['accuracy']}%")

    # Serialize trained models
    print("\nSaving trained models to disk...")
    with open(os.path.join(models_dir, "ids_rf_binary.pkl"), "wb") as f:
        pickle.dump(trained_estimators["Random Forest"], f)
    with open(os.path.join(models_dir, "ids_dt_binary.pkl"), "wb") as f:
        pickle.dump(trained_estimators["Decision Tree"], f)
    with open(os.path.join(models_dir, "ids_lr_binary.pkl"), "wb") as f:
        pickle.dump(trained_estimators["Logistic Regression"], f)
    with open(os.path.join(models_dir, "ids_rf_multi.pkl"), "wb") as f:
        pickle.dump(rf_multi, f)

    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics_results, f, indent=2)

    print(f"All models & metrics successfully saved to '{models_dir}/'!")
    return metrics_results

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_and_evaluate(
        data_dir=os.path.join(base_dir, "data"),
        models_dir=os.path.join(base_dir, "models")
    )
