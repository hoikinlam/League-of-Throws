import warnings
warnings.filterwarnings("ignore")  # silence sklearn metric warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
)

# Load data and define features / target
df = pd.read_csv("features.csv")

target_col = "either_throw"

# Feature set
feature_cols = [
    "gameDuration",
    "gold_ratio", "kills_ratio", "towers_ratio",
    "dragons_ratio", "barons_ratio",
    "gold_diff_norm", "kills_diff_norm", "towers_diff_norm",
    "dragons_diff_norm", "barons_diff_norm",
]

# keep only columns that exist
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()
y = df[target_col].copy()

# Replace infinities / NaNs with zeros
X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
print("Throw rate (train):", y_train.mean())
print("Throw rate (test):", y_test.mean())

# Define models with class rebalancing
models = {
    "No Throw (Dummy)": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear"
        )
    ),

    "KNN (k=15)": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15)
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),
}

# Train / evaluate models
results = []

for name, model in models.items():
    print("\n==============================")
    print(f"Training model: {name}")
    print("==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # probability scores if available
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    acc = accuracy_score(y_test, y_pred)
    f1_throw = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    roc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan

    results.append([name, acc, f1_throw, f1_macro, roc])

    print(classification_report(y_test, y_pred, digits=3))

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "F1 (throw)", "F1 (average)", "ROC-AUC"]
)

print("\n=== Baseline Model Comparison ===")
print(results_df.sort_values(by="F1 (throw)", ascending=False))

# Hyperparameter tuning for Random Forest
# uses train split only; evaluated on test split
print("\n==============================")
print("Hyperparameter tuning: Random Forest")
print("==============================")

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

param_dist = {
    "n_estimators": [200, 300, 400],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}

rf_search = RandomizedSearchCV(
    rf_base,
    param_distributions=param_dist,
    n_iter=20,
    scoring="f1",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)

rf_search.fit(X_train, y_train)

print("\nBest RF params:", rf_search.best_params_)
print("Best CV F1 (throw):", rf_search.best_score_)

best_rf = rf_search.best_estimator_

# Evaluate tuned RF on test set
y_pred_rf = best_rf.predict(X_test)
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred_rf)
f1_throw_rf = f1_score(y_test, y_pred_rf, pos_label=1, zero_division=0)
f1_macro_rf = f1_score(y_test, y_pred_rf, average="macro", zero_division=0)
roc_rf = roc_auc_score(y_test, y_proba_rf)
ap_rf = average_precision_score(y_test, y_proba_rf)

print("\n=== Random Forest on Test Set ===")
print(classification_report(y_test, y_pred_rf, digits=3))
print(f"Accuracy:   {acc_rf:.4f}")
print(f"F1 (throw): {f1_throw_rf:.4f}")
print(f"F1 (average): {f1_macro_rf:.4f}")
print(f"ROC-AUC:    {roc_rf:.4f}")
print(f"PR-AUC (AP):{ap_rf:.4f}")

# Add tuned RF to comparison table
rf_row = pd.DataFrame([{
    "Model": "Random Forest (tuned)",
    "Accuracy": acc_rf,
    "F1 (throw)": f1_throw_rf,
    "F1 (average)": f1_macro_rf,
    "ROC-AUC": roc_rf,
}])

results_df = pd.concat([results_df, rf_row], ignore_index=True)

print("\n=== Final Model Comparison ===")
print(results_df.sort_values(by="F1 (throw)", ascending=False))

# Final model = tuned Random Forest
# plots: confusion matrix, PR curve, ROC curve, feature importance
final_model = best_rf
final_name = "Random Forest"
print("\nUsing final model:", final_name)

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(final_model, X_test, y_test)
plt.title(f"Confusion Matrix – {final_name} (Throw vs No Throw)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

# Precision–Recall Curve
precision, recall, thresholds = precision_recall_curve(y_test, y_proba_rf)
plt.figure(figsize=(6, 5))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall Curve – {final_name} (AP = {ap_rf:.3f})")
plt.tight_layout()
plt.savefig("pr_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# ROC Curve
RocCurveDisplay.from_estimator(final_model, X_test, y_test)
plt.title(f"ROC Curve – {final_name}")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Feature Importances
importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), np.array(feature_cols)[indices], rotation=90)
plt.title(f"Feature Importances – {final_name}")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=300, bbox_inches="tight")
plt.close()
