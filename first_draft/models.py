import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

df = pd.read_csv("../eda/Datasets/features.csv")

keep_cols = [
    "gameDuration",
    "gold_ratio", "kills_ratio", "towers_ratio", "dragons_ratio", "barons_ratio",
    "gold_diff_norm", "kills_diff_norm", "towers_diff_norm",
    "dragons_diff_norm", "barons_diff_norm",
    "either_throw"
]

df = df[keep_cols]
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(0)
'''
print(df.columns.tolist())
df.to_csv('reduced.csv', index = False)
'''

X = df.drop(columns=["either_throw"])
y = df["either_throw"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=67, stratify=y
)

# Define models
models = {
    "No Throw": DummyClassifier(strategy="constant", constant=0),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15)
    ),
    "Decision Tree": DecisionTreeClassifier(max_depth=6),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=67)
}
results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
        "ROC-AUC": roc_auc_score(y_test, probs),
    }
results = pd.DataFrame(results).T
print(results)

# Visualizations
best_model = models["Random Forest"]

# Confusion Matrix
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
plt.title("Confusion Matrix – Throw vs No Throw")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# ROC Curve
RocCurveDisplay.from_estimator(best_model, X_test, y_test)
plt.title("ROC Curve – Throw Classifier")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
plt.close()

# Feature Importance (Random Forest)
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), X.columns[indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=300, bbox_inches='tight')
plt.close()