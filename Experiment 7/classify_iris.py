import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

RANDOM_STATE = 42
COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

df = pd.read_csv("iris_data.csv", header=None, names=COLS)
X, y_raw = df.drop(columns="species"), df["species"]
le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
scaler = StandardScaler()
X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM (RBF kernel)": SVC(kernel="rbf", random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_results = {name: cross_val_score(m, X_train_s, y_train, cv=cv).mean() for name, m in models.items()}
print("Cross-validation accuracy (5-fold):")
for name, score in cv_results.items():
    print(f"  {name:22s} {score:.4f}")

best_name = max(cv_results, key=cv_results.get)
best_model = models[best_name]
best_model.fit(X_train_s, y_train)
y_pred = best_model.predict(X_test_s)
test_acc = accuracy_score(y_test, y_pred)

print(f"\nBest model: {best_name}")
print(f"Test accuracy: {test_acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion Matrix — {best_name} ({test_acc:.2%})")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close("all")

if hasattr(best_model, "feature_importances_"):
    print(pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).to_string())

joblib.dump({"model": best_model, "scaler": scaler, "label_encoder": le, "model_name": best_name}, "iris_classifier.joblib")
print("\nSaved iris_classifier.joblib and confusion_matrix.png")
