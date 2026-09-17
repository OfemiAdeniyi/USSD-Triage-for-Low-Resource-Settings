import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("synthetic_triage_data.csv")

X = pd.get_dummies(df.drop(columns=["label"]), columns=["who"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

print("=== Model A: default (no class weighting) ===")
clf_default = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_default.fit(X_train, y_train)
pred_default = clf_default.predict(X_test)
print(classification_report(y_test, pred_default, digits=3))

print("\n=== Model B: class_weight='balanced' ===")
clf_balanced = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
clf_balanced.fit(X_train, y_train)
pred_balanced = clf_balanced.predict(X_test)
print(classification_report(y_test, pred_balanced, digits=3))

print("\n=== Confusion matrix, Model B (rows=true, cols=predicted) ===")
labels = ["RED", "YELLOW", "GREEN"]
cm = confusion_matrix(y_test, pred_balanced, labels=labels)
print(pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels]))

import joblib
joblib.dump(clf_balanced, "triage_tree_balanced.joblib")

print("\n=== Learned tree (Model B), as readable rules ===")
print(export_text(clf_balanced, feature_names=list(X.columns)))
