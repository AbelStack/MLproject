"""
Practical Engineering Case Study: Supervised Machine Learning Classification
Algorithms: SVM, Gaussian Naive Bayes, Decision Tree, Random Forest, AdaBoost
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, ConfusionMatrixDisplay
)

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# PART 1. CREATE A NEW DATASET (240 RECORDS)

n_per_class = 120

normal = pd.DataFrame({
    "Temperature_C": rng.normal(63, 11, n_per_class).clip(35, 100),
    "Vibration_mm_s": rng.normal(2.5, 1.0, n_per_class).clip(0.3, 7.0),
    "Speed_RPM": rng.normal(1480, 65, n_per_class).clip(1150, 1650),
    "Operating_Hours": rng.normal(3200, 1900, n_per_class).clip(100, 9000),
    "Failure_Risk": 0
})

failure_risk = pd.DataFrame({
    "Temperature_C": rng.normal(79, 13, n_per_class).clip(40, 115),
    "Vibration_mm_s": rng.normal(4.7, 1.6, n_per_class).clip(0.5, 10.0),
    "Speed_RPM": rng.normal(1390, 85, n_per_class).clip(1050, 1600),
    "Operating_Hours": rng.normal(5400, 2100, n_per_class).clip(300, 10000),
    "Failure_Risk": 1
})

df = pd.concat([normal, failure_risk], ignore_index=True)
df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
df.insert(0, "Machine_ID", np.arange(1, len(df) + 1))
df["Temperature_C"] = df["Temperature_C"].round(1)
df["Vibration_mm_s"] = df["Vibration_mm_s"].round(2)
df["Speed_RPM"] = df["Speed_RPM"].round(0).astype(int)
df["Operating_Hours"] = df["Operating_Hours"].round(0).astype(int)

print("\nNEW DATASET: MACHINE CONDITION RECORDS")
print(df.head(10).to_string(index=False))
print(f"\nTotal records: {len(df)}")
print("\nClass distribution:")
print(df["Failure_Risk"].value_counts().rename(index={0: "Normal", 1: "Failure risk"}))

feature_cols = [
    "Temperature_C", "Vibration_mm_s", "Speed_RPM", "Operating_Hours"
]
X = df[feature_cols]
y = df["Failure_Risk"]


# PART 2. EXPLORE AND VISUALIZE THE DATA


sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df, x="Temperature_C", y="Vibration_mm_s",
    hue="Failure_Risk", style="Failure_Risk", alpha=0.8
)
plt.title("Machine Temperature vs. Vibration")
plt.xlabel("Temperature (°C)")
plt.ylabel("Vibration (mm/s)")
plt.legend(title="0 = Normal, 1 = Failure risk")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.heatmap(
    df[feature_cols + ["Failure_Risk"]].corr(),
    annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# PART 3. SPLIT THE DATA


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTraining records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")


# PART 4. DEFINE FIVE CLASSIFICATION MODELS

svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(kernel="rbf", C=1.0, gamma="scale"))
])

naive_bayes_model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GaussianNB())
])

decision_tree_model = DecisionTreeClassifier(
    max_depth=4, min_samples_leaf=3, random_state=RANDOM_STATE
)

random_forest_model = RandomForestClassifier(
    n_estimators=150, max_depth=6, min_samples_leaf=2,
    random_state=RANDOM_STATE, n_jobs=-1
)


adaboost_model = AdaBoostClassifier(
    n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE
)

models = {
    "SVM": svm_model,
    "Naive Bayes": naive_bayes_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "AdaBoost": adaboost_model
}


# PART 5. TRAIN, PREDICT, AND EVALUATE


results = []
predictions = {}

for model_name, model in models.items():
    print("\n" + "=" * 65)
    print(model_name.upper())
    print("=" * 65)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Algorithm": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1-score : {f1:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure risk"],
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Normal", "Failure risk"],
        cmap="Blues", colorbar=False
    )
    plt.title(f"{model_name}: Confusion Matrix")
    plt.tight_layout()
    plt.show()


# PART 6. COMPARE THE FIVE ALGORITHMS


results_df = pd.DataFrame(results).set_index("Algorithm")
print("\nMODEL COMPARISON")
print(results_df.round(3).to_string())

results_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
plt.title("Comparison of Supervised Learning Algorithms")
plt.ylabel("Score")
plt.xlabel("Algorithm")
plt.xticks(rotation=20)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

best_model_name = results_df["F1-score"].idxmax()
print(f"\nBest model by F1-score on this test split: {best_model_name}")


# PART 7. FEATURE IMPORTANCE (TREE-BASED MODELS)


tree_models = {
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}

for name, model in tree_models.items():
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)

    print(f"\n{name} feature importance:")
    print(importance.sort_values(ascending=False).round(3).to_string())

    plt.figure(figsize=(7, 4))
    importance.plot(kind="barh")
    plt.title(f"{name}: Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Machine feature")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(16, 8))
plot_tree(
    decision_tree_model,
    feature_names=feature_cols,
    class_names=["Normal", "Failure risk"],
    filled=True, rounded=True, fontsize=8
)
plt.title("Decision Tree Learned from Training Data")
plt.tight_layout()
plt.show()


# PART 8. PREDICT A NEW MACHINE OBSERVATION

new_machine = pd.DataFrame([{
    "Temperature_C": 84.0,
    "Vibration_mm_s": 5.2,
    "Speed_RPM": 1360,
    "Operating_Hours": 6100
}])

print("\nNEW MACHINE OBSERVATION")
print(new_machine.to_string(index=False))

for name, model in models.items():
    predicted_class = int(model.predict(new_machine)[0])
    label = "Failure risk" if predicted_class == 1 else "Normal"
    print(f"{name:15s} prediction: {label}")

print("\nEND OF PRACTICAL CASE STUDY")
