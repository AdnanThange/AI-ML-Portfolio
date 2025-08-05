<<<<<<< HEAD
=======
# heart_disease_pipeline.py

>>>>>>> 5fd550b (Add Handwritten Digit Recognition project)
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("heart.csv")

label_encoders = {
    "Sex": LabelEncoder(),
    "ChestPainType": LabelEncoder(),
    "RestingECG": LabelEncoder(),
    "ExerciseAngina": LabelEncoder(),
    "ST_Slope": LabelEncoder()
}

for col, encoder in label_encoders.items():
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns="HeartDisease")
y = df["HeartDisease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, train_size=0.8, random_state=42
)

models_config = {
    "Random Forest": {
        "model": RandomForestClassifier(),
        "params": {
            "max_depth": [1, 2, 3, None],
            "min_samples_leaf": [10, 30, 40],
            "min_samples_split": [10, 20, 30]
        }
    },
    "SVM": {
        "model": SVC(),
        "params": {
            "C": [1.0, 0.1, 0.8],
            "kernel": ["linear", "rbf", "poly"]
        }
    },
    "Logistic Regression": {
        "model": LogisticRegression(),
        "params": {
            "solver": ["liblinear"],
            "C": [1.0, 0.1, 0.8]
        }
    }
}

best_models = []

for name, cfg in models_config.items():
    search = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=15, n_jobs=-1)
    search.fit(X_train, y_train)
    best_models.append({
        "model_name": name,
        "best_estimator": search.best_estimator_,
        "best_score": search.best_score_
    })

best_models_sorted = sorted(best_models, key=lambda x: x["best_score"], reverse=True)
final_model = best_models_sorted[0]["best_estimator"]
best_model_name = best_models_sorted[0]["model_name"]

print(f"\n Best Model: {best_model_name}")
<<<<<<< HEAD
print(f"Best Cross-Validation Score: {best_models_sorted[0]['best_score']:.4f}")
=======
print(f"🔍 Best Cross-Validation Score: {best_models_sorted[0]['best_score']:.4f}")
>>>>>>> 5fd550b (Add Handwritten Digit Recognition project)

y_pred = final_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Disease", "Disease"]))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
