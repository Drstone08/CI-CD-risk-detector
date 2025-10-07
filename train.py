"""
train_risk_model.py
--------------------
Trains a simple risk classifier using commit metadata and diff features.

Input:
    commits_data_clean.csv

Output:
    risk_model.pkl — trained model
    feature_importances.png — visualization
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# 1️⃣ LOAD DATA
# ===============================
df = pd.read_csv("commits_data_clean.csv")

# Fill missing
df = df.fillna("")

# ===============================
# 2️⃣ FEATURE ENGINEERING
# ===============================

def contains_risky_keywords(text):
    risky_words = ["curl", "wget", "chmod", "scp", "token", "secret", "ssh", "eval", "bash", "powershell"]
    text_lower = text.lower()
    return int(any(word in text_lower for word in risky_words))

def contains_secret_like(text):
    # Detect strings that look like tokens, keys, or secrets
    return int(bool(re.search(r"(?i)(api[_-]?key|access[_-]?token|secret|AKIA[0-9A-Z]{16})", text)))

def is_ci_file(filename):
    return int(bool(re.search(r"(\.github/workflows|ci\.yml|jenkinsfile|gitlab-ci\.yml)", filename, re.I)))

def is_script_file(filename):
    return int(bool(re.search(r"\.(sh|py|ps1|bat|yml|yaml|json|tf|dockerfile)$", filename, re.I)))

def extract_commit_hour(timestamp):
    try:
        dt = pd.to_datetime(timestamp, errors="coerce")
        return dt.hour if pd.notnull(dt) else -1
    except Exception:
        return -1

df["num_additions"] = pd.to_numeric(df["additions"], errors="coerce").fillna(0)
df["num_deletions"] = pd.to_numeric(df["deletions"], errors="coerce").fillna(0)
df["commit_hour"] = df["timestamp"].apply(extract_commit_hour)
df["is_ci_file"] = df["filename"].apply(is_ci_file)
df["is_script_file"] = df["filename"].apply(is_script_file)
df["contains_risky_keywords"] = df["patch"].apply(contains_risky_keywords)
df["contains_secret_like"] = df["patch"].apply(contains_secret_like)

# ===============================
# 3️⃣ SIMPLE RISK LABELING (for supervised training)
# ===============================
def label_risky(row):
    # You can tune these rules based on your data
    if row["contains_secret_like"] or row["contains_risky_keywords"]:
        return 1
    if row["is_ci_file"] and (row["num_additions"] + row["num_deletions"]) > 20:
        return 1
    if 0 <= row["commit_hour"] <= 5:  # odd hours
        return 1
    return 0

df["risky_label"] = df.apply(label_risky, axis=1)

# ===============================
# 4️⃣ PREPARE FEATURES FOR MODEL
# ===============================
features = [
    "num_additions",
    "num_deletions",
    "commit_hour",
    "is_ci_file",
    "is_script_file",
    "contains_risky_keywords",
    "contains_secret_like"
]
X = df[features]
y = df["risky_label"]

# ===============================
# 5️⃣ TRAIN-TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ===============================
# 6️⃣ TRAIN MODEL
# ===============================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===============================
# 7️⃣ EVALUATE
# ===============================
y_pred = model.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# ===============================
# 8️⃣ SAVE MODEL
# ===============================
joblib.dump(model, "risk_model.pkl")
print("\n✅ Model saved as risk_model.pkl")

# ===============================
# 9️⃣ FEATURE IMPORTANCE PLOT
# ===============================
importances = model.feature_importances_
plt.figure(figsize=(8,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png")
print("📊 Saved feature_importances.png")
