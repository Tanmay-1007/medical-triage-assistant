# train_model.py
import os
import json
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from rules import triage_rules

SYN_PATH = "synthetic_vignettes.csv"
LOG_PATH = os.path.join("logs", "triage_logs.csv")
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.json"
META_PATH = "model_meta.json"
FIG_DIR = "model_figs"
os.makedirs(FIG_DIR, exist_ok=True)

CATEGORIES = ["Emergency", "Urgent", "Routine", "Self-care"]


def load_data():
    if not os.path.exists(SYN_PATH):
        print(f"{SYN_PATH} not found — generating synthetic data.")
        from data_gen import generate_csv
        generate_csv(SYN_PATH, n=600)

    syn = pd.read_csv(SYN_PATH)

    labels = []
    for _, row in syn.iterrows():
        symptoms = row["symptoms"].split(";") if pd.notna(row["symptoms"]) else []
        redflags = row["red_flags"].split(";") if pd.notna(row.get("red_flags", "")) else []
        age = int(row["age"])
        dur = int(row["duration_days"])
        cat, _, _ = triage_rules(symptoms, redflags, age, dur)
        labels.append(cat)
    syn["label"] = labels

    if os.path.exists(LOG_PATH):
        logs = pd.read_csv(LOG_PATH)
        logs = logs.rename(columns={"triage_category": "label"})

        def age_group_to_num(g):
            if pd.isna(g): return 30
            if g == "<2": return 1
            if g == "2-11": return 6
            if g == "12-17": return 14
            if g == "18-29": return 24
            if g == "30-44": return 37
            if g == "45-64": return 54
            if g == "65+": return 75
            return 30

        if "age_group" in logs.columns:
            logs["age"] = logs["age_group"].apply(age_group_to_num)
        else:
            logs["age"] = 30

        logs["symptoms"] = logs["symptoms"].fillna("")
        logs["red_flags"] = logs["red_flags"].fillna("")
        logs["duration_days"] = logs["duration_days"].fillna(1).astype(int)

        df = pd.concat([
            syn[["age", "symptoms", "red_flags", "duration_days", "label"]],
            logs[["age", "symptoms", "red_flags", "duration_days", "label"]]
        ], ignore_index=True)
    else:
        syn["age"] = syn["age"].astype(int)
        df = syn[["age", "symptoms", "red_flags", "duration_days", "label"]].copy()

    df["symptoms"] = df["symptoms"].fillna("").apply(lambda x: [s.strip() for s in x.split(";") if s.strip()])
    df["red_flags"] = df["red_flags"].fillna("").apply(lambda x: [s.strip() for s in x.split(";") if s.strip()])
    return df


def featurize(df):
    all_sym = sorted(set(s for row in df["symptoms"] for s in row))
    all_rf = sorted(set(s for row in df["red_flags"] for s in row))

    mlb_sym = MultiLabelBinarizer(classes=all_sym)
    X_sym = pd.DataFrame(mlb_sym.fit_transform(df["symptoms"]),
                         columns=[f"sym_{s}" for s in mlb_sym.classes_])

    mlb_rf = MultiLabelBinarizer(classes=all_rf)
    X_rf = pd.DataFrame(mlb_rf.fit_transform(df["red_flags"]),
                        columns=[f"rf_{s}" for s in mlb_rf.classes_])

    X = pd.concat([X_sym, X_rf], axis=1)
    X["age"] = df["age"].astype(int).values
    X["duration_days"] = df["duration_days"].astype(int).values

    y = df["label"].astype(str).values

    vectorizer = {
        "mlb_sym_classes": mlb_sym.classes_.tolist(),
        "mlb_rf_classes": mlb_rf.classes_.tolist(),
        "feature_columns": X.columns.tolist()
    }
    return X, y, vectorizer


def train_and_save():
    df = load_data()
    X, y, vectorizer = featurize(df)

    label2idx = {c: i for i, c in enumerate(CATEGORIES)}
    idx2label = {v: k for k, v in label2idx.items()}
    y_idx = np.array([label2idx.get(lbl, label2idx["Routine"]) for lbl in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx
    )

    clf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[idx2label[i] for i in sorted(idx2label.keys())]
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(idx2label.keys()))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[idx2label[i] for i in sorted(idx2label.keys())],
                yticklabels=[idx2label[i] for i in sorted(idx2label.keys())])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig_path_cm = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(fig_path_cm, bbox_inches="tight")
    plt.close(fig)

    # Feature importances
    importances = clf.feature_importances_
    feat_names = X.columns
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances}) \
        .sort_values("importance", ascending=False).head(40)

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.barplot(x="importance", y="feature", data=imp_df, ax=ax2)
    ax2.set_title("Top feature importances")
    fig_path_imp = os.path.join(FIG_DIR, "feature_importances.png")
    fig2.tight_layout()
    fig2.savefig(fig_path_imp, bbox_inches="tight")
    plt.close(fig2)

    # Save artifacts
    joblib.dump(clf, MODEL_PATH)
    with open(VECT_PATH, "w", encoding="utf-8") as f:
        json.dump(vectorizer, f, indent=2)
    meta = {
        "label2idx": label2idx,
        "idx2label": idx2label,
        "trained_on": datetime.utcnow().isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test)
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved model to {MODEL_PATH}, vectorizer to {VECT_PATH}, meta to {META_PATH}")
    print("Saved figures to", FIG_DIR)
    print("Training complete.")


if __name__ == "__main__":
    train_and_save()
