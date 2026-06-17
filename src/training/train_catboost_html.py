"""
train_catboost_html.py
===========================
Train HTML-only CatBoost model cho URL phishing detection.

Label:
    0 = phishing
    1 = legitimate

Nguyên tắc:
    - Chỉ dùng HTML features cũ
    - Chỉ train trên rows fetch HTML thành công
    - Không fill NaN = -1
    - Group split theo registered domain từ cột url
"""

import os
import re
import json
import joblib
import tldextract
import numpy as np
import pandas as pd

from pathlib import Path
from collections import Counter

from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# =======================
# CONFIG
# =======================
SEED = 42
TEST_SIZE = 0.2

DATA_PATH = "/content/drive/MyDrive/Đồ án tốt nghiệp/Datasets/urls_extracted_clean.csv"

MODEL_DIR = "/content/drive/MyDrive/Đồ án tốt nghiệp/Models"
# MODEL_PATH = f"{MODEL_DIR}/catboost_html_only.cbm"
# FEATURES_PATH = f"{MODEL_DIR}/html_features.json"

MODEL_PATH = "catboost_html_only.cbm"
FEATURES_PATH = "html_features.json"


LABEL_COL = "label"

URL_CANDIDATES = [
    "url", "URL", "Url",
    "domain", "Domain",
    "raw_url", "original_url",
]

HTML_FEATURES = [
    "NoOfExternalRef",
    "NoOfSelfRef",
    "LineOfCode",
    "NoOfImage",
    "LargestLineLength",
    "HasDescription",
    "HasSocialNet",
    "NoOfJS",
    "URLTitleMatchScore",
]


# =======================
# HELPERS
# =======================
def find_url_col(df: pd.DataFrame):
    return next((c for c in URL_CANDIDATES if c in df.columns), None)


def get_group_key(x):
    if pd.isna(x):
        return None

    x = str(x).strip()

    if not x:
        return None

    if "://" not in x and "@" not in x:
        x = "http://" + x

    try:
        ext = tldextract.extract(x)

        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"

        if ext.domain:
            return ext.domain

        return None

    except Exception:
        return None


def get_class_weights(y):
    counter = Counter(y)
    total = len(y)

    return {
        0: total / (2 * counter[0]),
        1: total / (2 * counter[1]),
    }


def get_phishing_proba(model, pool):
    classes = list(model.classes_)
    phish_idx = classes.index(0)
    return model.predict_proba(pool)[:, phish_idx]


def evaluate(model, test_pool, y_test, threshold=0.5):
    p_phish = get_phishing_proba(model, test_pool)

    y_pred = np.where(p_phish >= threshold, 0, 1)

    print("\n" + "=" * 60)
    print("HTML-ONLY EVALUATION")
    print("=" * 60)

    print(f"Threshold phishing : {threshold:.2f}")
    print(f"Accuracy           : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC            : {roc_auc_score((y_test == 0).astype(int), p_phish):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix [labels: 0 phishing, 1 legit]:")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


def threshold_sweep(model, test_pool, y_test):
    p_phish = get_phishing_proba(model, test_pool)

    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP FOR PHISHING CLASS")
    print("=" * 60)

    for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        y_pred = np.where(p_phish >= th, 0, 1)

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )

        phish_precision = report["0"]["precision"]
        phish_recall = report["0"]["recall"]
        phish_f1 = report["0"]["f1-score"]

        print(
            f"th={th:.2f} | "
            f"phish_precision={phish_precision:.4f} | "
            f"phish_recall={phish_recall:.4f} | "
            f"phish_f1={phish_f1:.4f}"
        )


# =======================
# MAIN
# =======================
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Reading dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("\nRaw dataset info:")
    print(df.info())

    missing_features = [c for c in HTML_FEATURES if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing HTML features: {missing_features}")

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing label column: {LABEL_COL}")

    url_col = find_url_col(df)

    if url_col:
        print(f"\nFound URL column: {url_col}")
    else:
        print("\n⚠ Không tìm thấy cột URL/domain, fallback random split.")

    # Clean label
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print("\nLabel distribution before HTML filtering:")
    print(df[LABEL_COL].value_counts())

    # Convert HTML features to numeric
    for col in HTML_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nMissing ratio before filtering:")
    print(df[HTML_FEATURES].isna().mean().sort_values(ascending=False))

    # =======================
    # IMPORTANT:
    # Train HTML model only on rows with successful HTML extraction
    # =======================
    html_success_mask = df[HTML_FEATURES].notna().all(axis=1)
    df_html = df[html_success_mask].copy()

    print("\nHTML rows kept:")
    print(f"{len(df_html):,} / {len(df):,} rows")
    print(f"Retention rate: {len(df_html) / len(df):.4f}")

    print("\nLabel distribution after HTML filtering:")
    print(df_html[LABEL_COL].value_counts())

    if df_html[LABEL_COL].nunique() < 2:
        raise ValueError("Sau khi lọc HTML success, dataset chỉ còn 1 class. Không train được.")

    # Features / target
    X = df_html[HTML_FEATURES].copy()
    y = df_html[LABEL_COL].copy()

    # No NaN should remain
    assert X.isna().sum().sum() == 0, "HTML features vẫn còn NaN sau filtering."

    # =======================
    # SPLIT
    # =======================
    if url_col:
        groups = df_html[url_col].apply(get_group_key)

        n_none = groups.isna().sum()
        if n_none:
            print(f"\n⚠ {n_none:,} rows không parse được group key -> dùng row-specific unknown group.")
            groups = groups.where(groups.notna(), "unknown_" + df_html.index.astype(str))

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=SEED,
        )

        train_idx, test_idx = next(gss.split(X, y, groups))

        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        leak = train_groups & test_groups

        print("\nGroup split result:")
        print(f"Train groups : {len(train_groups):,}")
        print(f"Test groups  : {len(test_groups):,}")
        print(f"Group leak   : {len(leak):,}")

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=SEED,
            stratify=y,
        )

    print("\nTrain label distribution:")
    print(y_train.value_counts())

    print("\nTest label distribution:")
    print(y_test.value_counts())

    class_weights = get_class_weights(y_train)

    print("\nClass weights:")
    print(class_weights)

    # =======================
    # CATBOOST POOLS
    # =======================
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    # =======================
    # MODEL
    # =======================
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=5,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_seed=SEED,
        early_stopping_rounds=150,
        verbose=100,
    )

    # =======================
    # TRAIN
    # =======================
    model.fit(
        train_pool,
        eval_set=test_pool,
        plot=False,
    )

    # =======================
    # EVALUATE
    # =======================
    evaluate(model, test_pool, y_test, threshold=0.5)
    threshold_sweep(model, test_pool, y_test)

    # =======================
    # FEATURE IMPORTANCE
    # =======================
    fi = pd.DataFrame({
        "feature": HTML_FEATURES,
        "importance": model.get_feature_importance(train_pool),
    }).sort_values("importance", ascending=False)

    print("\nFeature importance:")
    print(fi)

    # =======================
    # SAVE
    # =======================
    model.save_model(MODEL_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(HTML_FEATURES, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"Model    : {MODEL_PATH}")
    print(f"Features : {FEATURES_PATH}")


if __name__ == "__main__":
    main()