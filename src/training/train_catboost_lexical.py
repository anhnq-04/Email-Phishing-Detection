"""
train_catboost_lexical.py
=========================
Layer 2 - Lexical-only CatBoost model cho URL phishing detection.

Không dùng HTML-dependent features để tránh crawl/missing leakage.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
import tldextract


# =======================
# CONFIG
# =======================
SEED = 42
TEST_SIZE = 0.2

LABEL_COL = "label"

LEXICAL_FEATURES = [
    "IsHTTPS",
    "LetterRatioInURL",
    "NoOfSubDomain",
    "DegitRatioInURL",          # nếu có thể, sau này đổi tên thành DigitRatioInURL
    "SpacialCharRatioInURL",    # nếu có thể, sau này đổi tên thành SpecialCharRatioInURL
    "DomainLength",
]

HTML_FEATURES_TO_DROP = [
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
def get_group_key(x):
    """
    Group split theo registered domain nếu có URL/domain gốc.
    """
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


def find_url_col(df):
    candidates = [
        "URL", "url", "Url",
        "domain", "Domain",
        "raw_url", "RawURL",
        "original_url",
    ]
    return next((c for c in candidates if c in df.columns), None)


# =======================
# LOAD DATA
# =======================
df = pd.read_csv(
    "/content/drive/MyDrive/Đồ án tốt nghiệp/Datasets/urls_lexical_extracted.csv",
    low_memory=False,
)

print(df.info())
print("\nLabel distribution:")
print(df[LABEL_COL].value_counts(dropna=False))

missing_features = [c for c in LEXICAL_FEATURES if c not in df.columns]
if missing_features:
    raise ValueError(f"Missing lexical features: {missing_features}")

df = df.dropna(subset=[LABEL_COL]).copy()
df[LABEL_COL] = df[LABEL_COL].astype(int)

X = df[LEXICAL_FEATURES].copy()
y = df[LABEL_COL].copy()

# convert object/mixed columns safely
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

print("\nMissing ratio in lexical features:")
print(X.isna().mean().sort_values(ascending=False))

# lexical features không nên NaN nhiều; fill bằng median theo train sau split sẽ tốt hơn,
# nhưng trước split cần tạm giữ X nguyên.


# =======================
# SPLIT
# =======================
URL_COL = find_url_col(df)

if URL_COL:
    print(f"\nDùng group split theo cột '{URL_COL}'...")

    groups = df[URL_COL].apply(get_group_key)
    n_none = groups.isna().sum()

    if n_none:
        print(f"⚠ {n_none:,} rows không parse được group key -> dùng row-specific unknown group")
        groups = groups.where(groups.notna(), "unknown_" + df.index.astype(str))

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

    print(f"Train groups: {len(train_groups):,}")
    print(f"Test groups : {len(test_groups):,}")
    print(f"Group leak  : {len(leak):,}")

else:
    print("\n⚠ Không tìm thấy cột URL/domain gốc -> dùng stratified random split.")
    print("⚠ Metric từ random split KHÔNG đủ tin cậy cho robustness.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )


# =======================
# IMPUTE LEXICAL NaN
# =======================
medians = X_train.median(numeric_only=True)

X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

# fallback nếu một cột toàn NaN
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# =======================
# CLASS WEIGHTS
# =======================
counter = Counter(y_train)
total = len(y_train)

class_weights = {
    0: total / (2 * counter[0]),
    1: total / (2 * counter[1]),
}

print("\nTrain label distribution:")
print(y_train.value_counts())

print("\nClass weights:")
print(class_weights)


# =======================
# POOL
# =======================
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)


# =======================
# MODEL
# =======================
model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.01,
    depth=5,
    loss_function="Logloss",
    eval_metric="AUC",
    # class_weights=class_weights,
    random_seed=SEED,
    early_stopping_rounds=100,
    verbose=100,
)


# =======================
# TRAIN + EVALUATE
# =======================
if __name__ == "__main__":
    model.fit(
        train_pool,
        eval_set=test_pool,
        plot=False,
    )

    y_pred = model.predict(test_pool)
    y_proba = model.predict_proba(test_pool)[:, 1]

    print("\n" + "=" * 50)
    print("LEXICAL-ONLY EVALUATION")
    print("=" * 50)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nFeature importance:")
    fi = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.get_feature_importance(train_pool),
    }).sort_values("importance", ascending=False)

    print(fi)