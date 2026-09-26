import os
import joblib
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shiva\OneDrive\Desktop\Amazon_ML_Challenge_2026"

TRAINING_SAMPLE = os.path.join(
    BASE_DIR,
    "training_sample_10k.tsv"
)

MODEL_OUTPUT = os.path.join(
    BASE_DIR,
    "lightgbm_matching_model.txt"
)

VALIDATION_OUTPUT = os.path.join(
    BASE_DIR,
    "validation_predictions.tsv"
)


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "name_exact",
    "name_ratio",
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_token_overlap",
    "name_length_diff",
    "name_missing",
    "legal_suffix_difference",
    "word_order_difference",
    "same_script",

    "address_exact",
    "address_ratio",
    "address_token_sort",
    "address_token_set",
    "address_token_overlap",
    "address_length_diff",
    "address_missing",

    "country_exact"
]


TARGET = "label"


# ============================================================
# CHECK FILE
# ============================================================

print("=" * 70)
print("CHECKING TRAINING DATA")
print("=" * 70)

if not os.path.exists(TRAINING_SAMPLE):
    raise FileNotFoundError(
        f"\nTraining sample not found:\n{TRAINING_SAMPLE}\n\n"
        "Run build_training_sample.py first."
    )

print(f"Training data found:\n{TRAINING_SAMPLE}")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINING DATA")
print("=" * 70)

df = pd.read_csv(
    TRAINING_SAMPLE,
    sep="\t"
)

print(f"Rows loaded: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = (
    ["id1", "id2", TARGET]
    + FEATURE_COLUMNS
)

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns:\n"
        + "\n".join(
            f"  - {x}"
            for x in missing_columns
        )
    )

print("All required columns are present.")


# ============================================================
# BASIC DATA CHECK
# ============================================================

print()
print("=" * 70)
print("DATA DISTRIBUTION")
print("=" * 70)

positive_count = int(
    (df[TARGET] == 1).sum()
)

negative_count = int(
    (df[TARGET] == 0).sum()
)

print(f"Positive pairs: {positive_count:,}")
print(f"Negative pairs: {negative_count:,}")

if len(df) > 0:
    print(
        f"Positive rate: "
        f"{positive_count / len(df) * 100:.2f}%"
    )

if positive_count == 0:
    raise ValueError(
        "There are no positive examples in the training sample."
    )

if negative_count == 0:
    raise ValueError(
        "There are no negative examples in the training sample."
    )


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET].astype(int)

groups = df["id1"]


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print()
print("=" * 70)
print("CREATING TRAIN / VALIDATION SPLIT")
print("=" * 70)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, val_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)

X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]

y_train = y.iloc[train_idx]
y_val = y.iloc[val_idx]

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")

print(
    f"Training S1 entities: "
    f"{df.iloc[train_idx]['id1'].nunique():,}"
)

print(
    f"Validation S1 entities: "
    f"{df.iloc[val_idx]['id1'].nunique():,}"
)

print(
    f"Training positives: "
    f"{int(y_train.sum()):,}"
)

print(
    f"Validation positives: "
    f"{int(y_val.sum()):,}"
)


# ============================================================
# LIGHTGBM PARAMETERS
# ============================================================

print()
print("=" * 70)
print("LIGHTGBM TRAINING")
print("=" * 70)

model = lgb.LGBMClassifier(
    objective="binary",

    n_estimators=300,

    learning_rate=0.05,

    num_leaves=31,

    max_depth=-1,

    min_child_samples=30,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=0.1,

    random_state=42,

    n_jobs=-1,

    verbosity=-1
)


# ============================================================
# TRAIN
# ============================================================

model.fit(
    X_train,
    y_train,

    eval_set=[
        (X_val, y_val)
    ],

    callbacks=[
        lgb.early_stopping(
            stopping_rounds=30,
            verbose=True
        )
    ]
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

val_probability = model.predict_proba(
    X_val
)[:, 1]


# ============================================================
# THRESHOLD EVALUATION
# ============================================================

print()
print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

results = []

for threshold in thresholds:

    val_prediction = (
        val_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    results.append(
        (
            threshold,
            precision,
            recall,
            f1
        )
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )


# ============================================================
# SELECT PROTOTYPE THRESHOLD
# ============================================================

# For this prototype we select the threshold with
# the highest validation F1.
#
# This is NOT our final competition threshold.
# We will tune the final threshold after we have
# a much larger and more representative validation set.

best_threshold, best_precision, best_recall, best_f1 = max(
    results,
    key=lambda x: x[3]
)

print()
print("=" * 70)
print("PROTOTYPE THRESHOLD")
print("=" * 70)

print(
    f"Prototype threshold: {best_threshold:.2f}"
)

print(
    f"Precision: {best_precision:.4f}"
)

print(
    f"Recall:    {best_recall:.4f}"
)

print(
    f"F1:        {best_f1:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

best_predictions = (
    val_probability >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_val,
    best_predictions
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print("                 Predicted")
print("                 0       1")
print(
    f"Actual 0       {cm[0,0]:7d} "
    f"{cm[0,1]:7d}"
)
print(
    f"Actual 1       {cm[1,0]:7d} "
    f"{cm[1,1]:7d}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_val,
        best_predictions,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

importance = pd.DataFrame({
    "feature": FEATURE_COLUMNS,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

for _, row in importance.iterrows():

    print(
        f"{row['feature']:<30} "
        f"{int(row['importance']):>8}"
    )


# ============================================================
# SAVE VALIDATION PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("SAVING VALIDATION PREDICTIONS")
print("=" * 70)

validation_df = df.iloc[
    val_idx
][
    ["id1", "id2", "label"]
].copy()

validation_df["probability"] = val_probability

validation_df["prediction"] = (
    val_probability >= best_threshold
).astype(int)

validation_df.to_csv(
    VALIDATION_OUTPUT,
    sep="\t",
    index=False
)

print(
    f"Saved validation predictions:\n"
    f"{VALIDATION_OUTPUT}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.booster_.save_model(
    MODEL_OUTPUT
)

print(
    f"Saved LightGBM model:\n"
    f"{MODEL_OUTPUT}"
)


# ============================================================
# SAVE MODEL METADATA
# ============================================================

METADATA_OUTPUT = os.path.join(
    BASE_DIR,
    "model_metadata.txt"
)

with open(
    METADATA_OUTPUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Amazon ML Challenge - Matching Model\n"
    )

    f.write(
        "=====================================\n\n"
    )

    f.write(
        f"Training rows: {len(X_train):,}\n"
    )

    f.write(
        f"Validation rows: {len(X_val):,}\n"
    )

    f.write(
        f"Training positives: {int(y_train.sum()):,}\n"
    )

    f.write(
        f"Validation positives: {int(y_val.sum()):,}\n"
    )

    f.write(
        f"Best prototype threshold: "
        f"{best_threshold:.4f}\n"
    )

    f.write(
        f"Validation precision: "
        f"{best_precision:.6f}\n"
    )

    f.write(
        f"Validation recall: "
        f"{best_recall:.6f}\n"
    )

    f.write(
        f"Validation F1: "
        f"{best_f1:.6f}\n"
    )

    f.write(
        "\nFeatures:\n"
    )

    for feature in FEATURE_COLUMNS:
        f.write(
            f"- {feature}\n"
        )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("MODEL TRAINING COMPLETE")
print("=" * 70)

print(
    f"Model: {MODEL_OUTPUT}"
)

print(
    f"Validation predictions: "
    f"{VALIDATION_OUTPUT}"
)

print(
    f"Metadata: {METADATA_OUTPUT}"
)

print()
print("IMPORTANT:")
print(
    "The threshold selected here is only a "
    "prototype threshold."
)
print(
    "Do NOT treat it as the final competition "
    "threshold yet."
)

print("=" * 70)