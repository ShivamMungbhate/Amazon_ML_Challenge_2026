import os
import pandas as pd

from matching_features import make_pair_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\shiva\OneDrive\Desktop\Amazon_ML_Challenge_2026"

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "6ab10eb3b23ba_student_resource",
    "student_resource",
    "dataset",
    "train"
)

SOURCE1 = os.path.join(TRAIN_DIR, "train_source1.tsv")
SOURCE2 = os.path.join(TRAIN_DIR, "train_source2.tsv")
SOURCE3 = os.path.join(TRAIN_DIR, "train_source3.tsv")

GROUND_TRUTH = os.path.join(
    TRAIN_DIR,
    "train_ground_truth.tsv"
)

# ------------------------------------------------------------
# Candidate file
#
# IMPORTANT:
# Change this path if Teammate 3's candidate file is copied
# somewhere else on your computer.
# ------------------------------------------------------------

CANDIDATE_FILE = os.path.join(
    BASE_DIR,
    "train_candidate_pairs_token.tsv"
)

# Output
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "training_sample_10k.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

N_CANDIDATES = 10_000

CHUNK_SIZE = 100_000


# ============================================================
# HELPER
# ============================================================

def check_file(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n{name} not found:\n{path}\n"
        )


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

check_file(SOURCE1, "Source 1")
check_file(SOURCE2, "Source 2")
check_file(SOURCE3, "Source 3")
check_file(GROUND_TRUTH, "Ground truth")
check_file(CANDIDATE_FILE, "Candidate file")

print("All required files found.")


# ============================================================
# STEP 1: READ CANDIDATE SAMPLE
# ============================================================

print()
print("=" * 70)
print("STEP 1: READING CANDIDATE SAMPLE")
print("=" * 70)

candidates = pd.read_csv(
    CANDIDATE_FILE,
    sep="\t",
    nrows=N_CANDIDATES,
    dtype=str
)

# Make sure expected columns exist
required_columns = {"id1", "id2"}

if not required_columns.issubset(candidates.columns):
    raise ValueError(
        f"Candidate file must contain columns {required_columns}. "
        f"Found: {list(candidates.columns)}"
    )

candidates = candidates[["id1", "id2"]].drop_duplicates()

print(
    f"Candidate pairs loaded: {len(candidates):,}"
)


# ============================================================
# STEP 2: DETERMINE REQUIRED ENTITIES
# ============================================================

print()
print("=" * 70)
print("STEP 2: IDENTIFYING REQUIRED ENTITIES")
print("=" * 70)

s1_ids = set(candidates["id1"].dropna())

s2_ids = set(
    candidates.loc[
        candidates["id2"].str.startswith("S2-", na=False),
        "id2"
    ]
    .dropna()
)

s3_ids = set(
    candidates.loc[
        candidates["id2"].str.startswith("S3-", na=False),
        "id2"
    ]
    .dropna()
)

print(f"S1 entities required: {len(s1_ids):,}")
print(f"S2 entities required: {len(s2_ids):,}")
print(f"S3 entities required: {len(s3_ids):,}")


# ============================================================
# STEP 3: LOAD REQUIRED SOURCE ENTITIES
# ============================================================

def load_required_entities(
    filepath,
    required_ids,
    source_name
):

    print()
    print(f"Loading required entities from {source_name}...")
    print(f"Required IDs: {len(required_ids):,}")

    if not required_ids:
        return pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "address",
                "country"
            ]
        )

    pieces = []

    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False
    ):

        matches = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        if len(matches) > 0:
            pieces.append(
                matches[
                    [
                        "entity_id",
                        "business_name",
                        "address",
                        "country"
                    ]
                ].copy()
            )

        # Stop once we have found everything.
        found = sum(len(x) for x in pieces)

        if found >= len(required_ids):
            break

    if pieces:
        result = pd.concat(
            pieces,
            ignore_index=True
        ).drop_duplicates(
            subset=["entity_id"]
        )
    else:
        result = pd.DataFrame(
            columns=[
                "entity_id",
                "business_name",
                "address",
                "country"
            ]
        )

    print(
        f"Loaded {len(result):,} entities from {source_name}"
    )

    return result


source1 = load_required_entities(
    SOURCE1,
    s1_ids,
    "source1"
)

source2 = load_required_entities(
    SOURCE2,
    s2_ids,
    "source2"
)

source3 = load_required_entities(
    SOURCE3,
    s3_ids,
    "source3"
)


# ============================================================
# STEP 4: CREATE LOOKUP DICTIONARIES
# ============================================================

print()
print("=" * 70)
print("STEP 4: CREATING ENTITY LOOKUPS")
print("=" * 70)

source1_lookup = {
    row["entity_id"]: row
    for _, row in source1.iterrows()
}

source2_lookup = {
    row["entity_id"]: row
    for _, row in source2.iterrows()
}

source3_lookup = {
    row["entity_id"]: row
    for _, row in source3.iterrows()
}

print("Lookup dictionaries created.")


# ============================================================
# STEP 5: LOAD GROUND TRUTH
# ============================================================

print()
print("=" * 70)
print("STEP 5: LOADING GROUND TRUTH")
print("=" * 70)

# We only need ground truth for S1 entities appearing
# in our candidate sample.

true_pairs = set()

ground_truth_found = 0

for chunk in pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    chunksize=CHUNK_SIZE,
    dtype=str,
    keep_default_na=False
):

    relevant = chunk[
        chunk["source1_entity_id"].isin(s1_ids)
    ]

    for _, row in relevant.iterrows():

        id1 = row["source1_entity_id"]

        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for id2 in matched_ids.split(","):

            id2 = id2.strip()

            if id2:
                true_pairs.add(
                    (id1, id2)
                )

        ground_truth_found += 1

print(
    f"S1 ground-truth rows found: "
    f"{ground_truth_found:,}"
)

print(
    f"True pairs for sampled S1 entities: "
    f"{len(true_pairs):,}"
)


# ============================================================
# STEP 6: CREATE LABELS
# ============================================================

print()
print("=" * 70)
print("STEP 6: CREATING LABELS")
print("=" * 70)

candidates["label"] = [
    1 if (id1, id2) in true_pairs else 0
    for id1, id2
    in zip(
        candidates["id1"],
        candidates["id2"]
    )
]

positive_count = int(
    candidates["label"].sum()
)

negative_count = (
    len(candidates) - positive_count
)

print(f"Positive pairs: {positive_count:,}")
print(f"Negative pairs: {negative_count:,}")

if len(candidates) > 0:

    positive_pct = (
        positive_count / len(candidates) * 100
    )

    print(
        f"Positive percentage: "
        f"{positive_pct:.2f}%"
    )


# ============================================================
# STEP 7: GENERATE FEATURES
# ============================================================

print()
print("=" * 70)
print("STEP 7: GENERATING MATCHING FEATURES")
print("=" * 70)

feature_rows = []

missing_source1 = 0
missing_source2 = 0
missing_source3 = 0


for i, row in enumerate(
    candidates.itertuples(index=False),
    start=1
):

    id1 = row.id1
    id2 = row.id2

    entity1 = source1_lookup.get(id1)

    if entity1 is None:
        missing_source1 += 1
        continue

    if id2.startswith("S2-"):

        entity2 = source2_lookup.get(id2)

        if entity2 is None:
            missing_source2 += 1
            continue

    elif id2.startswith("S3-"):

        entity2 = source3_lookup.get(id2)

        if entity2 is None:
            missing_source3 += 1
            continue

    else:
        continue

    features = make_pair_features(
        entity1,
        entity2
    )

    features["id1"] = id1
    features["id2"] = id2
    features["label"] = row.label

    feature_rows.append(features)

    if i % 1000 == 0:
        print(
            f"Processed {i:,}/{len(candidates):,} "
            f"candidate pairs..."
        )


# ============================================================
# STEP 8: BUILD DATAFRAME
# ============================================================

print()
print("=" * 70)
print("STEP 8: BUILDING TRAINING DATASET")
print("=" * 70)

training_df = pd.DataFrame(feature_rows)

print(
    f"Final feature rows: "
    f"{len(training_df):,}"
)

print(
    f"Missing S1 entities: "
    f"{missing_source1:,}"
)

print(
    f"Missing S2 entities: "
    f"{missing_source2:,}"
)

print(
    f"Missing S3 entities: "
    f"{missing_source3:,}"
)


# ============================================================
# STEP 9: REORDER COLUMNS
# ============================================================

feature_columns = [
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

final_columns = [
    "id1",
    "id2"
] + feature_columns + [
    "label"
]

training_df = training_df[
    final_columns
]


# ============================================================
# STEP 10: SAVE
# ============================================================

print()
print("=" * 70)
print("STEP 10: SAVING TRAINING SAMPLE")
print("=" * 70)

training_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

print(
    f"Saved training sample to:\n{OUTPUT_FILE}"
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING SAMPLE COMPLETE")
print("=" * 70)

print(
    f"Rows                    : "
    f"{len(training_df):,}"
)

print(
    f"Positive labels         : "
    f"{int(training_df['label'].sum()):,}"
)

print(
    f"Negative labels         : "
    f"{int((training_df['label'] == 0).sum()):,}"
)

print(
    f"Number of features      : "
    f"{len(feature_columns)}"
)

print()
print("Features:")

for feature in feature_columns:
    print(f"  - {feature}")

print()
print("Output:")
print(OUTPUT_FILE)

print("=" * 70)