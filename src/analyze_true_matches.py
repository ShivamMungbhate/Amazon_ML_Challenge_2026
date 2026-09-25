import os
import random
import pandas as pd
import re



BASE = "6ab10eb3b23ba_student_resource/student_resource/dataset"

GROUND_TRUTH = os.path.join(
    BASE,
    "train/train_ground_truth.tsv"
)

SOURCE1 = os.path.join(
    BASE,
    "train/train_source1.tsv"
)

SOURCE2 = os.path.join(
    BASE,
    "train/train_source2.tsv"
)

SOURCE3 = os.path.join(
    BASE,
    "train/train_source3.tsv"
)


NUM_PAIRS = 100


random.seed(42)


print("=" * 70)
print("STEP 1: Reading ground truth")
print("=" * 70)

sample_pairs = []

for chunk in pd.read_csv(
    GROUND_TRUTH,
    sep="\t",
    chunksize=100_000
):


    chunk = chunk[
        chunk["matched_entity_ids"].notna()
        & (chunk["matched_entity_ids"].astype(str).str.strip() != "")
    ]

    for _, row in chunk.iterrows():

        s1_id = row["source1_entity_id"]

        matched_ids = str(row["matched_entity_ids"]).split(",")

        for match_id in matched_ids:

            match_id = match_id.strip()

            if match_id:
                sample_pairs.append(
                    (s1_id, match_id)
                )


print("Total real matching pairs found:", len(sample_pairs))


sample_pairs = random.sample(
    sample_pairs,
    min(NUM_PAIRS, len(sample_pairs))
)

print("Selected pairs:", len(sample_pairs))




s1_ids = set()
s2_ids = set()
s3_ids = set()

for s1_id, match_id in sample_pairs:

    s1_ids.add(s1_id)

    if match_id.startswith("S2-"):
        s2_ids.add(match_id)

    elif match_id.startswith("S3-"):
        s3_ids.add(match_id)


print("\nS1 IDs needed:", len(s1_ids))
print("S2 IDs needed:", len(s2_ids))
print("S3 IDs needed:", len(s3_ids))




def find_records(path, wanted_ids):

    results = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000
    ):

        matches = chunk[
            chunk["entity_id"].isin(wanted_ids)
        ]

        for _, row in matches.iterrows():

            results[row["entity_id"]] = row.to_dict()


        if len(results) == len(wanted_ids):
            break

    return results




print("\n" + "=" * 70)
print("STEP 2: Finding actual records")
print("=" * 70)

s1_records = find_records(
    SOURCE1,
    s1_ids
)

s2_records = find_records(
    SOURCE2,
    s2_ids
)

s3_records = find_records(
    SOURCE3,
    s3_ids
)

print("Found S1 records:", len(s1_records))
print("Found S2 records:", len(s2_records))
print("Found S3 records:", len(s3_records))




def normalize_text(value):

    if pd.isna(value):
        return ""

    value = str(value).lower()

  
    value = re.sub(r"\s+", " ", value)


    value = value.strip()

    return value



print("\n" + "=" * 70)
print("REAL MATCH EXAMPLES")
print("=" * 70)

for number, (s1_id, match_id) in enumerate(
    sample_pairs,
    start=1
):

    s1 = s1_records.get(s1_id)

    if match_id.startswith("S2-"):
        match = s2_records.get(match_id)
    else:
        match = s3_records.get(match_id)

    if s1 is None or match is None:
        continue

    s1_name = s1["business_name"]
    s1_address = s1["business_address"]
    s1_country = s1["country"]

    match_name = match["business_name"]
    match_address = match["business_address"]
    match_country = match["country"]

    print("\n" + "-" * 70)

    print(f"MATCH #{number}")
    print(f"{s1_id}  <-->  {match_id}")

    print("\nBusiness name:")
    print("S1 :", s1_name)
    print("M  :", match_name)

    print("\nAddress:")
    print("S1 :", s1_address)
    print("M  :", match_address)

    print("\nCountry:")
    print("S1 :", s1_country)
    print("M  :", match_country)

    print("\nNormalized names:")
    print("S1 :", normalize_text(s1_name))
    print("M  :", normalize_text(match_name))