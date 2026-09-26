import csv
import os
from collections import defaultdict


GROUND_TRUTH = (
    r"C:\Users\shiva\OneDrive\Desktop\Amazon_ML_Challenge_2026"
    r"\6ab10eb3b23ba_student_resource\student_resource"
    r"\dataset\train\train_ground_truth.tsv"
)

OUTPUT = r"C:\Users\shiva\OneDrive\Desktop\Amazon_ML_Challenge_2026"
OUTPUT = os.path.join(OUTPUT, "ground_truth_pairs_sample.tsv")

MAX_ROWS = 100_000


def load_ground_truth(path, max_rows=None):
    """
    Expand:

    source1_entity_id    matched_entity_ids

    into:

    id1    id2
    """

    pairs = set()

    with open(path, "r", encoding="utf-8", newline="") as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row_num, row in enumerate(reader):

            id1 = row["source1_entity_id"].strip()
            matched_ids = row["matched_entity_ids"]

            if not id1 or not matched_ids:
                continue

            for id2 in matched_ids.split(","):

                id2 = id2.strip()

                if id2:
                    pairs.add((id1, id2))

            if max_rows and row_num + 1 >= max_rows:
                break

    return pairs


print("=" * 60)
print("LOADING GROUND TRUTH SAMPLE")
print("=" * 60)

pairs = load_ground_truth(
    GROUND_TRUTH,
    max_rows=MAX_ROWS
)

print(f"Unique true pairs loaded: {len(pairs):,}")


with open(
    OUTPUT,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    writer = csv.writer(f, delimiter="\t")

    writer.writerow(["id1", "id2", "label"])

    for id1, id2 in pairs:
        writer.writerow([id1, id2, 1])


print()
print(f"Saved: {OUTPUT}")
print("=" * 60)