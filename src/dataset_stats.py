import os
import pandas as pd

BASE = "6ab10eb3b23ba_student_resource/student_resource/dataset"

files = [
    "train/train_source1.tsv",
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "train/train_ground_truth.tsv",
    "test/test_source1.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]

for relative_path in files:

    path = os.path.join(BASE, relative_path)

    print("\n" + "=" * 70)
    print(relative_path)

    total_rows = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        chunksize=100_000
    ):
        total_rows += len(chunk)

    print("Total rows:", total_rows)