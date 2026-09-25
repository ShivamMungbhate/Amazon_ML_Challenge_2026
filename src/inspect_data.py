import os
import pandas as pd

DATASET_DIR = "6ab10eb3b23ba_student_resource/student_resource/dataset"

for root, dirs, files in os.walk(DATASET_DIR):

    for file in files:

        path = os.path.join(root, file)

        size_gb = os.path.getsize(path) / (1024 ** 3)

        print("\n" + "=" * 60)
        print("FILE:", path)
        print(f"SIZE: {size_gb:.2f} GB")

        if file.endswith(".tsv"):

            try:
                sample = pd.read_csv(
                    path,
                    sep="\t",
                    nrows=5
                )

                print("\nCOLUMNS:")
                print(list(sample.columns))

                print("\nFIRST 5 ROWS:")
                print(sample.to_string(index=False))

            except Exception as e:
                print("\nERROR:", e)