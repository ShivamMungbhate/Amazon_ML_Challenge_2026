import pandas as pd

from preprocessing import normalize_address


def build_address_index(source_df):
    """
    Build an index:
        normalized_address -> list of Source 2 entity IDs
    """

    index = {}

    for _, row in source_df.iterrows():

        address = normalize_address(
            row["business_address"]
        )

        if not address:
            continue

        entity_id = row["entity_id"]

        if address not in index:
            index[address] = []

        index[address].append(entity_id)

    return index


def generate_candidates(source1_df, source2_address_index):
    """
    Generate S1-S2 candidate pairs using
    exact normalized-address blocking.
    """

    candidates = []

    for _, row in source1_df.iterrows():

        address = normalize_address(
            row["business_address"]
        )

        if not address:
            continue

        source2_ids = source2_address_index.get(
            address,
            []
        )

        for source2_id in source2_ids:

            candidates.append(
                (
                    row["entity_id"],
                    source2_id
                )
            )

    return pd.DataFrame(
        candidates,
        columns=[
            "source1_entity_id",
            "source2_entity_id"
        ]
    ).drop_duplicates()


def calculate_blocking_recall(
    candidates,
    source1_df,
    ground_truth_path,
    source2_ids
):
    """
    Calculate recall of the blocking strategy.

    Only Source 2 matches are considered.
    """

    sample_ids = set(
        source1_df["entity_id"]
    )

    ground_truth = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    ground_truth = ground_truth[
        ground_truth["source1_entity_id"].isin(
            sample_ids
        )
    ]

    true_pairs = set()

    for _, row in ground_truth.iterrows():

        source1_id = row["source1_entity_id"]

        matched_ids = row["matched_entity_ids"]

        if not matched_ids:
            continue

        for source2_id in matched_ids.split(","):

            source2_id = source2_id.strip()

            if source2_id in source2_ids:

                true_pairs.add(
                    (
                        source1_id,
                        source2_id
                    )
                )

    candidate_pairs = set(
        zip(
            candidates["source1_entity_id"],
            candidates["source2_entity_id"]
        )
    )

    captured = len(
        true_pairs & candidate_pairs
    )

    total = len(true_pairs)

    recall = (
        captured / total
        if total > 0
        else 0
    )

    print("\n" + "=" * 50)
    print("ADDRESS BLOCKING RECALL")
    print("=" * 50)

    print(
        f"True S2 matches considered : "
        f"{total:,}"
    )

    print(
        f"True S2 matches captured   : "
        f"{captured:,}"
    )

    print(
        f"Blocking recall            : "
        f"{recall:.4f}"
    )

    print("=" * 50)


if __name__ == "__main__":

    SAMPLE_SIZE = 10_000

    source1_path = (
        "6ab10eb3b23ba_student_resource/"
        "student_resource/dataset/train/"
        "train_source1.tsv"
    )

    source2_path = (
        "6ab10eb3b23ba_student_resource/"
        "student_resource/dataset/train/"
        "train_source2.tsv"
    )

    ground_truth_path = (
        "6ab10eb3b23ba_student_resource/"
        "student_resource/dataset/train/"
        "train_ground_truth.tsv"
    )

    print("Loading Source 1 sample...")

    source1 = pd.read_csv(
        source1_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    source1 = source1.head(SAMPLE_SIZE)

    print("Loading Source 2...")

    source2 = pd.read_csv(
        source2_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    source2_ids = set(
        source2["entity_id"]
    )

    print("Building address index...")

    index = build_address_index(
        source2
    )

    print("Generating candidates...")

    candidates = generate_candidates(
        source1,
        index
    )

    print("\n" + "=" * 50)
    print("BLOCKING STATISTICS")
    print("=" * 50)

    print(
        f"Candidate pairs : "
        f"{len(candidates):,}"
    )

    covered = (
        candidates[
            "source1_entity_id"
        ].nunique()
        if not candidates.empty
        else 0
    )

    print(
        f"S1 covered      : "
        f"{covered:,}"
    )

    if covered > 0:

        avg = (
            len(candidates) / covered
        )

        print(
            f"Avg candidates  : "
            f"{avg:.2f}"
        )

    print("=" * 50)

    print("Loading ground truth...")

    calculate_blocking_recall(
        candidates,
        source1,
        ground_truth_path,
        source2_ids
    )