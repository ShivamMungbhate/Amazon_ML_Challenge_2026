import pandas as pd
from collections import defaultdict

from preprocessing import normalize_name



STOP_TOKENS = {
    "inc",
    "llc",
    "ltd",
    "limited",
    "private",
    "pvt",
    "plc",
    "corp",
    "corporation",
    "company",
    "co",
    "llp",
}


def get_tokens(name):
    """
    Normalize a business name and return useful tokens.
    """

    normalized = normalize_name(name)

    tokens = normalized.split()

    tokens = [
        token
        for token in tokens
        if len(token) >= 3
        and token not in STOP_TOKENS
    ]

    return set(tokens)


def build_token_index(source_df):
    """
    Build:

        token -> list of Source 2 entity IDs
    """

    index = defaultdict(list)

    for _, row in source_df.iterrows():

        tokens = get_tokens(
            row["business_name"]
        )

        for token in tokens:
            index[token].append(
                row["entity_id"]
            )

    return index


def generate_candidates(
    source1_df,
    token_index
):
    """
    Generate candidates when S1 and S2
    share at least one useful name token.
    """

    candidates = set()

    for _, row in source1_df.iterrows():

        source1_id = row["entity_id"]

        tokens = get_tokens(
            row["business_name"]
        )

        for token in tokens:

            source2_ids = token_index.get(
                token,
                []
            )

            for source2_id in source2_ids:

                candidates.add(
                    (
                        source1_id,
                        source2_id
                    )
                )

    return pd.DataFrame(
        list(candidates),
        columns=[
            "source1_entity_id",
            "source2_entity_id"
        ]
    )


def calculate_blocking_recall(
    candidates,
    source1_df,
    ground_truth_path,
    source2_ids
):
    """
    Calculate blocking recall for Source 2.
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

        source1_id = row[
            "source1_entity_id"
        ]

        matched_ids = row[
            "matched_entity_ids"
        ]

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
            candidates[
                "source1_entity_id"
            ],
            candidates[
                "source2_entity_id"
            ]
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
    print("NAME TOKEN BLOCKING RECALL")
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

    source1 = source1.head(
        SAMPLE_SIZE
    )

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

    print("Building token index...")

    index = build_token_index(
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