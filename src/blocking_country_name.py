import pandas as pd

from preprocessing import normalize_name, normalize_country


def build_country_name_index(source_df):
    """
    Build an index using:

        (country, normalized_name)
            ->
        list of Source 2 entity IDs
    """

    index = {}

    for _, row in source_df.iterrows():

        country = normalize_country(
            row["country"]
        )

        name = normalize_name(
            row["business_name"]
        )

        if not country or not name:
            continue

        key = (country, name)

        entity_id = row["entity_id"]

        if key not in index:
            index[key] = []

        index[key].append(entity_id)

    return index


def generate_candidates(
    source1_df,
    source2_index
):
    """
    Generate candidates using exact normalized
    country + normalized business name.
    """

    results = []

    for _, row in source1_df.iterrows():

        s1_id = row["entity_id"]

        country = normalize_country(
            row["country"]
        )

        name = normalize_name(
            row["business_name"]
        )

        key = (country, name)

        candidates = source2_index.get(
            key,
            []
        )

        for s2_id in candidates:

            results.append(
                {
                    "source1_entity_id": s1_id,
                    "source2_entity_id": s2_id
                }
            )

    return pd.DataFrame(results)


def calculate_blocking_recall(
    candidates,
    source1_df,
    ground_truth_path,
    source2_ids
):
    """
    Calculate recall of the blocking strategy
    against the Source 2 portion of ground truth.
    """

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
        ground_truth_path,
        sep="\t"
    )

    sample_s1_ids = set(
        source1_df["entity_id"]
    )

    ground_truth = ground_truth[
        ground_truth["source1_entity_id"].isin(
            sample_s1_ids
        )
    ].copy()

    ground_truth["matched_entity_ids"] = (
        ground_truth["matched_entity_ids"]
        .fillna("")
        .apply(
            lambda x: {
                item.strip()
                for item in str(x).split(",")
                if item.strip()
            }
        )
    )


    truth_lookup = {}

    for _, row in ground_truth.iterrows():

        true_matches = row[
            "matched_entity_ids"
        ]

        true_s2_matches = (
            true_matches.intersection(
                source2_ids
            )
        )

        truth_lookup[
            row["source1_entity_id"]
        ] = true_s2_matches

    candidate_lookup = {}

    for _, row in candidates.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        s2_id = row[
            "source2_entity_id"
        ]

        if s1_id not in candidate_lookup:
            candidate_lookup[s1_id] = set()

        candidate_lookup[s1_id].add(
            s2_id
        )

    total_true = 0
    captured_true = 0

    for s1_id, true_matches in truth_lookup.items():

        predicted = candidate_lookup.get(
            s1_id,
            set()
        )

        total_true += len(true_matches)

        captured_true += len(
            true_matches.intersection(
                predicted
            )
        )

    recall = (
        captured_true / total_true
        if total_true > 0
        else 0
    )

    print("\n" + "=" * 50)
    print("COUNTRY + NAME BLOCKING RECALL")
    print("=" * 50)

    print(
        f"True S2 matches considered : "
        f"{total_true:,}"
    )

    print(
        f"True S2 matches captured   : "
        f"{captured_true:,}"
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
        nrows=SAMPLE_SIZE
    )

    print("Loading Source 2...")

    source2 = pd.read_csv(
        source2_path,
        sep="\t"
    )

    source2_ids = set(
        source2["entity_id"]
    )

    print("Building country + name index...")

    index = build_country_name_index(
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

    print(
        f"S1 covered      : "
        f"{candidates['source1_entity_id'].nunique():,}"
    )

    if not candidates.empty:

        avg = (
            len(candidates)
            / candidates[
                "source1_entity_id"
            ].nunique()
        )

        print(
            f"Avg candidates  : "
            f"{avg:.2f}"
        )

    print("=" * 50)

    calculate_blocking_recall(
        candidates,
        source1,
        ground_truth_path,
        source2_ids
    )