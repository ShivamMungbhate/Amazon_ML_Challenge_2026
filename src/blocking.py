import pandas as pd

from preprocessing import normalize_name


def build_name_index(source_df):
    """
    Build an index:

        normalized_name -> list of Source 2 entity IDs

    Example:

        "brightic data" -> ["S2_123", "S2_456"]
    """

    index = {}

    for _, row in source_df.iterrows():

        name = normalize_name(row["business_name"])

        if not name:
            continue

        entity_id = row["entity_id"]

        if name not in index:
            index[name] = []

        index[name].append(entity_id)

    return index


def generate_candidates(source1_df, source2_name_index):
    """
    Generate Source 2 candidates for Source 1 records
    using exact normalized business-name matching.
    """

    results = []

    for _, row in source1_df.iterrows():

        s1_id = row["entity_id"]

        name = normalize_name(row["business_name"])

        candidates = source2_name_index.get(name, [])

        for s2_id in candidates:

            results.append(
                {
                    "source1_entity_id": s1_id,
                    "source2_entity_id": s2_id,
                }
            )

    return pd.DataFrame(results)


def print_blocking_statistics(candidates):
    """
    Print basic statistics about generated candidates.
    """

    print("\n" + "=" * 50)
    print("BLOCKING STATISTICS")
    print("=" * 50)

    if candidates.empty:
        print("No candidates generated.")
        return

    total_pairs = len(candidates)

    unique_s1 = candidates["source1_entity_id"].nunique()

    avg_candidates = (
        total_pairs / unique_s1
        if unique_s1 > 0
        else 0
    )

    print(f"Candidate pairs      : {total_pairs:,}")
    print(f"S1 entities covered  : {unique_s1:,}")
    print(f"Avg candidates / S1  : {avg_candidates:.2f}")

    print("=" * 50)


def calculate_blocking_recall(
    candidates,
    ground_truth_path,
    source2_ids
):
    """
    Measure how many TRUE Source 2 matches were captured
    by the blocking strategy.

    Important:
    Ground truth contains both Source 2 and Source 3 IDs.

    Therefore we first identify which IDs belong to Source 2,
    then evaluate only the Source 2 matches.
    """

    print("\nLoading ground truth...")

    ground_truth = pd.read_csv(
    ground_truth_path,
    sep="\t"
)

    sample_s1_ids = set(
        candidates["source1_entity_id"]
    )

    ground_truth = ground_truth[
        ground_truth["source1_entity_id"].isin(sample_s1_ids)
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

        s1_id = row["source1_entity_id"]

        true_matches = row["matched_entity_ids"]

        true_s2_matches = (
            true_matches.intersection(source2_ids)
        )

        truth_lookup[s1_id] = true_s2_matches

    
    candidate_lookup = {}

    for _, row in candidates.iterrows():

        s1_id = row["source1_entity_id"]
        s2_id = row["source2_entity_id"]

        if s1_id not in candidate_lookup:
            candidate_lookup[s1_id] = set()

        candidate_lookup[s1_id].add(s2_id)

    
    total_true_s2 = 0
    captured_true_s2 = 0

    for s1_id, true_s2_matches in truth_lookup.items():

        predicted_candidates = candidate_lookup.get(
            s1_id,
            set()
        )

        total_true_s2 += len(true_s2_matches)

        captured_true_s2 += len(
            true_s2_matches.intersection(
                predicted_candidates
            )
        )

    recall = (
        captured_true_s2 / total_true_s2
        if total_true_s2 > 0
        else 0
    )

    print("\n" + "=" * 50)
    print("BLOCKING RECALL")
    print("=" * 50)

    print(
        f"True S2 matches considered : "
        f"{total_true_s2:,}"
    )

    print(
        f"True S2 matches captured   : "
        f"{captured_true_s2:,}"
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

    print("Building name index...")

    name_index = build_name_index(source2)


    print("Generating candidates...")

    candidates = generate_candidates(
        source1,
        name_index
    )

   
    print_blocking_statistics(
        candidates
    )

    
    calculate_blocking_recall(
        candidates,
        ground_truth_path,
        source2_ids
    )