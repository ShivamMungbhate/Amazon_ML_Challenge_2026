import pandas as pd
from collections import defaultdict, Counter



SAMPLE_SIZE = 10_000


MAX_TOKEN_FREQUENCY = 0.01


MAX_CANDIDATES_PER_ENTITY = 500




S1_PATH = (
    "6ab10eb3b23ba_student_resource/"
    "student_resource/dataset/train/train_source1.tsv"
)

S2_PATH = (
    "6ab10eb3b23ba_student_resource/"
    "student_resource/dataset/train/train_source2.tsv"
)

GT_PATH = (
    "6ab10eb3b23ba_student_resource/"
    "student_resource/dataset/train/train_ground_truth.tsv"
)




def normalize_text(text):
    """
    Basic normalization for names.

    Keeps Unicode characters so multilingual names
    such as Hindi are not destroyed.
    """

    if pd.isna(text):
        return ""

    text = str(text).casefold()


    text = "".join(
        char if char.isalnum() else " "
        for char in text
    )


    text = " ".join(text.split())

    return text




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
    Convert a normalized business name into useful tokens.

    Very short tokens and common legal suffixes are ignored.
    """

    normalized = normalize_text(name)

    tokens = normalized.split()

    useful_tokens = {
        token
        for token in tokens
        if len(token) >= 3
        and token not in STOP_TOKENS
    }

    return useful_tokens




def build_token_frequency(source2):
    """
    Count how many Source 2 entities contain each token.

    We count entities, not total token occurrences.
    """

    token_frequency = Counter()

    for name in source2["business_name"]:
        tokens = get_tokens(name)

     
        for token in tokens:
            token_frequency[token] += 1

    return token_frequency




def build_rare_token_index(source2, token_frequency):
    """
    Build:

        token -> list of Source 2 entity IDs

    Only sufficiently rare tokens are indexed.
    """

    total_entities = len(source2)

    max_frequency = total_entities * MAX_TOKEN_FREQUENCY

    token_index = defaultdict(list)

    for row in source2.itertuples(index=False):

        entity_id = row.entity_id
        tokens = get_tokens(row.business_name)

        for token in tokens:

            if token_frequency[token] <= max_frequency:
                token_index[token].append(entity_id)

    return token_index




def generate_candidates(source1_row, token_index, token_frequency):
    """
    Generate Source 2 candidates for one Source 1 entity.

    Candidates come from rare/informative name tokens.
    """

    tokens = get_tokens(source1_row.business_name)

    candidate_scores = Counter()

    for token in tokens:

        if token not in token_index:
            continue

        for candidate_id in token_index[token]:
            candidate_scores[candidate_id] += 1


    ranked_candidates = sorted(
        candidate_scores.items(),
        key=lambda x: (-x[1], x[0])
    )

   
    ranked_candidates = ranked_candidates[
        :MAX_CANDIDATES_PER_ENTITY
    ]

    return [candidate_id for candidate_id, _ in ranked_candidates]




def load_ground_truth(path):

    gt = pd.read_csv(path, sep="\t")

    ground_truth = {}

    for row in gt.itertuples(index=False):

        matched_ids = set()

        if pd.notna(row.matched_entity_ids):

            matched_ids = {
                x.strip()
                for x in str(row.matched_entity_ids).split(",")
                if x.strip()
            }

        ground_truth[row.source1_entity_id] = matched_ids

    return ground_truth




def evaluate_blocking(source1, token_index, token_frequency, ground_truth):

    total_candidates = 0
    entities_with_candidates = 0

    true_matches_considered = 0
    true_matches_captured = 0

    for row in source1.itertuples(index=False):

        candidates = generate_candidates(
            row,
            token_index,
            token_frequency
        )

        total_candidates += len(candidates)

        if candidates:
            entities_with_candidates += 1

        candidate_set = set(candidates)

        true_matches = ground_truth.get(
            row.entity_id,
            set()
        )

        true_matches_considered += len(true_matches)

        true_matches_captured += len(
            candidate_set.intersection(true_matches)
        )

    recall = (
        true_matches_captured / true_matches_considered
        if true_matches_considered > 0
        else 0
    )

    average_candidates = (
        total_candidates / entities_with_candidates
        if entities_with_candidates > 0
        else 0
    )

    print("\n========== RARE TOKEN BLOCKING RESULTS ==========")

    print(
        f"S1 entities evaluated      : {len(source1):,}"
    )

    print(
        f"Candidate pairs             : {total_candidates:,}"
    )

    print(
        f"S1 entities with candidates : "
        f"{entities_with_candidates:,}"
    )

    print(
        f"Avg candidates / covered S1 : "
        f"{average_candidates:.2f}"
    )

    print(
        f"True S2 matches considered  : "
        f"{true_matches_considered:,}"
    )

    print(
        f"True S2 matches captured    : "
        f"{true_matches_captured:,}"
    )

    print(
        f"Blocking recall             : "
        f"{recall:.4f}"
    )

    print("===============================================\n")



def main():

    print("Loading Source 1 sample...")

    source1 = pd.read_csv(
        S1_PATH,
        sep="\t",
        nrows=SAMPLE_SIZE
    )

    print(
        f"Loaded Source 1 rows: {len(source1):,}"
    )

    print("\nLoading Source 2...")

    source2 = pd.read_csv(
        S2_PATH,
        sep="\t"
    )

    print(
        f"Loaded Source 2 rows: {len(source2):,}"
    )

    print("\nBuilding ground truth...")

    ground_truth = load_ground_truth(GT_PATH)

    print(
        f"Ground truth entities: {len(ground_truth):,}"
    )

    print("\nCalculating token frequencies...")

    token_frequency = build_token_frequency(source2)

    print(
        f"Unique tokens found: {len(token_frequency):,}"
    )

    print("\nBuilding rare-token index...")

    token_index = build_rare_token_index(
        source2,
        token_frequency
    )

    print(
        f"Rare tokens indexed: {len(token_index):,}"
    )

    print("\nEvaluating blocking...")

    evaluate_blocking(
        source1,
        token_index,
        token_frequency,
        ground_truth
    )


if __name__ == "__main__":
    main()