import pandas as pd


def load_ground_truth(path):
    """
    Load the training ground truth.

    matched_entity_ids is stored as a comma-separated string,
    so we convert it into a Python set for easy comparison.
    """

    df = pd.read_csv(path, sep="\t")

    df["matched_entity_ids"] = (
        df["matched_entity_ids"]
        .fillna("")
        .apply(
            lambda x: {
                item.strip()
                for item in str(x).split(",")
                if item.strip()
            }
        )
    )

    return df


def calculate_metrics(ground_truth, predictions):
    """
    Calculate precision, recall and F1 for entity matching.

    ground_truth:
        Dictionary:
            source1_entity_id -> set of true matched IDs

    predictions:
        Dictionary:
            source1_entity_id -> set of predicted matched IDs
    """

    total_true = 0
    total_predicted = 0
    total_correct = 0

    for entity_id, true_matches in ground_truth.items():

        predicted_matches = predictions.get(entity_id, set())

        total_true += len(true_matches)
        total_predicted += len(predicted_matches)

        total_correct += len(
            true_matches.intersection(predicted_matches)
        )

    precision = (
        total_correct / total_predicted
        if total_predicted > 0
        else 0.0
    )

    recall = (
        total_correct / total_true
        if total_true > 0
        else 0.0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_matches": total_correct,
        "predicted_matches": total_predicted,
        "true_matches": total_true,
    }


def print_metrics(metrics):
    """
    Print evaluation results in a readable format.
    """

    print("\n" + "=" * 50)
    print("MATCHING EVALUATION")
    print("=" * 50)

    print(f"Precision       : {metrics['precision']:.4f}")
    print(f"Recall          : {metrics['recall']:.4f}")
    print(f"F1 Score        : {metrics['f1']:.4f}")

    print("-" * 50)

    print(f"Correct matches : {metrics['correct_matches']:,}")
    print(f"Predicted       : {metrics['predicted_matches']:,}")
    print(f"True matches    : {metrics['true_matches']:,}")

    print("=" * 50)