import pandas as pd

PATH = (
    "6ab10eb3b23ba_student_resource/"
    "student_resource/dataset/train/train_ground_truth.tsv"
)

total = 0
zero_matches = 0

match_count_distribution = {}

s2_matches = 0
s3_matches = 0
both_matches = 0

for chunk in pd.read_csv(
    PATH,
    sep="\t",
    chunksize=100_000
):

    for matched in chunk["matched_entity_ids"].fillna(""):

        matched = str(matched).strip()

        
        if not matched:
            zero_matches += 1
            count = 0

        else:
            ids = matched.split(",")

            count = len(ids)

            has_s2 = any(x.startswith("S2-") for x in ids)
            has_s3 = any(x.startswith("S3-") for x in ids)

            if has_s2:
                s2_matches += 1

            if has_s3:
                s3_matches += 1

            if has_s2 and has_s3:
                both_matches += 1

        match_count_distribution[count] = (
            match_count_distribution.get(count, 0) + 1
        )

        total += 1


print("\n" + "=" * 60)
print("GROUND TRUTH ANALYSIS")
print("=" * 60)

print("Total Source 1 entities:", total)

print("No-match / singleton entities:", zero_matches)

print(
    "Percentage with no match:",
    round(zero_matches / total * 100, 2),
    "%"
)

print("\nS1 entities with S2 matches:", s2_matches)
print("S1 entities with S3 matches:", s3_matches)
print("S1 entities with BOTH S2 and S3 matches:", both_matches)

print("\nMatch count distribution:")
print("-" * 40)

for count in sorted(match_count_distribution):
    number = match_count_distribution[count]
    percentage = number / total * 100

    print(
        f"{count:>3} matches : "
        f"{number:>10} entities "
        f"({percentage:>6.2f}%)"
    )