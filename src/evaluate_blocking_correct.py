import duckdb
import os



def main():

    print("=" * 60)
    print("CORRECT BLOCKING EVALUATION")
    print("=" * 60)


    # ENTER LOCAL FILE PATHS
    

    ground_truth = input(
        "\nEnter full path to train_ground_truth.tsv: "
    ).strip().strip('"')

    candidates = input(
        "Enter full path to train_candidate_pairs.tsv: "
    ).strip().strip('"')

    db_file = "blocking_evaluation.duckdb"

   
    if not os.path.exists(ground_truth):
        raise FileNotFoundError(
            f"\nGround truth file not found:\n{ground_truth}"
        )

    if not os.path.exists(candidates):
        raise FileNotFoundError(
            f"\nCandidate file not found:\n{candidates}"
        )

    print("\nGround truth:")
    print(ground_truth)

    print("\nCandidate pairs:")
    print(candidates)

    

    con = duckdb.connect(db_file)

    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=4")
    con.execute("PRAGMA preserve_insertion_order=false")

    

    print("\n[1/4] Expanding ground truth into individual pairs...")

    con.execute("""
        CREATE OR REPLACE TABLE ground_truth_pairs AS
        SELECT
            source1_entity_id AS id1,
            TRIM(matched_id) AS id2
        FROM read_csv(
            ?,
            delim='\\t',
            header=true,
            columns={
                'source1_entity_id': 'VARCHAR',
                'matched_entity_ids': 'VARCHAR'
            }
        ),
        UNNEST(string_split(matched_entity_ids, ',')) AS t(matched_id)
        WHERE matched_id IS NOT NULL
          AND TRIM(matched_id) <> ''
    """, [ground_truth])

    total_gt = con.execute("""
        SELECT COUNT(*)
        FROM ground_truth_pairs
    """).fetchone()[0]

    print(f"Total true pairs: {total_gt:,}")


    print("\n[2/4] Reading candidate pairs...")

    candidate_columns = con.execute("""
        SELECT *
        FROM read_csv_auto(
            ?,
            delim='\\t',
            header=true
        )
        LIMIT 0
    """, [candidates]).description

    column_names = [column[0] for column in candidate_columns]

    print("Candidate columns:", column_names)


    if "id1" in column_names and "id2" in column_names:
        id1_col = "id1"
        id2_col = "id2"
    else:
        id1_col = column_names[0]
        id2_col = column_names[1]

        print(
            f"Using first two columns as IDs: "
            f"{id1_col}, {id2_col}"
        )

    con.execute(f"""
        CREATE OR REPLACE TABLE candidate_pairs AS
        SELECT
            CAST("{id1_col}" AS VARCHAR) AS id1,
            CAST("{id2_col}" AS VARCHAR) AS id2
        FROM read_csv_auto(
            ?,
            delim='\\t',
            header=true
        )
    """, [candidates])

    total_candidates = con.execute("""
        SELECT COUNT(*)
        FROM candidate_pairs
    """).fetchone()[0]

    print(f"Total candidate pairs: {total_candidates:,}")


    print("\n[3/4] Checking recovered true pairs...")

    recovered = con.execute("""
        SELECT COUNT(*)
        FROM ground_truth_pairs gt
        INNER JOIN candidate_pairs c
            ON gt.id1 = c.id1
           AND gt.id2 = c.id2
    """).fetchone()[0]

  

    recall = recovered / total_gt if total_gt else 0

    print("\n[4/4] RESULTS")
    print("=" * 60)

    print(f"Total true pairs       : {total_gt:,}")
    print(f"Candidate pairs        : {total_candidates:,}")
    print(f"Recovered true pairs   : {recovered:,}")
    print(f"Blocking recall        : {recall:.4%}")

    print("=" * 60)

  

    covered_s1 = con.execute("""
        SELECT COUNT(DISTINCT id1)
        FROM candidate_pairs
    """).fetchone()[0]

    avg_candidates = con.execute("""
        SELECT AVG(cnt)
        FROM (
            SELECT id1, COUNT(*) AS cnt
            FROM candidate_pairs
            GROUP BY id1
        )
    """).fetchone()[0]

    print(
        f"S1 entities with candidates : "
        f"{covered_s1:,}"
    )

    print(
        f"Avg candidates / S1         : "
        f"{avg_candidates:.2f}"
    )

    print("=" * 60)

    con.close()

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()