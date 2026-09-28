"""Create patient-level stratified 70/15/15 train/val/test splits.

Run once; all other scripts must reuse the generated split CSVs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    labels_csv: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    max_patients: int | None = None,
) -> None:
    """Split patients into train/val/test CSVs, stratified by positive presence.

    Args:
        labels_csv: Path to `stage_2_train_labels.csv`.
        output_dir: Directory to write `train.csv`, `val.csv`, `test.csv`.
        seed: Random seed for reproducibility.
        train_frac: Fraction of patients assigned to train.
        val_frac: Fraction of patients assigned to val (remainder goes to test).
        max_patients: If set, stratified-subsample the patient pool to this
            size before splitting (reduces execution time; ratios unchanged).
    """
    df = pd.read_csv(labels_csv)

    patients = df[["patientId"]].drop_duplicates().reset_index(drop=True)
    has_positive = (
        df.groupby("patientId")["Target"].max().reindex(patients["patientId"]).values
    )
    patients["stratify_key"] = has_positive

    if max_patients is not None and max_patients < len(patients):
        patients, _ = train_test_split(
            patients,
            train_size=max_patients,
            random_state=seed,
            stratify=patients["stratify_key"],
        )
        patients = patients.reset_index(drop=True)

    train_ids, temp_ids = train_test_split(
        patients["patientId"],
        train_size=train_frac,
        random_state=seed,
        stratify=patients["stratify_key"],
    )

    temp_df = patients[patients["patientId"].isin(temp_ids)]
    remaining_frac = 1.0 - train_frac
    val_share_of_temp = val_frac / remaining_frac

    val_ids, test_ids = train_test_split(
        temp_df["patientId"],
        train_size=val_share_of_temp,
        random_state=seed,
        stratify=temp_df["stratify_key"],
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df[df["patientId"].isin(train_ids)].to_csv(output_path / "train.csv", index=False)
    df[df["patientId"].isin(val_ids)].to_csv(output_path / "val.csv", index=False)
    df[df["patientId"].isin(test_ids)].to_csv(output_path / "test.csv", index=False)

    print(f"train patients: {len(train_ids)}")
    print(f"val patients:   {len(val_ids)}")
    print(f"test patients:  {len(test_ids)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Path to stage_2_train_labels.csv")
    parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Subsample to this many patients (stratified) before splitting",
    )
    args = parser.parse_args()

    create_splits(args.labels, args.output_dir, seed=args.seed, max_patients=args.max_patients)
