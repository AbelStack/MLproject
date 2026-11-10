"""
Pump Condition Classification - Shared Team ML Pipeline
=========================================================
Assignment Team Responsibilities:
  - Abel:   Normal observations & base pipeline setup (Section 1)
  - Mearwi: Warning observations & warning ingestion code (Section 2)
  - Fitsum: Fault observations & fault safety evaluation code (Section 3)
"""

import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

# Ensure console output handles UTF-8 / non-ASCII gracefully on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

EXPECTED_COLS = ['vibration_mm_s', 'rpm', 'temperature_c',
                 'pressure', 'operating_hours', 'label']
FEATURE_COLS = ['vibration_mm_s', 'rpm',
                'temperature_c', 'pressure', 'operating_hours']
VALID_LABELS = {'Normal', 'Warning', 'Fault'}


# =====================================================================
# SECTION 1: ABEL'S CONTRIBUTION (Normal Data & Base Pipeline Setup)
# =====================================================================

def validate_schema_and_quality(df, filename):
    """
    Abel's data validation helper: checks schema, nulls, labels, and numeric types.
    Reports issues without silently altering or dropping questionable data.
    """
    issues = []
    if list(df.columns) != EXPECTED_COLS:
        issues.append(
            f"{filename}: Schema mismatch. Expected {EXPECTED_COLS}, got {list(df.columns)}")

    nulls = df.isnull().sum()
    if nulls.any():
        issues.append(
            f"{filename}: Missing values found: {dict(nulls[nulls > 0])}")

    invalid_labels = set(df['label'].dropna().unique()) - VALID_LABELS
    if invalid_labels:
        issues.append(
            f"{filename}: Invalid labels found: {invalid_labels}. Allowed: {VALID_LABELS}")

    for col in FEATURE_COLS:
        if col in df.columns:
            non_numeric = pd.to_numeric(
                df[col], errors='coerce').isna() & df[col].notna()
            if non_numeric.any():
                issues.append(
                    f"{filename}: Non-numeric values in '{col}' at row indices {list(df[non_numeric].index)}")

    return issues


def load_normal_data(data_dir='data'):
    """
    Abel's function: Loads and validates Normal pump observations from data/abel_normal.csv.
    """
    csv_path = Path(data_dir) / 'abel_normal.csv'
    if not csv_path.exists():
        print(f"[-] Abel (Normal): File not found ({csv_path.name})")
        return None, [f"Missing file: {csv_path.name}"]

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[!] Abel (Normal): {csv_path.name} is empty (header only).")
        return None, [f"{csv_path.name}: 0 rows"]

    # Verify Abel's file only contains Normal observations
    abnormal_labels = set(df['label'].dropna().unique()) - {'Normal'}
    if abnormal_labels:
        print(
            f"[!] Abel (Normal): Found unexpected labels: {abnormal_labels} (expected only 'Normal')")

    issues = validate_schema_and_quality(df, csv_path.name)
    if issues:
        print(f"[!] Issues in {csv_path.name}: {issues}")
    else:
        print(
            f"[+] Abel (Normal): {len(df)} observations successfully loaded from {csv_path.name}")

    return df, issues


# =====================================================================
# SECTION 2: MEARWI'S CONTRIBUTION (Warning Data Ingestion)
# =====================================================================
def load_warning_data(data_dir='data'):
    """
    Mearwi's function: Loads and validates Warning pump observations.
    """
    csv_candidates = [Path(data_dir) / 'mearwi_warning.csv',
                      Path(data_dir) / 'merawi_warning.csv']
    csv_path = next((p for p in csv_candidates if p.exists()
                    and p.stat().st_size > len(','.join(EXPECTED_COLS))), None)
    if csv_path is None:
        print("[-] Mearwi (Warning): File not found or empty")
        return None, ["Missing Warning data"]
    df = pd.read_csv(csv_path)
    issues = validate_schema_and_quality(df, csv_path.name)
    if issues:
        print(f"[!] Issues in {csv_path.name}: {issues}")
    else:
        print(
            f"[+] Mearwi (Warning): {len(df)} observations successfully loaded from {csv_path.name}")
    return df, issues


def main():
    print("=" * 65)
    print("PUMP CONDITION CLASSIFICATION - THREE-CLASS MODEL PIPELINE")
    print("=" * 65)

    # 1. Abel's Section
    normal_df, _ = load_normal_data()
    # 2. Mearwi's Section
    warning_df, _ = load_warning_data()

    if normal_df is not None:
        print("\nAbel's Normal Observations Summary:")
        print(f"  Total observations: {len(normal_df)}")
        print(f"  Features loaded: {FEATURE_COLS}")
        print("  Status: Normal data verified and ready for team integration.")

    print("\nNext Team Contributions:")
    print("  - Section 2: Mearwi will add load_warning_data() & 2-class integration.")
    print("  - Section 3: Fitsum will add load_fault_data() & 3-class safety evaluation.")

    # Required safety notice
    print("\n" + "=" * 65)
    print("CAUTION - ENGINEERING SAFETY NOTICE:")
    print("=" * 65)
    print("This machine learning model is intended for academic support.")
    print("Model predictions support—NOT replace—engineering inspection,")
    print("standard maintenance procedures, and certified safety protocols.")
    print("=" * 65)

    return 0


if __name__ == '__main__':
    sys.exit(main())
