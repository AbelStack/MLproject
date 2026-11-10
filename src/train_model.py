import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

EXPECTED_COLS = ['vibration_mm_s', 'rpm', 'temperature_c',
                 'pressure', 'operating_hours', 'label']
FEATURE_COLS = ['vibration_mm_s', 'rpm',
                'temperature_c', 'pressure', 'operating_hours']
VALID_LABELS = {'Normal', 'Warning', 'Fault'}


def validate_schema_and_quality(df, filename):
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
    csv_path = Path(data_dir) / 'abel_normal.csv'
    if not csv_path.exists():
        print(f"[-] Abel (Normal): File not found ({csv_path.name})")
        return None, [f"Missing file: {csv_path.name}"]

    df = pd.read_csv(csv_path)
    if len(df) == 0:
        print(f"[!] Abel (Normal): {csv_path.name} is empty (header only).")
        return None, [f"{csv_path.name}: 0 rows"]

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


def load_warning_data(data_dir='data'):
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


def load_fault_data(data_dir='data'):
    csv_path = Path(data_dir) / 'fitsum_fault.csv'
    if not csv_path.exists():
        print(f"[-] Fitsum (Fault): File not found ({csv_path.name})")
        return None, ["Missing Fault data"]
    df = pd.read_csv(csv_path)
    issues = validate_schema_and_quality(df, csv_path.name)
    if issues:
        print(f"[!] Issues in {csv_path.name}: {issues}")
    else:
        print(f"[+] Fitsum (Fault): {len(df)} observations successfully loaded from {csv_path.name}")
    return df, issues


def evaluate_fault_safety_metrics(y_true, y_pred):
    if 'Fault' in y_true.values:
        f_rec = recall_score(y_true, y_pred, labels=['Fault'], average='micro', zero_division=0)
        f_prec = precision_score(y_true, y_pred, labels=['Fault'], average='micro', zero_division=0)
        f_f1 = f1_score(y_true, y_pred, labels=['Fault'], average='micro', zero_division=0)
        print("\n" + "-" * 65)
        print("CRITICAL SAFETY METRIC - FAULT DETECTION (Fitsum's Module):")
        print("-" * 65)
        print(f"  * Fault Recall:    {f_rec:.4f}  (Ability to catch breakdowns)")
        print(f"  * Fault Precision: {f_prec:.4f}  (Reliability of warnings)")
        print(f"  * Fault F1 Score:  {f_f1:.4f}")


def train_and_evaluate(df):
    X = df[FEATURE_COLS]
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    eval_labels = ['Normal', 'Warning', 'Fault']
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred, labels=eval_labels))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=eval_labels, zero_division=0))
    print(f"Macro F1 Score: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    evaluate_fault_safety_metrics(y_test, y_pred)
    return model


def main():
    print("=" * 65)
    print("PUMP CONDITION CLASSIFICATION - THREE-CLASS MODEL PIPELINE")
    print("=" * 65)

    normal_df, _ = load_normal_data()
    warning_df, _ = load_warning_data()
    fault_df, _ = load_fault_data()

    if normal_df is not None:
        print("\nAbel's Normal Observations Summary:")
        print(f"  Total observations: {len(normal_df)}")
        print(f"  Features loaded: {FEATURE_COLS}")
        print("  Status: Normal data verified and ready for team integration.")

    loaded_dfs = [df for df in [normal_df, warning_df, fault_df] if df is not None]
    if not loaded_dfs:
        print("\n[!] Error: No data files found. Please ensure the 'data/' directory exists and contains the CSV files.")
        return 1

    combined_df = pd.concat(loaded_dfs, ignore_index=True)
    train_and_evaluate(combined_df)

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