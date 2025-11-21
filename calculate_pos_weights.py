"""
Calculate optimal pos_weight values for each class based on class distribution
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CONFIG

def calculate_pos_weights():
    """Calculate pos_weight for each label based on the class distribution."""

    cleaned_data_path = CONFIG['cleaned_dataset_path']
    print(f"Loading cleaned dataset from: {cleaned_data_path}")

    df = pd.read_csv(cleaned_data_path)
    total_instances = len(df)

    labels = ['IAS', 'SUA', 'VA']
    pos_weights = []

    print(f"\nTotal instances: {total_instances}")
    print("\nClass-specific pos_weight calculations:")
    print("pos_weight = num_negative / num_positive")
    print("-" * 60)

    for label in labels:
        num_positive = df[label].sum()
        num_negative = total_instances - num_positive
        pos_weight = num_negative / num_positive

        pos_weights.append(pos_weight)

        print(f"{label}:")
        print(f"  Positive: {num_positive}")
        print(f"  Negative: {num_negative}")
        print(f"  pos_weight: {pos_weight:.4f}")
        print()

    print("=" * 60)
    print(f"Final pos_weight tensor: {pos_weights}")
    print(f"Formatted for code: [{pos_weights[0]:.4f}, {pos_weights[1]:.4f}, {pos_weights[2]:.4f}]")
    print("=" * 60)

    return pos_weights

if __name__ == "__main__":
    calculate_pos_weights()
