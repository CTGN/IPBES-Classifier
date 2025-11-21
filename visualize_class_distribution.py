"""
Visualize class distribution after data cleaning
Generates a pie chart showing the number of instances in each class
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import CONFIG

def visualize_class_distribution():
    """Load cleaned dataset and create pie chart of class distribution."""

    # Load the cleaned dataset
    cleaned_data_path = CONFIG['cleaned_dataset_path']
    print(f"Loading cleaned dataset from: {cleaned_data_path}")

    df = pd.read_csv(cleaned_data_path)
    print(f"Total instances in cleaned dataset: {len(df)}")

    # Define the class labels
    labels = ['IAS', 'SUA', 'VA']

    # Count instances for each class (multi-label, so an instance can have multiple labels)
    class_counts = {}
    for label in labels:
        if label in df.columns:
            class_counts[label] = df[label].sum()
        else:
            print(f"Warning: Label '{label}' not found in dataset columns")
            class_counts[label] = 0

    print("\nClass distribution:")
    for label, count in class_counts.items():
        print(f"  {label}: {count} instances")

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#ff9999', '#66b3ff', '#99ff99']
    explode = (0.05, 0.05, 0.05)  # Slightly separate all slices

    wedges, texts, autotexts = ax.pie(
        class_counts.values(),
        labels=class_counts.keys(),
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        shadow=True,
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)

    # Add title
    ax.set_title('Class Distribution After Data Cleaning\n(Multi-label Dataset)',
                 fontsize=16, weight='bold', pad=20)

    # Add legend with counts
    legend_labels = [f'{label}: {count:,} instances' for label, count in class_counts.items()]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=11)

    plt.tight_layout()

    # Save the figure
    plot_dir = Path(CONFIG['plot_dir'])
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / 'class_distribution_after_cleaning.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPie chart saved to: {output_path}")

    # Show the plot
    plt.show()

if __name__ == "__main__":
    visualize_class_distribution()
