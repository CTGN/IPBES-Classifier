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
    negative_counts = {}
    total_instances = len(df)

    for label in labels:
        if label in df.columns:
            class_counts[label] = df[label].sum()
            negative_counts[label] = total_instances - df[label].sum()
        else:
            print(f"Warning: Label '{label}' not found in dataset columns")
            class_counts[label] = 0
            negative_counts[label] = total_instances

    print("\nClass distribution:")
    for label in labels:
        pos_count = class_counts[label]
        neg_count = negative_counts[label]
        print(f"  {label}: {pos_count} positive instances ({100*pos_count/total_instances:.1f}%), "
              f"{neg_count} negative instances ({100*neg_count/total_instances:.1f}%)")

    # Combine positive and negative counts for pie chart
    pie_labels = []
    pie_values = []
    pie_colors = []
    pie_explode = []

    # Colors for each class - lighter shade for negative, darker for positive
    color_pairs = [
        ('#ffcccc', '#ff9999'),  # IAS: light red, darker red
        ('#b3d9ff', '#66b3ff'),  # SUA: light blue, darker blue
        ('#ccffcc', '#99ff99')   # VA: light green, darker green
    ]

    for i, label in enumerate(labels):
        # Add positive instances
        pie_labels.append(f'{label} (Positive)')
        pie_values.append(class_counts[label])
        pie_colors.append(color_pairs[i][1])
        pie_explode.append(0.05)

        # Add negative instances
        pie_labels.append(f'{label} (Negative)')
        pie_values.append(negative_counts[label])
        pie_colors.append(color_pairs[i][0])
        pie_explode.append(0.02)

    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 10))

    wedges, texts, autotexts = ax.pie(
        pie_values,
        labels=pie_labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        explode=pie_explode,
        shadow=True,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )

    # Enhance the percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)

    # Add title
    ax.set_title('Class Distribution After Data Cleaning\n(Multi-label Dataset - Positive vs Negative Instances)',
                 fontsize=16, weight='bold', pad=20)

    # Add legend with counts
    legend_labels = []
    for label in labels:
        pos_count = class_counts[label]
        neg_count = negative_counts[label]
        legend_labels.append(f'{label} Positive: {pos_count:,} ({100*pos_count/total_instances:.1f}%)')
        legend_labels.append(f'{label} Negative: {neg_count:,} ({100*neg_count/total_instances:.1f}%)')

    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

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
