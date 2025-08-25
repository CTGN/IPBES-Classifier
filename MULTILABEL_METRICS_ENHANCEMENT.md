# Multi-Label Enhancement for detailed_metrics Function

## Overview
The `detailed_metrics` function in `src/utils/utils.py` has been enhanced to support multi-label classification while maintaining full backward compatibility with binary classification.

## Key Features

### 1. Automatic Detection
The function automatically detects whether the input is for binary or multi-label classification based on the shape of the input arrays:
- **Binary**: 1D arrays or 2D arrays with single column
- **Multi-label**: 2D arrays with multiple columns

### 2. Multi-Label Capabilities

#### Confusion Matrix Visualization
- Creates individual confusion matrices for each label
- Saves multi-label confusion matrix plot as `multilabel_confusion_matrix.png`
- Uses subplot layout for clear visualization of each label's performance

#### Per-Label Metrics
For each label (e.g., "IAS", "SUA", "VA"), the function computes:
- `f1_{label_name}`: F1-score for the specific label
- `recall_{label_name}`: Recall for the specific label  
- `precision_{label_name}`: Precision for the specific label
- `TN_{label_name}`, `FP_{label_name}`, `FN_{label_name}`, `TP_{label_name}`: Confusion matrix components

#### Aggregate Metrics
- `f1_macro`, `f1_micro`, `f1_weighted`: F1-scores with different averaging strategies
- `recall_macro`, `recall_micro`, `recall_weighted`: Recall with different averaging strategies
- `precision_macro`, `precision_micro`, `precision_weighted`: Precision with different averaging strategies
- `accuracy`: Multi-label accuracy
- `MCC`: Matthews Correlation Coefficient
- `kappa`: Cohen's Kappa

#### Score-Based Metrics (when scores provided)
- `roc_auc_macro`, `roc_auc_micro`, `roc_auc_weighted`: ROC-AUC with different averaging
- `roc_auc_{label_name}`: Per-label ROC-AUC scores
- `AP_macro`, `AP_micro`, `AP_weighted`: Average Precision with different averaging
- `AP_{label_name}`: Per-label Average Precision scores
- `NDCG`: Normalized Discounted Cumulative Gain

### 3. Backward Compatibility
The function maintains full compatibility with existing binary classification code:
- Same function signature (with optional new parameters)
- Same output format for binary cases
- All existing metrics preserved

## Usage Examples

### Multi-Label Classification
```python
import numpy as np
from src.utils.utils import detailed_metrics

# Multi-label data (n_samples, n_labels)
predictions = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
labels = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 0]])
scores = np.array([[0.8, 0.2, 0.9], [0.1, 0.7, 0.3], [0.9, 0.6, 0.1]])

# With custom label names
metrics = detailed_metrics(
    predictions=predictions,
    labels=labels, 
    scores=scores,
    label_names=["IAS", "SUA", "VA"]
)

# Without label names (uses Label_0, Label_1, etc.)
metrics = detailed_metrics(predictions=predictions, labels=labels, scores=scores)
```

### Binary Classification (unchanged usage)
```python
# Binary data (n_samples,)
predictions = np.array([1, 0, 1, 0])
labels = np.array([1, 0, 0, 1])
scores = np.array([0.8, 0.2, 0.9, 0.3])

metrics = detailed_metrics(predictions=predictions, labels=labels, scores=scores)
```

## Implementation Details

### Function Structure
The enhanced function uses a dispatcher pattern:
1. `detailed_metrics()`: Main function that detects input type and routes to appropriate handler
2. `_compute_binary_metrics()`: Handles binary classification (original logic)
3. `_compute_multilabel_metrics()`: Handles multi-label classification (new logic)

### Key Dependencies
- `sklearn.metrics.multilabel_confusion_matrix`: For multi-label confusion matrices
- `sklearn.metrics.*`: For various metric computations
- `matplotlib.pyplot`: For visualization
- `numpy`: For array operations

### Error Handling
- Graceful handling of edge cases (all zeros, all ones)
- Robust error handling for score-based metrics
- Warning messages for inconsistent label names
- Zero-division protection in metric computations

## Testing
Two test scripts have been created:
1. `test_multilabel_metrics.py`: Full test with actual dependencies
2. `test_multilabel_simple.py`: Simplified test with mocked dependencies

## Migration Guide
Existing code using `detailed_metrics` requires no changes. To use new multi-label features:

1. **Basic multi-label**: Pass 2D arrays instead of 1D arrays
2. **Custom label names**: Add `label_names` parameter
3. **Access new metrics**: Use the new metric names in the returned dictionary

## Example Metrics Output

### Multi-Label Output Structure
```python
{
    # Per-label metrics
    'f1_IAS': 0.85, 'f1_SUA': 0.72, 'f1_VA': 0.90,
    'recall_IAS': 0.80, 'recall_SUA': 0.75, 'recall_VA': 0.95,
    'precision_IAS': 0.90, 'precision_SUA': 0.70, 'precision_VA': 0.85,
    'TN_IAS': 45, 'FP_IAS': 5, 'FN_IAS': 8, 'TP_IAS': 42,
    
    # Aggregate metrics
    'f1_macro': 0.82, 'f1_micro': 0.84, 'f1_weighted': 0.83,
    'accuracy': 0.78, 'MCC': 0.65, 'kappa': 0.62,
    
    # Score-based metrics (if scores provided)
    'roc_auc_macro': 0.88, 'roc_auc_IAS': 0.92,
    'AP_macro': 0.85, 'NDCG': 0.89
}
```

This enhancement makes the `detailed_metrics` function suitable for the IPBES multi-label classification project while preserving all existing functionality.
