# threshold optimizer 
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report


def optimize_metrics(coeffs, raw_preds, targets):
    # optimize threshold
    if not np.all(np.diff(coeffs) > 0):
        return 10.0
    
    preds = np.digitize(raw_preds, coeffs)
    acc = accuracy_score(targets, preds)
    kap = cohen_kappa_score(targets, preds, weights='quadratic')
    return -(0.5 * acc + 0.5 * kap)


def optimize_thresholds(raw_preds, labels):
    # find best thresholds using nelder-mead method
    initial_thresholds = [0.7, 1.5, 2.5, 3.5]
    
    result = minimize(optimize_metrics, initial_thresholds, 
                      args=(raw_preds, labels), 
                      method='nelder-mead',
                      options={'xatol': 1e-4})
    
    best_thresholds = np.sort(result.x)
    final_optimized_preds = np.digitize(raw_preds, best_thresholds)
    
    return best_thresholds, final_optimized_preds


def evaluate_and_print(labels, final_optimized_preds, target_names=None):
    # print the metrics
    if target_names is None:
        target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    
    print(f"Accuracy: {accuracy_score(labels, final_optimized_preds):.4f}")
    print(f"Kappa: {cohen_kappa_score(labels, final_optimized_preds, weights='quadratic'):.4f}")
    print(classification_report(labels, final_optimized_preds, target_names=target_names))
